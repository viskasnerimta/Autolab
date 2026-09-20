#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AutoLab  -  hardened v4
=======================

Autonomous research -> critique -> repair -> curate loop that builds an SFT
(+ preference) dataset from public web evidence, using a local llama.cpp
server as the model.

Pipeline per cycle:
    plan -> search -> fetch/rank evidence -> answer -> automated grounding
    audit -> hostile critic -> repair -> re-audit -> curator -> quality gates
    -> sft.jsonl / preferences.jsonl / research_conclusions.jsonl

Run:
    python autolab.py                       # continuous mode
    python autolab.py --check               # preflight only (server, dirs, search)
    python autolab.py --once                # a single research cycle
    python autolab.py --selftest            # offline tests + stub-server dry run
    python autolab.py --dry-run             # full cycle, nothing written to the dataset
    python autolab.py --verify              # validate/repair existing dataset files
    python autolab.py --stats               # print dataset + lifetime statistics
    python autolab.py --topic "..."         # force one specific research question

Useful environment variables (all optional):
    AUTOLAB_LLAMA_URL         base URL of llama-server   (default http://127.0.0.1:8080)
    AUTOLAB_MODEL_NAME        model id sent to the server
    AUTOLAB_BASE_DIR          output root                (default ./autolab_data)
    AUTOLAB_MEMORY_PATH       optional persona/context text file (first AUTOLAB_MEMORY_MAX_CHARS chars are used)
    AUTOLAB_MEMORY_MAX_CHARS  memory excerpt size limit  (default 6000)
    AUTOLAB_AREAS_FILE        text file, one research area per line (replaces the built-in list)
    AUTOLAB_MODEL_TIMEOUT     seconds allowed per model call (default 600)
    AUTOLAB_EXTRA_PARAMS      JSON merged into every request, e.g. {"chat_template_kwargs":{"enable_thinking":false}}
    AUTOLAB_SEARCH_PROVIDER   auto | serper | ddgs       (auto = serper if key present)
    SERPER_API_KEY            Serper.dev key
    AUTOLAB_CTX               context window you launched llama-server with (auto-detected via /props)
    AUTOLAB_MAX_HOURS / AUTOLAB_MAX_EXAMPLES
    AUTOLAB_SFT_CITATIONS     keep | named | strip       (how [S1] markers are written to sft.jsonl)
    AUTOLAB_LLAMA_API_KEY     bearer token if llama-server runs with --api-key
    AUTOLAB_SFT_SYSTEM        system prompt stored in sft.jsonl records
    AUTOLAB_STRICT_GROUNDING  1 = reject on failed automated grounding audit (default 1)

Exit codes:
    0 ok / clean stop        1 fatal error        2 missing dependency
    3 preflight failed       4 another instance holds the lock
"""
from __future__ import annotations

import argparse
import atexit
import hashlib
import importlib
import io
import ipaddress
import json
import logging
import logging.handlers
import os
import random
import re
import shutil
import signal
import socket
import sys
import tempfile
import threading
import time
import warnings
import zlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeout
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse, urlunparse

try:
    import requests
    from requests.adapters import HTTPAdapter
except ImportError:  # pragma: no cover
    sys.stderr.write("Missing dependency 'requests'. Run: pip install requests beautifulsoup4 pypdf ddgs\n")
    raise SystemExit(2)

try:
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover
    sys.stderr.write("Missing dependency 'beautifulsoup4'. Run: pip install beautifulsoup4\n")
    raise SystemExit(2)

# bs4 emits noisy warnings on short strings / XML served as HTML.
try:
    from bs4 import MarkupResemblesLocatorWarning, XMLParsedAsHTMLWarning

    warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)
    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
except Exception:  # pragma: no cover
    pass

# ---- optional dependencies (each degrades gracefully) ------------------------
try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    import fontTools  # noqa: F401  # quiets pypdf CFF warnings
except ImportError:
    pass

try:
    import trafilatura  # better main-content extraction when installed
except Exception:
    trafilatura = None

DDGS = None
DDGS_PACKAGE = None
for _mod_name in ("ddgs", "duckduckgo_search"):
    try:
        DDGS = importlib.import_module(_mod_name).DDGS
        DDGS_PACKAGE = _mod_name
        break
    except Exception:
        continue

# Suppress the noisy per-glyph warnings from pypdf on LaTeX PDFs.
logging.getLogger("pypdf").setLevel(logging.ERROR)
logging.getLogger("pypdf._reader").setLevel(logging.ERROR)

PIPELINE_VERSION = "4.0-hardened"
PY_MIN = (3, 8)
if sys.version_info < PY_MIN:  # pragma: no cover
    sys.stderr.write("Python %d.%d+ is required (running %s).\n" % (PY_MIN + (sys.version.split()[0],)))
    raise SystemExit(2)


# ============================================================
# CONFIG HELPERS
# ============================================================

def _env_str(name, default=""):
    v = os.environ.get(name)
    return default if v is None or not str(v).strip() else str(v).strip()


def _env_int(name, default):
    v = os.environ.get(name)
    if v is None or not str(v).strip():
        return default
    try:
        return int(float(str(v).strip()))
    except ValueError:
        return default


def _env_float(name, default):
    v = os.environ.get(name)
    if v is None or not str(v).strip():
        return default
    try:
        return float(str(v).strip())
    except ValueError:
        return default


def _env_bool(name, default):
    v = os.environ.get(name)
    if v is None or not str(v).strip():
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "on", "y"}


# ============================================================
# PATHS  (re-pointable with configure_paths(); nothing is created at import)
# ============================================================

BASE_DIR = Path(_env_str("AUTOLAB_BASE_DIR", str(Path.cwd() / "autolab_data")))
DATASET_DIR = BASE_DIR / "dataset"
RESEARCH_DIR = BASE_DIR / "research"
SOURCES_DIR = BASE_DIR / "sources"
REJECTED_DIR = BASE_DIR / "rejected"
EVAL_DIR = BASE_DIR / "evaluations"
LOG_DIR = BASE_DIR / "logs"

SFT_PATH = DATASET_DIR / "sft.jsonl"
PREF_PATH = DATASET_DIR / "preferences.jsonl"
SEEN_PATH = DATASET_DIR / "seen_hashes.json"
STATS_PATH = DATASET_DIR / "stats.json"
ATTEMPTED_QUESTIONS_PATH = DATASET_DIR / "attempted_questions.json"
CONCLUSIONS_PATH = DATASET_DIR / "research_conclusions.jsonl"
RUN_LOG_PATH = LOG_DIR / "autolab.log"
LOCK_PATH = BASE_DIR / "autolab.lock"


def configure_paths(base_dir):
    """Re-point every output path (used by --base-dir and the self-test)."""
    global BASE_DIR, DATASET_DIR, RESEARCH_DIR, SOURCES_DIR, REJECTED_DIR, EVAL_DIR, LOG_DIR
    global SFT_PATH, PREF_PATH, SEEN_PATH, STATS_PATH, ATTEMPTED_QUESTIONS_PATH
    global CONCLUSIONS_PATH, RUN_LOG_PATH, LOCK_PATH
    BASE_DIR = Path(base_dir)
    DATASET_DIR = BASE_DIR / "dataset"
    RESEARCH_DIR = BASE_DIR / "research"
    SOURCES_DIR = BASE_DIR / "sources"
    REJECTED_DIR = BASE_DIR / "rejected"
    EVAL_DIR = BASE_DIR / "evaluations"
    LOG_DIR = BASE_DIR / "logs"
    SFT_PATH = DATASET_DIR / "sft.jsonl"
    PREF_PATH = DATASET_DIR / "preferences.jsonl"
    SEEN_PATH = DATASET_DIR / "seen_hashes.json"
    STATS_PATH = DATASET_DIR / "stats.json"
    ATTEMPTED_QUESTIONS_PATH = DATASET_DIR / "attempted_questions.json"
    CONCLUSIONS_PATH = DATASET_DIR / "research_conclusions.jsonl"
    RUN_LOG_PATH = LOG_DIR / "autolab.log"
    LOCK_PATH = BASE_DIR / "autolab.lock"


def ensure_dirs():
    for d in (DATASET_DIR, RESEARCH_DIR, SOURCES_DIR, REJECTED_DIR, EVAL_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ============================================================
# MODEL ENDPOINT
# ============================================================

def _normalize_llama_base(url):
    """Accept any of http://h:p, .../v1, .../v1/chat/completions and return the base."""
    base = (url or "").strip().rstrip("/")
    if base and "://" not in base:
        base = "http://" + base
    for suffix in ("/v1/chat/completions", "/chat/completions", "/v1"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    return base.rstrip("/")


LLAMA_BASE_URL = _normalize_llama_base(_env_str("AUTOLAB_LLAMA_URL", "http://127.0.0.1:8080"))
LLAMA_API = LLAMA_BASE_URL + "/v1/chat/completions"
LLAMA_MODELS_URL = LLAMA_BASE_URL + "/v1/models"
LLAMA_PROPS_URL = LLAMA_BASE_URL + "/props"
LLAMA_API_KEY = _env_str("AUTOLAB_LLAMA_API_KEY", "")


def configure_endpoint(url):
    """Re-point the model endpoint (used by --url and the self-test stub server)."""
    global LLAMA_BASE_URL, LLAMA_API, LLAMA_MODELS_URL, LLAMA_PROPS_URL
    LLAMA_BASE_URL = _normalize_llama_base(url)
    LLAMA_API = LLAMA_BASE_URL + "/v1/chat/completions"
    LLAMA_MODELS_URL = LLAMA_BASE_URL + "/v1/models"
    LLAMA_PROPS_URL = LLAMA_BASE_URL + "/props"
    return LLAMA_BASE_URL

MODEL_NAME = os.environ.get("AUTOLAB_MODEL_NAME", "local-model")
_MEMORY_ENV = os.environ.get("AUTOLAB_MEMORY_PATH", "").strip()
MEMORY_PATH = Path(_MEMORY_ENV) if _MEMORY_ENV else None   # optional persona/context text file
MEMORY_MAX_CHARS = _env_int("AUTOLAB_MEMORY_MAX_CHARS", 6000)
AUTOLAB_MEMORY = ""  # filled by load_memory() at startup

# Which pipeline stages get the memory excerpt in their system prompt.
# Planner / critic / curator emit strict JSON; persona text in front of them only
# costs context and dilutes the "JSON only" instruction.
MEMORY_ROLES = {
    r.strip() for r in _env_str("AUTOLAB_MEMORY_ROLES", "candidate,repair,conclusion").split(",") if r.strip()
}

# ============================================================
# PERFORMANCE / QUALITY
# ============================================================
# Quality gates. Tune these instead of editing save_example().
MIN_QUALITY_SCORE = _env_float("AUTOLAB_MIN_QUALITY", 8.0)
MIN_ACCURACY_SCORE = _env_float("AUTOLAB_MIN_ACCURACY", 8.0)
MIN_CITATION_SCORE = 7.5
MIN_RELEVANCE_SCORE = 7.5
MIN_CONFIDENCE = 0.70
MAX_REQUIRED_FIXES = 2
MAX_ISSUES = 4

# Minimum source quality (see classify_source) required for at least one
# cited source in an accepted answer. 7+ = .edu, arXiv, DOI, gov, peer-
# reviewed. 10 = NASA/mil/gov primary. 4 = generic news/blog.
MIN_CITED_SOURCE_QUALITY = 7

# If true, reject when the highest-quality cited source is below
# MIN_CITED_SOURCE_QUALITY. If false, only log a warning.
ENFORCE_MIN_CITED_SOURCE_QUALITY = True

# Require at least this many distinct cited sources in an accepted answer.
MIN_CITED_SOURCES = 2

# Require cited sources to come from at least this many distinct BASE domains
# (bbc.co.uk and theguardian.co.uk are two domains, not one "co.uk").
MIN_CITED_DOMAINS = 2

# Community / mid-tier requirement. The answer must cite at least this many
# sources whose tier is >= 5 but BELOW MIN_CITED_SOURCE_QUALITY (forums, Q&A,
# quality press). The original check counted tier >= 5, which any tier-7+
# source already satisfied, so it never constrained anything. It now enforces
# what its comment always described: engage with what people report, not only
# what institutions publish.
MIN_TIER5_SOURCES = 1
REQUIRE_TIER5_SOURCE = True
TIER5_MUST_BE_NON_INSTITUTIONAL = _env_bool("AUTOLAB_TIER5_NON_INSTITUTIONAL", True)

# If the planner picks the same area this many times in the last N cycles,
# forcibly rotate to a different area.
STUCK_AREA_THRESHOLD = 3
STUCK_AREA_WINDOW = 6

# Save-time / plan-time novelty guards.
AREA_SATURATION_WINDOW = 12
AREA_SATURATION_LIMIT = 2      # reject when the area already appeared this many times in the window
NEAR_DUP_JACCARD = 0.70
SUBJECT_REPEAT_HITS = 3

# How many recent questions to feed the planner as "do not repeat".
PLANNER_AVOID_HISTORY = 40

# Declared context target. The real value is read from llama-server /props at
# startup and the smaller of the two wins; on HTTP 400 "exceeds context" the
# script also learns the true n_ctx from the error body.
CONTEXT_SIZE = _env_int("AUTOLAB_CTX", 131072)
CHARS_PER_TOKEN = 3.0          # conservative (English + PDF/LaTeX noise)
PROMPT_SAFETY_TOKENS = 1024

# One candidate is much faster than 3 near-identical 5K-token generations.
# Quality is protected by grounding checks + critic + repair + curator.
CANDIDATES_PER_TASK = _env_int("AUTOLAB_CANDIDATES", 1)
MAX_REPAIR_ROUNDS = _env_int("AUTOLAB_REPAIR_ROUNDS", 1)

RESULTS_PER_QUERY = 12
MAX_QUERIES = 7
MAX_FETCHED_SOURCES = 24
MAX_SOURCE_CHARS = 18000        # per-source cap inside a prompt

PLANNER_MAX_TOKENS = 2200
CANDIDATE_MAX_TOKENS = 4096
CRITIC_MAX_TOKENS = 1500
CURATOR_MAX_TOKENS = 1000
CONCLUSION_MAX_TOKENS = 600

SEARCH_TEMPERATURE = 0.15
GENERATION_TEMPERATURE = 0.60
CRITIC_TEMPERATURE = 0.15
CURATOR_TEMPERATURE = 0.15

MODEL_TIMEOUT = _env_int("AUTOLAB_MODEL_TIMEOUT", 600)   # seconds per model call
MODEL_CONNECT_TIMEOUT = 15
MODEL_RETRIES = _env_int("AUTOLAB_MODEL_RETRIES", 3)   # attempts per call
SERVER_WAIT_SECONDS = _env_int("AUTOLAB_SERVER_WAIT", 300)
SLEEP_BETWEEN_CYCLES = 5
MAX_EXAMPLES = _env_int("AUTOLAB_MAX_EXAMPLES", 0)
MAX_RUNTIME_SECONDS = int(_env_float("AUTOLAB_MAX_HOURS", 4.0) * 3600)  # 4 hours
STATS_EVERY = 10
MAX_CONSECUTIVE_FAILURES = _env_int("AUTOLAB_MAX_FAILURES", 25)
FAILURE_BACKOFF_MAX = 300
MIN_FREE_DISK_MB = _env_int("AUTOLAB_MIN_FREE_MB", 500)
MAX_RECORD_FILES = _env_int("AUTOLAB_MAX_RECORD_FILES", 4000)   # research/ and rejected/ retention

# Answer sanity.
MIN_ANSWER_CHARS = 800          # hard floor for saving (prompt asks for 1500)
RETRY_ANSWER_CHARS = 1000       # below this a stricter retry is attempted
MIN_CITED_CLAIMS = 3            # cited sentences required (matches CANDIDATE_SYSTEM)
GROUNDING_MAX_FLAGGED_FRACTION = 0.34
WRITE_CONCLUSIONS = _env_bool("AUTOLAB_WRITE_CONCLUSIONS", True)

# How [S1] markers are written into sft.jsonl:
#   keep  -> unchanged (original behaviour)
#   named -> [nasa.gov; reddit.com]  (no dangling ids, no invented titles)
#   strip -> markers removed
# NOTE: the SFT prompt does not contain the sources, so "keep" teaches the model
# to emit [S1] markers that point at nothing. "named" or "strip" avoids that.
SFT_CITATION_MODE = _env_str("AUTOLAB_SFT_CITATIONS", "keep").lower()
if SFT_CITATION_MODE not in {"keep", "named", "strip"}:
    SFT_CITATION_MODE = "keep"

# ---------------- SEARCH ----------------
# "auto": Serper when SERPER_API_KEY is set, otherwise ddgs scraping backends.
SEARCH_PROVIDER = os.environ.get("AUTOLAB_SEARCH_PROVIDER", "auto").strip().lower() or "auto"
SERPER_API_KEY = os.environ.get("SERPER_API_KEY", "").strip()
SERPER_ENDPOINT = "https://google.serper.dev/search"
SEARCH_FALLBACK_TO_DDGS = True

WEB_SEARCH_BACKENDS = [
    b.strip() for b in _env_str(
        "AUTOLAB_SEARCH_BACKENDS", "brave,yahoo,startpage,duckduckgo,google"
    ).split(",") if b.strip()
]
BACKEND_COOLDOWN_SECONDS = 900          # doubled per repeated hard block, max 1h
WEB_SEARCH_DELAY_MIN = 0.25
WEB_SEARCH_DELAY_MAX = 0.7

# ---------------- FETCH ----------------
WEB_FETCH_RETRIES = 3
WEB_FETCH_TIMEOUT = 20                  # read timeout (s)
WEB_FETCH_CONNECT_TIMEOUT = 10
FETCH_TOTAL_TIMEOUT = 45                # hard wall-clock per URL (slow-drip servers)
FETCH_DEADLINE_SECONDS = 180            # wall-clock for the whole source-collection fetch phase
FETCH_WORKERS = _env_int("AUTOLAB_FETCH_WORKERS", 8)
MAX_REDIRECTS = 6
MAX_DOWNLOAD_BYTES_HTML = 6 * 1024 * 1024
MAX_DOWNLOAD_BYTES_PDF = 25 * 1024 * 1024
PDF_MAX_PAGES = 40
FETCH_KEEP_CHARS = 60000                # kept in memory; prompt windows chosen later
MIN_SOURCE_CHARS = 300
MIN_FETCH_TIER = 2                      # tier-1 (x.com, youtube, quora...) is never worth fetching
MAX_SOURCES_PER_DOMAIN = 3
COMMUNITY_SOURCE_QUOTA = 4              # slots reserved for mid-tier sources when they exist
MIN_SOURCE_RELEVANCE = 0.10
MIN_SOURCES_FOR_CYCLE = 2
REQUIRE_LATIN_TEXT = True
SKIP_DNS_CHECK = _env_bool("AUTOLAB_SKIP_DNS_CHECK", False)
FETCH_USER_AGENT = _env_str(
    "AUTOLAB_FETCH_UA",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/128 Safari/537.36",
)

# Critic/repair see real evidence excerpts (the original critic only saw titles).
CRITIC_EXCERPT_CHARS_PER_SOURCE = 1800
CRITIC_EXCERPT_TOTAL_CHARS = 24000
REPAIR_EXCERPT_CHARS_PER_SOURCE = 3000
REPAIR_EXCERPT_TOTAL_CHARS = 40000

STOP_EVENT = threading.Event()
# ============================================================
# RESEARCH AREAS
# ============================================================

CORE_AREAS = [
    # Science and technology
    "astronomy and astrobiology",
    "exoplanets and habitability",
    "space science",
    "scientific papers",
    "new scientific discoveries",
    "contradictory evidence and competing explanations",
    "mathematics and formal methods",
    "physics and fundamental constants",
    "chemistry and materials science",
    "biology and genomics",
    "earth and climate science",
    "neuroscience and cognition",
    "linguistics and language models",
    "computer science and algorithms",
    "engineering and failure analysis",
    "medicine and epidemiology",
    # Society and history
    "declassified government documents",
    "government records and hearings",
    "institutional and policy history",
    "economics and game theory",
    "law and constitutional analysis",
    "philosophy of science",
    "history of science and technology",
    "archaeology and prehistory",
]


def _load_areas_file():
    """Optional: AUTOLAB_AREAS_FILE = text file, one research area per line ('#' comments allowed)."""
    path = _env_str("AUTOLAB_AREAS_FILE", "")
    if not path:
        return None
    try:
        lines = [ln.strip() for ln in Path(path).read_text(encoding="utf-8").splitlines()]
        areas = [ln for ln in lines if ln and not ln.startswith("#")]
        return areas or None
    except Exception as e:
        sys.stderr.write("Could not read AUTOLAB_AREAS_FILE %s: %s\n" % (path, e))
        return None


_CUSTOM_AREAS = _load_areas_file()
if _CUSTOM_AREAS:
    CORE_AREAS = _CUSTOM_AREAS

# Areas that are closely related. If the planner gets stuck on one of them, the whole
# cluster is excluded from the next forced rotation.
RELATED_AREA_CLUSTERS = [
    ["declassified government documents", "government records and hearings"],
    ["astronomy and astrobiology", "exoplanets and habitability", "space science"],
]

GENERAL_AREAS = [
    "reasoning", "coding", "debugging", "mathematics", "science",
    "history", "research methodology", "source evaluation",
    "critical thinking", "instruction following", "long-form analysis",
]

# Words that mark a query as community/forum-targeted.
COMMUNITY_QUERY_MARKERS = (
    "reddit", "hacker news", "stack exchange", "stackexchange", "stack overflow",
    "users reported", "first-hand", "first hand", "forum", "community consensus",
    "mailing list", "discussion thread",
)

# ============================================================
# PROMPTS
# ============================================================

PLANNER_SYSTEM = r"""
You are AutoLab's autonomous research planner.
Choose one concrete, high-value research question the model should learn.
Never create vague topic questions. Target a specific event, person, program,
document, experiment, scientific finding, technical issue, failure mode,
benchmark, historical episode, disputed claim, dataset, or date range.
Use public evidence.

For controversial or disputed subjects, investigate both supporting and challenging
evidence. Do not assume a conclusion the evidence does not support.

Prefer primary documents, official government sources, FOIA releases,
congressional records, hearing transcripts, archives, original papers,
peer-reviewed research, official datasets, technical documentation and source
code.

For each research question, include at least one search query aimed at
community / forum evidence.
IMPORTANT: Never use the "site:" operator in any query.
Use plain descriptive terms instead. Examples:
  - "<topic> reddit"
  - "<topic> hacker news discussion"
  - "<topic> stack exchange"
  - "<topic> forum discussion"
  - "<topic> users reported"
  - "<topic> first-hand report"
  - "<topic> community consensus"

Community queries surface what people actually observed, reported, or
disputed. Treat their output as testimony, not fact.

Return JSON only:
{
 "area": "...",
 "research_question": "...",
 "search_queries": ["...", "..."],
 "learning_goal": "...",
 "evidence_requirements": ["...", "..."],
 "counterargument_target": "...",
 "knowledge_gap": "...",
 "risk_notes": ["..."]
}
Return ONLY the JSON object. Do NOT write any prose, explanation, or commentary.
The response must start with "{" and end with "}".
Generate 5-7 materially different search queries.
"""

CANDIDATE_SYSTEM = r"""
You are AutoLab's research-answer generator.
Use ONLY supplied research material as evidence for factual claims.
Answer clearly, precisely and information-densely.
Every important factual claim needs [S1], [S2], etc. Put ONE source id per
bracket and write several brackets side by side when needed: [S1][S2]. Never
write [S1, S2]. Put the citation at the end of the sentence it supports.
Distinguish FACT, DOCUMENTED CLAIM, ALLEGATION, INTERPRETATION, HYPOTHESIS,
and SPECULATION when applicable.
Do not manufacture citations or outside facts.
Copy numbers, dates and quotations exactly as they appear in the cited source.
Only put text in quotation marks if it appears verbatim in the cited source.
For controversial or extraordinary claims, examine both supporting and conventional
explanations. Distinguish observation from interpretation and testimony from
independently verified physical evidence.

STRUCTURE REQUIREMENTS (mandatory):
- Minimum 1500 characters.
- First sentence directly answers the question.
- At least 3 distinct factual claims, each with a [S1]-style citation.
- Include one explicit sentence about uncertainty or counter-evidence.
- If a claim cannot be sourced, say so instead of asserting it.
- Do not pad. If the sources genuinely do not support an answer,
  say that explicitly and cite the closest available source.
- Write in English only. Do not add sign-offs, translations, or repeated text.

"""

CRITIC_SYSTEM = r"""
You are AutoLab's hostile fact-checking critic.
Try to BREAK the proposed answer.
Check every important claim against the supplied EVIDENCE EXCERPTS, citation
correctness, source quality, contradictions, dates/numbers, unsupported outside
knowledge, missing counter-evidence, reasoning errors, misleading certainty and
relevance.
The excerpts are verbatim but partial: a claim that is missing from the excerpt
of its cited source is suspicious, and any number, date or quotation that does
not appear in the cited source's excerpt must be listed in unsupported_claims.
If an AUTOMATED GROUNDING PRE-CHECK is provided, verify each flagged item first.
For extraordinary claims, an unexplained observation does not establish a specific
explanation; testimony does not automatically establish the explanation; distinguish physical,
sensor, documentary and testimonial evidence.

TESTIMONY VS FACT (critical rule):
- Community platforms (Reddit, Hacker News, Stack Exchange, project
  forums, mailing-list mirrors) are evidence of what people REPORTED,
  OBSERVED, or CLAIMED. They are NOT evidence that a claim is true.
- An answer that cites a Reddit thread to support "X is true" must be
  flagged as unsupported. An answer that cites the same thread to support
  "users on r/foo reported X" is acceptable.
- If the answer treats a Reddit / forum / Stack Exchange post as if it
  were a primary document, peer-reviewed paper, or government record,
  lower accuracy_score and list it in unsupported_claims.
- If the answer cites a forum thread AND independently corroborates the
  same claim with a tier-7+ source, that is acceptable and should not be
  penalized.
- Do NOT penalize an answer merely for citing a forum. Penalize it for
  treating forum testimony as established fact.

SOURCE QUALITY SCORING RULES (apply strictly):
- A claim cited to a blog, news aggregator, opinion piece, LinkedIn post,
  or fan wiki should NOT score above 6 on citation_score unless the same
  claim is independently corroborated by a primary, peer-reviewed, or
  government source in the same answer.
- Popular retellings of what a named expert said (e.g. a news article
  summarizing a hearing) are NOT the same as the expert's own testimony,
  paper, or interview transcript. If the answer cites the retelling and
  not the original, flag it as an issue.
- Scores of 9 or 10 on citation_score require that the load-bearing
  factual claims cite primary/authoritative sources (gov, .edu, DOI,
  peer-reviewed, FOIA reading room, congressional record, court filing,
  official dataset, technical documentation). One primary source among
  ten blogs is not enough for a 9.
- If the answer uses words like "documented", "verified", "confirmed",
  or "evidence" but the supporting citation is a blog or aggregator,
  list it in unsupported_claims.
- Do not give a 10 on citation_score to any answer that relies on Salon,
  CBN, patheos, Medium, Substack, LinkedIn, Quora, or similar outlets
  for its key claims.
Scores are numbers from 0 to 10 (confidence is 0.0 to 1.0). Be honest: any
zeros in a template are placeholders, not suggested values.
Return ONLY the JSON object. Do NOT write any prose, explanation, or commentary. Your response MUST start with the character { and end with }. Do not write chain-of-thought reasoning. Do not write bullet points. If you have notes, do not include them. The very first character of your response must be {.
"""

CURATOR_SYSTEM = r"""
You are AutoLab's final dataset curator.
Be extremely strict, but do not reject merely because a topic is controversial.
ACCEPT when the question is specific, the sources directly address it, the
answer is supported and cited, uncertainty is preserved, and the example has
real learning value.
REJECT generic filler, unsupported claims, citation mismatches, irrelevant
sources, and conclusions stronger than evidence.
Return valid JSON only.
"""
# ============================================================
# LOGGING
# ============================================================

log = logging.getLogger("autolab")
if not log.handlers:
    log.addHandler(logging.NullHandler())
log.setLevel(logging.INFO)


def setup_logging(console_only=False):
    """File (rotating) + console logging. Safe to call more than once."""
    for h in list(log.handlers):
        log.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    if not console_only:
        try:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            fh = logging.handlers.RotatingFileHandler(
                RUN_LOG_PATH, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
            )
            fh.setFormatter(fmt)
            log.addHandler(fh)
        except Exception as e:  # unwritable log dir must not kill the run
            sys.stderr.write("WARNING: file logging disabled (%s)\n" % e)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    log.addHandler(sh)
    log.setLevel(logging.INFO)
    log.propagate = False


session = requests.Session()
session.headers.update({"User-Agent": "AutoLab/4.0 (research collector)"})
session.mount("http://", HTTPAdapter(pool_connections=4, pool_maxsize=8))
session.mount("https://", HTTPAdapter(pool_connections=4, pool_maxsize=8))


class ModelError(RuntimeError):
    """Model endpoint failed after retries (transport / HTTP error)."""


class StopRequested(Exception):
    """Raised between pipeline stages when a graceful stop was requested."""


def check_stop():
    if STOP_EVENT.is_set():
        raise StopRequested()


def sleep_interruptible(seconds):
    STOP_EVENT.wait(max(0.0, seconds))


# ============================================================
# SMALL UTILITIES
# ============================================================

def now_iso():
    return datetime.now(timezone.utc).isoformat()


_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SURR_RE = re.compile("[\ud800-\udfff]")


def clean_text(s):
    """Drop NULs / control chars / lone surrogates (they break JSON files and UTF-8 writes)."""
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    s = _SURR_RE.sub("", s)
    s = _CTRL_RE.sub("", s)
    return s.replace("\ufeff", "").replace("\u200b", "")


def safe_float(v, default=0.0, lo=None, hi=None):
    try:
        if isinstance(v, bool):
            x = 1.0 if v else 0.0
        elif isinstance(v, (int, float)):
            x = float(v)
        elif isinstance(v, str):
            m = re.search(r"-?\d+(?:\.\d+)?", v)
            x = float(m.group()) if m else float(default)
        else:
            x = float(default)
    except Exception:
        x = float(default)
    if x != x or x in (float("inf"), float("-inf")):
        x = float(default)
    if lo is not None:
        x = max(lo, x)
    if hi is not None:
        x = min(hi, x)
    return x


def safe_bool(v, default=False):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v != 0
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"true", "yes", "y", "1", "pass", "passed", "accept", "accepted"}:
            return True
        if s in {"false", "no", "n", "0", "fail", "failed", "reject", "rejected", ""}:
            return False
    return default


def as_str_list(v, limit=20, maxlen=400):
    if v is None:
        return []
    if isinstance(v, str):
        v = [v]
    if isinstance(v, dict):
        v = list(v.values())
    if not isinstance(v, (list, tuple)):
        return []
    out = []
    for item in v:
        if isinstance(item, (dict, list)):
            item = json.dumps(item, ensure_ascii=False)
        s = clean_text(item).strip()
        if s:
            out.append(s[:maxlen])
        if len(out) >= limit:
            break
    return out


def sha256_text(text):
    return hashlib.sha256(text.strip().lower().encode("utf-8", "replace")).hexdigest()


def normalize_question(q):
    """Collapse a question to a comparable form for dedupe."""
    if not q:
        return ""
    q = str(q).lower().strip()
    q = re.sub(r"\s+", " ", q)
    q = re.sub(r"[^a-z0-9 ]+", "", q)
    return q


def token_set(s):
    return set(re.findall(r"[a-z0-9]+", (s or "").lower()))


def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def latin_ratio(text):
    """Share of alphabetic characters that are Latin (0..1)."""
    letters = [c for c in text[:20000] if c.isalpha()]
    if not letters:
        return 0.0
    latin = sum(1 for c in letters if c < "\u0250")
    return latin / len(letters)


def alnum_ratio(text):
    sample = text[:20000]
    if not sample:
        return 0.0
    return sum(1 for c in sample if c.isalnum() or c.isspace()) / len(sample)


def atomic_write_text(path, text):
    """Write-then-rename so a crash can never leave a half-written state file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("%s.tmp-%d" % (path.name, os.getpid()))
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
        os.replace(tmp, path)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass


_APPEND_LOCK = threading.Lock()


def _ensure_trailing_newline(path):
    """A killed process can leave a partial last line; isolate it so the next record stays valid."""
    try:
        if path.exists() and path.stat().st_size > 0:
            with path.open("rb+") as f:
                f.seek(-1, os.SEEK_END)
                if f.read(1) != b"\n":
                    f.seek(0, os.SEEK_END)
                    f.write(b"\n")
                    log.warning("Repaired missing trailing newline in %s", path.name)
    except OSError as e:
        log.warning("Could not verify trailing newline of %s: %s", path, e)


def append_jsonl(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(obj, ensure_ascii=False, default=str) + "\n"
    line = _SURR_RE.sub("", line)
    with _APPEND_LOCK:
        _ensure_trailing_newline(path)
        with path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(line)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass


def read_tail_lines(path, n, max_bytes=8 * 1024 * 1024):
    """Last n non-empty lines without reading the whole file."""
    path = Path(path)
    try:
        size = path.stat().st_size
    except OSError:
        return []
    if size == 0:
        return []
    block, data, pos = 65536, b"", size
    try:
        with path.open("rb") as f:
            while pos > 0 and data.count(b"\n") <= n and len(data) < max_bytes:
                step = min(block, pos)
                pos -= step
                f.seek(pos)
                data = f.read(step) + data
    except OSError:
        return []
    lines = data.decode("utf-8", errors="replace").splitlines()
    if pos > 0 and lines:
        lines = lines[1:]  # first line may be cut in half
    return [ln for ln in lines if ln.strip()][-n:]


_COUNT_CACHE = {}


def dataset_count(path):
    path = Path(path)
    try:
        st = path.stat()
    except OSError:
        return 0
    sig = (st.st_size, st.st_mtime_ns)
    hit = _COUNT_CACHE.get(str(path))
    if hit and hit[0] == sig:
        return hit[1]
    n = 0
    try:
        with path.open("rb") as f:
            for line in f:
                if line.strip():
                    n += 1
    except OSError:
        return 0
    if len(_COUNT_CACHE) > 64:
        _COUNT_CACHE.clear()
    _COUNT_CACHE[str(path)] = (sig, n)
    return n


def prune_dir(directory, keep):
    """Keep only the newest `keep` files in a directory."""
    try:
        files = sorted(Path(directory).glob("*.json"), key=lambda p: p.name)
        for p in files[: max(0, len(files) - keep)]:
            try:
                p.unlink()
            except OSError:
                pass
    except OSError:
        pass


def free_disk_mb(path):
    try:
        probe = Path(path)
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        return shutil.disk_usage(probe).free / (1024 * 1024)
    except Exception:
        return float("inf")


# ============================================================
# MODEL-OUTPUT CLEANING + JSON EXTRACTION
# ============================================================

_SPECIAL_TOKENS = (
    "<end_of_turn>", "<start_of_turn>", "</s>", "<|end_of_turn|>", "<|im_end|>",
    "<|im_start|>", "<|eot_id|>", "<|endoftext|>",
)
_THINK_RE = re.compile(r"<think>.*?</think>", re.S | re.I)


def clean_model_text(text):
    """Strip control chars, special tokens and <think> blocks from model output."""
    text = clean_text(text)
    if not text:
        return ""
    text = _THINK_RE.sub("", text)
    lower = text.lower()
    idx = lower.find("<think>")
    if idx != -1:  # unclosed: the model ran out of budget while thinking
        text = text[:idx]
        lower = text.lower()
    idx = lower.rfind("</think>")
    if idx != -1:  # stray closer: everything before it is leaked reasoning
        text = text[idx + len("</think>"):]
    for tok in _SPECIAL_TOKENS:
        text = text.replace(tok, "")
    return text.strip()


def _iter_json_objects(text):
    """Yield every complete top-level {...} block (string/escape aware)."""
    i, n = 0, len(text)
    while True:
        start = text.find("{", i)
        if start == -1:
            return
        depth, in_str, esc, end = 0, False, False, None
        for j in range(start, n):
            ch = text[j]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = j
                    break
        if end is None:
            return
        yield text[start:end + 1]
        i = end + 1


def _extract_first_json_object(text):
    """Kept for compatibility: first complete JSON object by brace counting."""
    for obj in _iter_json_objects(text or ""):
        return obj
    return None


def _close_truncated_json(s):
    """Best-effort repair of JSON cut off by max_tokens."""
    stack, in_str, esc = [], False, False
    for ch in s:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]" and stack and stack[-1] == ch:
            stack.pop()
    out = s
    if in_str:
        out += '"'
    out = re.sub(r",\s*$", "", out.rstrip())
    if out.endswith(":"):
        out += " null"
    return out + "".join(reversed(stack))


def _try_parse(s):
    variants = [s, re.sub(r",\s*([}\]])", r"\1", s)]
    py = re.sub(r"(:\s*)True\b", r"\1true", variants[1])
    py = re.sub(r"(:\s*)False\b", r"\1false", py)
    py = re.sub(r"(:\s*)None\b", r"\1null", py)
    variants.append(py)
    for cand in variants:
        try:
            return json.loads(cand, strict=False)
        except Exception:
            continue
    return None


def safe_json_loads(text, fallback=None, want_keys=None):
    """Parse model output into a dict. Never raises; returns `fallback` on failure.

    Handles code fences, prose around the JSON, several objects in one reply
    (prefers the one containing `want_keys`), trailing commas, Python literals,
    and JSON truncated by max_tokens.
    """
    fb = dict(fallback) if isinstance(fallback, dict) else ({} if fallback is None else fallback)
    if not text:
        return fb
    text = clean_model_text(text)
    if not text:
        return fb

    cleaned = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned)
    v = _try_parse(cleaned)
    if isinstance(v, dict):
        return v
    if isinstance(v, list):
        for item in v:
            if isinstance(item, dict):
                return item

    candidates = []
    for obj in _iter_json_objects(text):
        v = _try_parse(obj)
        if isinstance(v, dict):
            candidates.append(v)
    if candidates:
        if want_keys:
            wk = set(want_keys)
            candidates.sort(key=lambda d: len(wk & set(d.keys())), reverse=True)
        return candidates[0]

    start = text.find("{")
    if start != -1:
        frag = text[start:]
        for cut in (len(frag), frag.rfind(",")):
            if cut <= 0:
                continue
            v = _try_parse(_close_truncated_json(frag[:cut]))
            if isinstance(v, dict) and v:
                log.info("safe_json_loads: recovered truncated JSON (%d keys)", len(v))
                return v

    log.warning("safe_json_loads FAILED. First 500 chars:\n%s", text[:500])
    return fb


# ============================================================
# PERSISTENT STATE  (seen hashes, attempted questions, lifetime stats)
# ============================================================

SEEN_HASHES = set()
ATTEMPTED_QUESTIONS = []
ACCEPTED_NORMALIZED = set()
ACCEPTED_TOKENSETS = []
LIFETIME = {"cycles": 0, "accepted": 0, "rejected": 0, "skipped": 0, "errors": 0, "reasons": {}}
RUN_COUNTS = Counter()


def _quarantine(path, why):
    try:
        dest = path.with_name("%s.corrupt-%s" % (path.name, datetime.now().strftime("%Y%m%d-%H%M%S")))
        os.replace(path, dest)
        log.error("%s is unreadable (%s). Moved to %s and starting fresh.", path.name, why, dest.name)
    except OSError as e:
        log.error("%s is unreadable (%s) and could not be moved: %s", path.name, why, e)


def load_json_file(path, default, expected_type):
    path = Path(path)
    if not path.exists():
        return default
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception as e:
        _quarantine(path, e)
        return default
    if not isinstance(data, expected_type):
        _quarantine(path, "unexpected JSON type %s" % type(data).__name__)
        return default
    return data


def load_seen():
    return {str(x) for x in load_json_file(SEEN_PATH, [], list)}


def save_seen(seen):
    atomic_write_text(SEEN_PATH, json.dumps(sorted(seen), ensure_ascii=False, indent=2))


def load_attempted_questions():
    items = load_json_file(ATTEMPTED_QUESTIONS_PATH, [], list)
    return [a for a in items if isinstance(a, dict)]


def save_attempted_questions(items):
    # Keep only the most recent 200 to bound file size.
    atomic_write_text(
        ATTEMPTED_QUESTIONS_PATH,
        json.dumps(items[-200:], ensure_ascii=False, indent=2),
    )


def first_user_message(record):
    try:
        for m in record.get("messages", []):
            if isinstance(m, dict) and m.get("role") == "user":
                return str(m.get("content") or "")
    except Exception:
        pass
    return ""


def rebuild_accepted_index(limit=5000):
    """Index accepted questions so novelty checks survive the 200-item attempted cap."""
    global ACCEPTED_NORMALIZED, ACCEPTED_TOKENSETS
    ACCEPTED_NORMALIZED, ACCEPTED_TOKENSETS = set(), []
    if not SFT_PATH.exists():
        return
    questions = []
    try:
        with SFT_PATH.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    q = first_user_message(json.loads(line))
                except Exception:
                    continue
                if q:
                    questions.append(q)
    except OSError as e:
        log.warning("Could not index accepted questions: %s", e)
        return
    for q in questions[-limit:]:
        ACCEPTED_NORMALIZED.add(normalize_question(q))
        ACCEPTED_TOKENSETS.append((q, token_set(q)))


def load_state():
    global SEEN_HASHES, ATTEMPTED_QUESTIONS
    SEEN_HASHES = load_seen()
    ATTEMPTED_QUESTIONS = load_attempted_questions()
    for a in ATTEMPTED_QUESTIONS:
        a.setdefault("id", "legacy-%s" % sha256_text(str(a.get("question", "")))[:10])
    stats = load_json_file(STATS_PATH, {}, dict)
    life = stats.get("lifetime")
    if isinstance(life, dict):
        for k in ("cycles", "accepted", "rejected", "skipped", "errors"):
            LIFETIME[k] = int(safe_float(life.get(k), 0))
        reasons = life.get("reasons")
        LIFETIME["reasons"] = {str(k): int(safe_float(v, 0)) for k, v in reasons.items()} if isinstance(reasons, dict) else {}
    rebuild_accepted_index()


def load_memory():
    """Read the persona/memory excerpt. Never fatal."""
    global AUTOLAB_MEMORY
    AUTOLAB_MEMORY = ""
    if MEMORY_PATH is None:
        log.info("No memory file configured (optional: --memory PATH or AUTOLAB_MEMORY_PATH).")
        return
    try:
        if MEMORY_PATH.is_file():
            AUTOLAB_MEMORY = clean_text(MEMORY_PATH.read_text(encoding="utf-8", errors="replace"))[:MEMORY_MAX_CHARS]
            log.info("Memory excerpt loaded: %d chars from %s", len(AUTOLAB_MEMORY), MEMORY_PATH)
        else:
            log.warning("Memory file not found (%s). Continuing without a memory excerpt.", MEMORY_PATH)
    except Exception as e:
        log.warning("Could not read memory file %s: %s", MEMORY_PATH, e)


def sample_recent_examples(n=15):
    out = []
    for line in read_tail_lines(SFT_PATH, n):
        try:
            rec = json.loads(line)
            if isinstance(rec, dict):
                out.append(rec)
        except Exception:
            continue
    return out


def read_dataset_stats():
    return {
        "sft_examples": dataset_count(SFT_PATH),
        "preference_examples": dataset_count(PREF_PATH),
    }


def save_stats():
    payload = {"updated": now_iso(), **read_dataset_stats(), "lifetime": LIFETIME,
               "pipeline_version": PIPELINE_VERSION}
    atomic_write_text(STATS_PATH, json.dumps(payload, ensure_ascii=False, indent=2))


def bump_stat(outcome, reasons=()):
    LIFETIME["cycles"] = LIFETIME.get("cycles", 0) + 1
    if outcome in LIFETIME:
        LIFETIME[outcome] = LIFETIME.get(outcome, 0) + 1
    for r in reasons:
        key = str(r).split(":")[0][:60]
        LIFETIME["reasons"][key] = LIFETIME["reasons"].get(key, 0) + 1
    RUN_COUNTS[outcome] += 1


# ============================================================
# MODEL CLIENT (llama.cpp server, OpenAI-compatible)
# ============================================================

_MODEL_STATE = {"n_ctx": None, "json_mode": True}


def _load_extra_params():
    raw = os.environ.get("AUTOLAB_EXTRA_PARAMS", "").strip()
    if not raw:
        return {}
    try:
        v = json.loads(raw)
        return v if isinstance(v, dict) else {}
    except Exception:
        return {}


# e.g. AUTOLAB_EXTRA_PARAMS='{"repeat_penalty":1.05,"chat_template_kwargs":{"enable_thinking":false}}'
EXTRA_PARAMS = _load_extra_params()

MODEL_STOP_SEQUENCES = ["<end_of_turn>", "<start_of_turn>", "<|end_of_turn|>", "</s>", "<|im_end|>"]
# NOTE: "\n\n\n" was removed from the stop list. It silently truncated any long answer
# (or JSON with a blank line inside a string) at the first triple newline.


def estimate_tokens(text):
    return int(len(text) / CHARS_PER_TOKEN) + 1


def effective_ctx():
    n = _MODEL_STATE["n_ctx"]
    ctx = min(n, CONTEXT_SIZE) if n else CONTEXT_SIZE
    return max(4096, int(ctx))


def prompt_budget_chars(max_tokens, fixed_chars=0):
    """Characters available for variable prompt content given the completion budget."""
    tokens = effective_ctx() - int(max_tokens) - PROMPT_SAFETY_TOKENS
    return max(2000, int(tokens * CHARS_PER_TOKEN) - int(fixed_chars))


def build_system_prompt(system_prompt, role):
    sp = (system_prompt or "").strip()
    if AUTOLAB_MEMORY and role in MEMORY_ROLES:
        return (
            "MEMORY (excerpt):\n\n"
            + AUTOLAB_MEMORY[:MEMORY_MAX_CHARS]
            + "\n\nEND MEMORY.\n\n"
            + sp
        )
    return sp


def shrink_middle(text, max_chars, head_frac=0.4):
    """Keep the head (question/rules) and tail (answer/output format); drop the middle."""
    if len(text) <= max_chars:
        return text
    marker = "\n\n[... content trimmed to fit the context window ...]\n\n"
    keep = max(200, max_chars - len(marker))
    head = int(keep * head_frac)
    return text[:head] + marker + text[len(text) - (keep - head):]


def _auth_headers():
    h = {"Content-Type": "application/json"}
    if LLAMA_API_KEY:
        h["Authorization"] = "Bearer " + LLAMA_API_KEY
    return h


_CTX_ERR_RE = re.compile(
    r"exceed|context (?:size|length|window)|too (?:long|large)|n_ctx|maximum context|n_prompt_tokens", re.I
)


def _learn_ctx_from_error(body):
    n_ctx = None
    try:
        data = json.loads(body)
        err = data.get("error", data) if isinstance(data, dict) else {}
        if isinstance(err, dict) and isinstance(err.get("n_ctx"), (int, float)):
            n_ctx = int(err["n_ctx"])
    except Exception:
        pass
    if n_ctx is None:
        m = re.search(r"n_ctx\D{0,5}(\d{3,7})", body) or re.search(r"maximum context length is (\d{3,7})", body)
        if m:
            n_ctx = int(m.group(1))
    if n_ctx and n_ctx >= 512 and _MODEL_STATE["n_ctx"] != n_ctx:
        log.warning("Learned real context window from server error: n_ctx=%d", n_ctx)
        _MODEL_STATE["n_ctx"] = n_ctx


def _sleep_backoff(attempt, base=2.0, cap=30.0):
    sleep_interruptible(min(cap, base ** attempt + random.uniform(0, 1)))


def _fit_user_prompt(system, user, max_tokens):
    allowed = prompt_budget_chars(max_tokens, fixed_chars=len(system))
    if len(user) <= allowed:
        return user
    log.warning(
        "Prompt too large for context (%d chars, budget %d, ctx %d). Trimming the middle.",
        len(user), allowed, effective_ctx(),
    )
    return shrink_middle(user, allowed)


def call_model_ex(system_prompt, user_prompt, temperature=0.2, max_tokens=2000,
                  timeout=MODEL_TIMEOUT, retries=MODEL_RETRIES, force_json=False, role="generic"):
    """Call the model. Returns {"content", "finish_reason", "prompt_tokens", "completion_tokens", "elapsed"}.

    Recovery built in:
      * prompt is fitted to the real context window before sending
      * HTTP 400 "exceeds context" -> learn n_ctx, shrink the prompt, retry (no sleep)
      * response_format unsupported -> JSON mode disabled for the rest of the run
      * empty output (grammar dead-end / thinking ate the budget) -> JSON mode off,
        more tokens, slightly higher temperature
      * 429 / 5xx / timeouts / connection errors -> exponential backoff
      * a final empty reply is returned as "" (callers have their own fallbacks);
        transport failures raise ModelError.
    """
    system = build_system_prompt(system_prompt, role)
    user = clean_text(user_prompt)
    max_tokens = max(64, int(max_tokens))
    max_tokens = min(max_tokens, max(256, effective_ctx() // 2))
    user = _fit_user_prompt(system, user, max_tokens)
    temperature = float(temperature)
    json_mode = bool(force_json) and _MODEL_STATE["json_mode"]
    attempts = max(1, int(retries))
    last_error, last_was_empty = None, False
    # Deterministic recoveries (shrink the prompt, drop response_format) fix the
    # request itself, so they get their own small budget instead of eating the
    # retry budget reserved for transient transport failures.
    recoveries = 3

    attempt = 0
    while attempt < attempts:
        attempt += 1
        check_stop()
        last_was_empty = False
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": max(0.0, temperature),
            "top_p": 0.9,
            "max_tokens": max_tokens,
            "stop": MODEL_STOP_SEQUENCES,
        }
        payload.update(EXTRA_PARAMS)
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        started = time.time()
        try:
            resp = session.post(
                LLAMA_API, json=payload, headers=_auth_headers(),
                timeout=(MODEL_CONNECT_TIMEOUT, timeout),
            )
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            last_error = e
            log.warning("Model call failed (%d/%d): %s", attempt, attempts, str(e)[:300])
            if attempt < attempts:
                _sleep_backoff(attempt)
            continue
        except requests.exceptions.RequestException as e:
            last_error = e
            log.warning("Model call failed (%d/%d): %s", attempt, attempts, str(e)[:300])
            if attempt < attempts:
                _sleep_backoff(attempt)
            continue

        status = resp.status_code
        if status >= 400:
            body = (resp.text or "")[:1200]
            last_error = ModelError("HTTP %d: %s" % (status, body[:300]))
            log.warning("Model HTTP %d (%d/%d): %s", status, attempt, attempts, body[:200].replace("\n", " "))
            if status == 400 and _CTX_ERR_RE.search(body):
                _learn_ctx_from_error(body)
                target = min(int(len(user) * 0.65), prompt_budget_chars(max_tokens, len(system)))
                user = shrink_middle(user, max(2000, target))
                if recoveries > 0:
                    recoveries -= 1
                    attempt -= 1
                continue
            if status == 400 and json_mode and re.search(r"response_format|json_object|grammar|schema", body, re.I):
                json_mode = False
                _MODEL_STATE["json_mode"] = False
                log.warning("Server rejected response_format; JSON mode disabled for this run.")
                if recoveries > 0:
                    recoveries -= 1
                    attempt -= 1
                continue
            if status in (408, 409, 425, 429) or status >= 500:
                if attempt < attempts:
                    sleep_interruptible(10 * attempt if status == 503 else min(30, 2 ** attempt))
                continue
            raise last_error  # other 4xx: retrying the same request cannot help

        try:
            data = resp.json()
            choice = data["choices"][0]
            message = choice.get("message") or {}
            raw_content = message.get("content")
            if isinstance(raw_content, list):
                raw_content = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in raw_content)
            content = clean_model_text(raw_content or "")
            finish = choice.get("finish_reason")
            usage = data.get("usage") or {}
        except Exception as e:
            last_error = ModelError("malformed response: %s" % str(e)[:200])
            log.warning("Malformed model response (%d/%d): %s", attempt, attempts, str(e)[:200])
            if attempt < attempts:
                _sleep_backoff(attempt)
            continue

        elapsed = time.time() - started
        log.info(
            "MODEL CALL OK: %.1fs, %d chars, finish=%s, prompt_tokens=%s",
            elapsed, len(content), finish, usage.get("prompt_tokens", "?"),
        )
        if content.strip():
            return {
                "content": content,
                "finish_reason": finish,
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "elapsed": elapsed,
            }

        # ---- empty reply ----
        last_was_empty = True
        last_error = ModelError("empty response")
        log.warning(
            "EMPTY RESPONSE (%d/%d): finish=%s prompt_size=%d chars elapsed=%.1fs reasoning=%s",
            attempt, attempts, finish, len(system) + len(user), elapsed,
            "yes" if message.get("reasoning_content") else "no",
        )
        json_mode = False  # grammar-constrained decoding can dead-end; retry unconstrained
        if finish == "length":
            room = effective_ctx() - estimate_tokens(system + user) - PROMPT_SAFETY_TOKENS
            max_tokens = max(64, min(int(max_tokens * 1.5), room))
        elif estimate_tokens(system + user) > effective_ctx() * 0.6:
            user = shrink_middle(user, int(len(user) * 0.75))
        temperature = min(1.0, temperature + 0.1)

    if last_was_empty:
        return {"content": "", "finish_reason": "empty", "prompt_tokens": None,
                "completion_tokens": None, "elapsed": 0.0}
    raise ModelError("Model API failed after %d attempts: %s" % (attempts, last_error))


def call_model(system_prompt, user_prompt, temperature=0.2, max_tokens=2000,
               timeout=MODEL_TIMEOUT, retries=MODEL_RETRIES, force_json=False, role="generic"):
    """String-returning wrapper around call_model_ex (original signature + `role`)."""
    return call_model_ex(
        system_prompt, user_prompt, temperature=temperature, max_tokens=max_tokens,
        timeout=timeout, retries=retries, force_json=force_json, role=role,
    )["content"]


# ---------------- server helpers ----------------

def server_is_up():
    try:
        r = session.get(LLAMA_MODELS_URL, headers=_auth_headers(), timeout=10)
        return r.status_code == 200
    except requests.exceptions.RequestException:
        return False


def wait_for_server(max_wait=SERVER_WAIT_SECONDS):
    """Block until llama-server answers (it returns 503 while a model is still loading)."""
    deadline = time.time() + max_wait
    delay, announced = 2.0, False
    while True:
        if server_is_up():
            return True
        if STOP_EVENT.is_set() or time.time() >= deadline:
            return False
        if not announced:
            log.warning("llama-server not ready at %s; waiting up to %ds ...", LLAMA_BASE_URL, max_wait)
            announced = True
        sleep_interruptible(delay)
        delay = min(15.0, delay * 1.5)


def detect_server_context():
    """Read the per-slot context size llama-server was launched with (GET /props)."""
    try:
        r = session.get(LLAMA_PROPS_URL, headers=_auth_headers(), timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        dgs = data.get("default_generation_settings") or {}
        n = dgs.get("n_ctx") or (dgs.get("params") or {}).get("n_ctx") or data.get("n_ctx")
        n = int(n)
        return n if n >= 512 else None
    except Exception:
        return None


def init_context_window():
    n = detect_server_context()
    if n:
        _MODEL_STATE["n_ctx"] = n
        if n < CONTEXT_SIZE:
            log.warning(
                "Server context is %d tokens, below the declared target %d. Budgeting for %d.",
                n, CONTEXT_SIZE, n,
            )
        else:
            log.info("Server context window: %d tokens.", n)
    else:
        log.warning(
            "Could not read the server context via /props; assuming %d tokens "
            "(set AUTOLAB_CTX to the -c value you launched llama-server with).",
            CONTEXT_SIZE,
        )


# ============================================================
# URL HELPERS
# ============================================================

_TRACKING_PREFIXES = ("utm_", "mtm_", "pk_")
_TRACKING_NAMES = {
    "fbclid", "gclid", "dclid", "msclkid", "mc_cid", "mc_eid", "igshid", "yclid",
    "_hsenc", "_hsmi", "ref_src", "ref_url", "spm", "cmpid", "ocid", "ncid", "s_cid",
    "sr_share", "wt.mc_id",
}


def _is_tracking_param(name):
    n = (name or "").lower()
    return n in _TRACKING_NAMES or n.startswith(_TRACKING_PREFIXES)


def normalize_url(url):
    """Canonical form for dedupe: lowercase scheme/host, default ports and credentials
    dropped, fragment and tracking params removed, trailing slash trimmed."""
    raw = (url or "").strip()
    try:
        p = urlparse(raw)
        scheme = p.scheme.lower()
        host = (p.hostname or "").lower().rstrip(".")
        if not scheme or not host:
            return raw
        try:
            port = p.port
        except ValueError:
            port = None
        netloc = "[%s]" % host if ":" in host else host
        if port and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
            netloc += ":%d" % port
        path = p.path or "/"
        if path != "/":
            path = path.rstrip("/") or "/"
        query = "&".join(
            seg for seg in p.query.split("&")
            if seg and not _is_tracking_param(seg.split("=", 1)[0])
        )
        return urlunparse((scheme, netloc, path, "", query, ""))
    except Exception:
        return raw


_BLOCKED_HOST_SUFFIXES = (
    ".local", ".localhost", ".internal", ".lan", ".home", ".corp", ".intranet", ".localdomain",
)


def _ip_is_public(ip):
    mapped = getattr(ip, "ipv4_mapped", None)
    if mapped is not None:
        ip = mapped
    return bool(ip.is_global) and not ip.is_multicast


def is_public_url(url):
    """Cheap syntactic SSRF guard (scheme, literal IPs, intranet-style names)."""
    try:
        p = urlparse(url)
        if p.scheme not in {"http", "https"} or not p.hostname:
            return False
        host = p.hostname.lower().rstrip(".")
        if host in {"localhost", "localhost.localdomain", "ip6-localhost"}:
            return False
        if host.endswith(_BLOCKED_HOST_SUFFIXES):
            return False
        try:
            ip = ipaddress.ip_address(host)
        except ValueError:
            return "." in host  # single-label names are intranet hosts
        return _ip_is_public(ip)
    except Exception:
        return False


_DNS_CACHE = {}
_DNS_LOCK = threading.Lock()


def host_resolves_public(host):
    """True when every address the host resolves to is public (blocks DNS-based SSRF)."""
    if SKIP_DNS_CHECK:
        return True
    host = (host or "").lower().rstrip(".")
    now = time.time()
    with _DNS_LOCK:
        hit = _DNS_CACHE.get(host)
    if hit and now - hit[1] < 600:
        return hit[0]
    ok = False
    try:
        infos = socket.getaddrinfo(host, None)
        addrs = {i[4][0].split("%")[0] for i in infos}
        ok = bool(addrs) and all(_ip_is_public(ipaddress.ip_address(a)) for a in addrs)
    except (socket.gaierror, ValueError, OSError):
        ok = False
    with _DNS_LOCK:
        if len(_DNS_CACHE) > 2000:
            _DNS_CACHE.clear()
        _DNS_CACHE[host] = (ok, now)
    return ok


_MULTI_SUFFIXES = {
    "co.uk", "org.uk", "ac.uk", "gov.uk", "me.uk", "ltd.uk", "plc.uk", "nhs.uk",
    "com.au", "net.au", "org.au", "edu.au", "gov.au", "co.nz", "org.nz", "ac.nz", "govt.nz",
    "co.jp", "ac.jp", "go.jp", "or.jp", "ne.jp", "com.br", "gov.br", "edu.br",
    "com.cn", "edu.cn", "gov.cn", "org.cn", "co.in", "ac.in", "gov.in", "nic.in",
    "co.za", "ac.za", "gov.za", "com.mx", "com.ar", "com.tr", "edu.tr",
    "com.sg", "edu.sg", "gov.sg", "co.kr", "ac.kr", "go.kr", "co.il", "ac.il", "gov.il",
}


def base_domain(host):
    """registrable domain: www.bbc.co.uk -> bbc.co.uk, news.mit.edu -> mit.edu"""
    host = (host or "").lower().strip(".")
    if host.startswith("["):
        return host
    host = host.split(":")[0]
    labels = [x for x in host.split(".") if x]
    if len(labels) <= 2:
        return ".".join(labels)
    if ".".join(labels[-2:]) in _MULTI_SUFFIXES:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:])


def url_base_domain(url):
    try:
        return base_domain(urlparse(url).hostname or "")
    except Exception:
        return ""


# ============================================================
# SEARCH
# ============================================================

_HARD_BLOCK_RE = re.compile(
    r"(?<!\d)(?:429|403)(?!\d)|captcha|too many requests|rate.?limit|ratelimit|forbidden", re.I
)
_BAD_BACKEND_RE = re.compile(
    r"unknown backend|invalid backend|unsupported backend|no such backend|not a valid backend|"
    r"unexpected keyword|backend.{0,30}(?:not|invalid|unknown)", re.I
)


def _is_hard_block(msg: str) -> bool:
    return bool(_HARD_BLOCK_RE.search(msg or ""))


class _BackendHealth:
    """Circuit breaker with cooldown: a blocked backend is retried later instead of
    being disabled for the whole run."""

    def __init__(self):
        self._lock = threading.Lock()
        self.blocked_until = {}
        self.block_count = Counter()
        self.successes = Counter()

    def available(self, backend):
        with self._lock:
            return time.time() >= self.blocked_until.get(backend, 0.0)

    def block(self, backend, seconds=None):
        with self._lock:
            self.block_count[backend] += 1
            secs = seconds if seconds is not None else min(
                3600, BACKEND_COOLDOWN_SECONDS * 2 ** (self.block_count[backend] - 1)
            )
            self.blocked_until[backend] = time.time() + secs
        return secs

    def success(self, backend):
        with self._lock:
            self.successes[backend] += 1
            self.block_count[backend] = 0

    def live(self, backends):
        return [b for b in backends if self.available(b)]


_BACKENDS = _BackendHealth()


def clean_query(query):
    """Strip operators that some providers reject and normalise whitespace."""
    q = clean_text(query)
    q = re.sub(r"\bsite:\s*(\S+)", r"\1", q, flags=re.I)  # "site:arxiv.org foo" -> "arxiv.org foo"
    if q.count('"') % 2:
        q = q.replace('"', " ")
    q = re.sub(r"\s+", " ", q).strip()
    return q[:250]


def resolved_search_provider():
    if SEARCH_PROVIDER in ("serper", "auto"):
        return "serper" if SERPER_API_KEY else "ddgs"
    return "ddgs"


def search_with_backend(query, max_results, backend):
    if DDGS is None or not _BACKENDS.available(backend):
        return []
    results = []
    try:
        with DDGS() as ddgs:
            try:
                iterator = ddgs.text(query, max_results=max_results, backend=backend)
            except TypeError:
                return []  # older package without a backend kwarg
            for result in iterator or []:
                if not isinstance(result, dict):
                    continue
                title = clean_text(result.get("title", "")).strip()
                url = str(result.get("href") or result.get("url") or "").strip()
                body = clean_text(result.get("body") or result.get("snippet") or "").strip()
                if title and url and is_public_url(url):
                    results.append({
                        "title": title,
                        "url": normalize_url(url),
                        "snippet": body,
                        "search_backend": backend,
                    })
    except Exception as e:
        msg = "%s: %s" % (type(e).__name__, str(e))
        if _BAD_BACKEND_RE.search(msg):
            secs = _BACKENDS.block(backend, seconds=24 * 3600)
            log.warning("Search backend [%s] looks unsupported by this ddgs version; skipped for %ds: %s",
                        backend, secs, msg[:160])
        elif _is_hard_block(msg) or "ratelimit" in type(e).__name__.lower():
            secs = _BACKENDS.block(backend)
            log.warning("Backend [%s] HARD-BLOCKED (429/403/captcha); cooling down %ds: %s",
                        backend, secs, msg[:160])
        else:
            log.warning("Search backend [%s] soft error on this query: %s", backend, msg[:200])
    if results:
        _BACKENDS.success(backend)
    return results


def search_with_serper(query, max_results=RESULTS_PER_QUERY):
    """Google results via Serper.dev. Requires SERPER_API_KEY."""
    if not SERPER_API_KEY:
        log.error("SERPER_API_KEY not set. Cannot use Serper provider.")
        return []
    if not _BACKENDS.available("serper"):
        return []
    query = clean_query(query)
    for attempt in range(1, 4):
        try:
            response = session.post(
                SERPER_ENDPOINT,
                headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
                json={"q": query, "num": int(max_results), "hl": "en"},
                timeout=WEB_FETCH_TIMEOUT,
            )
        except requests.exceptions.RequestException as e:
            log.warning("Serper request failed (%d/3): %s", attempt, str(e)[:200])
            sleep_interruptible(min(8, 2 ** attempt))
            continue
        code = response.status_code
        if code == 429 or code >= 500:
            log.warning("Serper HTTP %d (%d/3).", code, attempt)
            sleep_interruptible(min(10, 2 ** attempt))
            continue
        if code in (401, 402, 403):
            secs = _BACKENDS.block("serper", seconds=3600)
            log.error("Serper auth/quota failed (%d). Check SERPER_API_KEY / credits. Pausing Serper %ds.",
                      code, secs)
            return []
        if code >= 400:
            log.warning("Serper HTTP %d: %s", code, response.text[:200])
            return []
        try:
            data = response.json()
        except ValueError:
            log.warning("Serper returned non-JSON.")
            return []
        results = []
        for item in (data.get("organic") or [])[:max_results]:
            if not isinstance(item, dict):
                continue
            url = str(item.get("link", "")).strip()
            title = clean_text(item.get("title", "")).strip()
            if not url or not title or not is_public_url(url):
                continue
            results.append({
                "title": title,
                "url": normalize_url(url),
                "snippet": clean_text(item.get("snippet", "")).strip(),
                "search_backend": "serper",
            })
        log.info("Serper returned %d results for: %s", len(results), query)
        return results
    return []


def search_via_ddgs(query, max_results):
    results, seen = [], set()
    for backend in WEB_SEARCH_BACKENDS:
        check_stop()
        if not _BACKENDS.available(backend):
            continue
        for result in search_with_backend(query, max_results, backend):
            url = normalize_url(result["url"])
            if url and url not in seen:
                seen.add(url)
                result["url"] = url
                results.append(result)
                if len(results) >= max_results:
                    break
        if len(results) >= max_results:
            break
        sleep_interruptible(random.uniform(WEB_SEARCH_DELAY_MIN, WEB_SEARCH_DELAY_MAX))
    return results


_QUERY_CACHE = {}
_QUERY_CACHE_LOCK = threading.Lock()
_DDGS_WARNED = False


def _warn_no_ddgs():
    global _DDGS_WARNED
    if not _DDGS_WARNED:
        _DDGS_WARNED = True
        log.error(
            "No search backend available: neither 'ddgs' nor 'duckduckgo_search' is "
            "installed and SERPER_API_KEY is unset. Run: pip install ddgs"
        )


def web_search(query, max_results=RESULTS_PER_QUERY):
    query = clean_query(query)
    if not query:
        return []
    key = query.lower()
    with _QUERY_CACHE_LOCK:
        cached = _QUERY_CACHE.get(key)
    if cached is not None:
        return [dict(r) for r in cached][:max_results]

    provider = resolved_search_provider()
    if provider == "ddgs" and DDGS is None:
        _warn_no_ddgs()
        return []
    log.info("SEARCH [%s]: %s", provider, query)
    if provider == "serper":
        results = search_with_serper(query, max_results)
        if not results and SEARCH_FALLBACK_TO_DDGS and DDGS is not None:
            log.info("Serper returned nothing; falling back to ddgs for this query.")
            results = search_via_ddgs(query, max_results)
    else:
        results = search_via_ddgs(query, max_results)

    log.info(
        "SEARCH RESULTS: %d unique for: %s  (live backends: %s)",
        len(results), query, _BACKENDS.live(WEB_SEARCH_BACKENDS) if provider == "ddgs" else ["serper"],
    )
    results = results[:max_results]
    if results:
        with _QUERY_CACHE_LOCK:
            if len(_QUERY_CACHE) > 300:
                _QUERY_CACHE.clear()
            _QUERY_CACHE[key] = [dict(r) for r in results]
    return results


# ============================================================
# PAGE FETCHING
# ============================================================

class FetchRetry(Exception):
    def __init__(self, msg, wait=None):
        super().__init__(msg)
        self.wait = wait


class FetchFail(Exception):
    pass


_RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}
_TLS = threading.local()


def _fetch_session():
    s = getattr(_TLS, "session", None)
    if s is None:
        s = requests.Session()
        s.headers.update({
            "User-Agent": FETCH_USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/pdf;q=0.8,*/*;q=0.7",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate",
            "Cache-Control": "no-cache",
        })
        adapter = HTTPAdapter(pool_connections=8, pool_maxsize=8, max_retries=0)
        s.mount("http://", adapter)
        s.mount("https://", adapter)
        _TLS.session = s
    return s


def _url_candidates(url):
    """Try an easier-to-scrape mirror first where one is known (old.reddit.com serves plain HTML)."""
    try:
        p = urlparse(url)
        host = (p.hostname or "").lower()
        if host in ("www.reddit.com", "reddit.com", "np.reddit.com"):
            alt = urlunparse((p.scheme, "old.reddit.com", p.path, "", p.query, ""))
            return [alt, url]
    except Exception:
        pass
    return [url]


def _get_following_redirects(sess, url, deadline):
    """GET with manual redirects so every hop is re-validated (scheme, IP, DNS)."""
    current = url
    for _hop in range(MAX_REDIRECTS + 1):
        if not is_public_url(current):
            raise FetchFail("non-public url")
        host = urlparse(current).hostname or ""
        if not host_resolves_public(host):
            raise FetchFail("host unresolvable or non-public")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise FetchFail("deadline")
        try:
            resp = sess.get(
                current,
                timeout=(WEB_FETCH_CONNECT_TIMEOUT, min(WEB_FETCH_TIMEOUT, max(2.0, remaining))),
                allow_redirects=False,
                stream=True,
            )
        except requests.exceptions.SSLError as e:
            raise FetchFail("ssl: %s" % str(e)[:80])
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            raise FetchRetry("%s" % type(e).__name__)
        except requests.exceptions.RequestException as e:
            raise FetchFail("%s: %s" % (type(e).__name__, str(e)[:80]))
        if resp.status_code in (301, 302, 303, 307, 308):
            loc = resp.headers.get("Location")
            resp.close()
            if not loc:
                raise FetchFail("redirect without Location")
            current = normalize_url(urljoin(current, loc))
            continue
        return resp, current
    raise FetchFail("too many redirects")


def _retry_after_seconds(resp):
    try:
        v = resp.headers.get("Retry-After")
        if v and v.strip().isdigit():
            return min(10.0, float(v))
    except Exception:
        pass
    return None


def _read_body(resp, limit, deadline, must_be_complete):
    chunks, total, truncated = [], 0, False
    try:
        for chunk in resp.iter_content(chunk_size=65536):
            if not chunk:
                continue
            chunks.append(chunk)
            total += len(chunk)
            if total >= limit:
                truncated = True
                break
            if time.monotonic() > deadline:
                raise FetchFail("read deadline")
    except (requests.exceptions.ChunkedEncodingError, requests.exceptions.ContentDecodingError,
            requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
        raise FetchRetry("body read: %s" % type(e).__name__)
    if truncated and must_be_complete:
        raise FetchFail("file larger than %d MB" % (limit // (1024 * 1024)))
    return b"".join(chunks), truncated


def _decode_bytes(data, content_type=""):
    m = re.search(r"charset=([\w\-]+)", content_type or "", re.I)
    if not m:
        m = re.search(rb"<meta[^>]+charset=[\"']?([\w\-]+)", data[:4096], re.I)
        enc = m.group(1).decode("ascii", "ignore") if m else "utf-8"
    else:
        enc = m.group(1)
    try:
        return data.decode(enc, errors="replace")
    except (LookupError, UnicodeDecodeError):
        return data.decode("utf-8", errors="replace")


_PDF_WARNED = False


def extract_pdf_text(data):
    global _PDF_WARNED
    if PdfReader is None:
        if not _PDF_WARNED:
            log.warning("pypdf is not installed - PDF sources are skipped (pip install pypdf).")
            _PDF_WARNED = True
        return ""
    try:
        reader = PdfReader(io.BytesIO(data), strict=False)
        if getattr(reader, "is_encrypted", False):
            try:
                if not reader.decrypt(""):
                    return ""
            except Exception:
                return ""
        parts, total = [], 0
        for i, page in enumerate(reader.pages):
            if i >= PDF_MAX_PAGES:
                break
            try:
                t = page.extract_text() or ""
            except Exception:
                continue
            parts.append(t)
            total += len(t)
            if total >= FETCH_KEEP_CHARS * 1.2:
                break
        return "\n".join(parts)
    except Exception as e:
        log.debug("PDF extract failed: %s", e)
        return ""


_NOISE_TAGS = ["script", "style", "noscript", "svg", "canvas", "iframe", "nav", "footer", "aside",
               "template", "button", "select", "option", "input", "textarea"]


def extract_html_text(data, content_type=""):
    if trafilatura is not None:
        try:
            html = _decode_bytes(data, content_type)
            t = trafilatura.extract(html, include_comments=True, include_tables=True, favor_recall=True)
            if t and len(t) >= MIN_SOURCE_CHARS:
                return t
        except Exception as e:
            log.debug("trafilatura failed: %s", e)
    try:
        try:
            soup = BeautifulSoup(data, "lxml")
        except Exception:
            soup = BeautifulSoup(data, "html.parser")
        for tag in soup(_NOISE_TAGS):
            tag.decompose()
        for tag in soup.find_all(attrs={"role": ["dialog", "alertdialog"]}):
            tag.decompose()
        # ASP.NET/legacy sites wrap the whole page in <form>; dropping it emptied the page.
        for form in soup.find_all("form"):
            form.unwrap()
        body = soup.body or soup
        total_len = len(body.get_text(" ", strip=True))
        node = body
        for sel in ("article", "main", "[role=main]"):
            el = soup.select_one(sel)
            if el is not None:
                n = len(el.get_text(" ", strip=True))
                if n >= 1500 and n >= 0.3 * total_len:
                    node = el
                    break
        return node.get_text("\n", strip=True)
    except Exception as e:
        log.debug("HTML parse failed: %s", e)
        return _decode_bytes(data, content_type)


def finalize_text(text):
    """Whitespace cleanup + removal of repeated menu/button lines."""
    text = clean_text(text)
    text = re.sub(r"[ \t\u00a0]+", " ", text)
    out, seen_short, prev = [], set(), None
    for ln in (x.strip() for x in text.split("\n")):
        if not ln:
            if out and out[-1] != "":
                out.append("")
            continue
        if ln == prev:
            continue
        if len(ln) < 60:
            if ln in seen_short:
                continue
            seen_short.add(ln)
        out.append(ln)
        prev = ln
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


_BLOCK_PATTERNS = re.compile(
    r"enable javascript|javascript is (?:required|disabled)|verify you are (?:a )?human|are you a robot|"
    r"captcha|access denied|request blocked|attention required|checking your browser|just a moment|"
    r"unusual traffic|403 forbidden|429 too many|too many requests|please turn javascript on|"
    r"pardon our interruption|cloudflare ray id|bot detection",
    re.I,
)


def looks_like_block_page(text):
    hits = len(_BLOCK_PATTERNS.findall(text[:1500]))
    return (len(text) < 2500 and hits >= 1) or hits >= 3


def text_problem(text, is_pdf=False):
    """Return a reason string if the extracted text is not usable evidence, else ''.

    PDF extraction legitimately produces more punctuation/ligature noise than HTML
    (math, hyphenation, column bleed), so its thresholds are a little looser.
    """
    if len(text) < MIN_SOURCE_CHARS:
        return "too short"
    if looks_like_block_page(text):
        return "block/captcha page"
    if alnum_ratio(text) < (0.42 if is_pdf else 0.5):
        return "garbled text"
    if REQUIRE_LATIN_TEXT and latin_ratio(text) < (0.6 if is_pdf else 0.7):
        return "non-latin text"
    if len(set(text[:4000])) < 12:
        return "degenerate text"
    return ""


def _fetch_once(sess, url, deadline):
    resp, final_url = _get_following_redirects(sess, url, deadline)
    try:
        status = resp.status_code
        if status in _RETRYABLE_STATUS:
            raise FetchRetry("HTTP %d" % status, _retry_after_seconds(resp))
        if status >= 400:
            raise FetchFail("HTTP %d" % status)
        ctype = (resp.headers.get("Content-Type") or "").lower()
        path = (urlparse(final_url).path or "").lower()
        pdf_hint = "pdf" in ctype or path.endswith(".pdf")
        if ctype.startswith(("image/", "video/", "audio/", "font/")) or any(
            z in ctype for z in ("zip", "x-tar", "gzip", "msword", "officedocument", "spreadsheet", "presentation")
        ):
            raise FetchFail("unsupported content-type %s" % ctype[:40])
        limit = MAX_DOWNLOAD_BYTES_PDF if pdf_hint else MAX_DOWNLOAD_BYTES_HTML
        clen = resp.headers.get("Content-Length", "")
        if pdf_hint and clen.isdigit() and int(clen) > limit:
            raise FetchFail("pdf larger than %d MB" % (limit // (1024 * 1024)))
        data, truncated = _read_body(resp, limit, deadline, must_be_complete=pdf_hint)
    finally:
        resp.close()

    is_pdf = data[:5] == b"%PDF-" or "application/pdf" in ctype
    if is_pdf and truncated:
        raise FetchFail("pdf truncated")
    if "octet-stream" in ctype and not is_pdf:
        raise FetchFail("binary content")
    if is_pdf:
        raw_text = extract_pdf_text(data)
    elif "text/plain" in ctype:
        raw_text = _decode_bytes(data, ctype)
    else:
        raw_text = extract_html_text(data, ctype)

    text = finalize_text(raw_text)
    problem = text_problem(text, is_pdf)
    if problem:
        raise FetchFail(problem)
    return {
        "ok": True,
        "text": text[:FETCH_KEEP_CHARS],
        "final_url": final_url,
        "content_type": "pdf" if is_pdf else (ctype.split(";")[0] or "html"),
        "bytes": len(data),
        "error": "",
    }


def _fetch_failure(url, error):
    return {"ok": False, "text": "", "final_url": url, "content_type": "",
            "bytes": 0, "error": error}


def fetch_page_ex(url):
    """Fetch one URL. Always returns a dict: {"ok": bool, "text", "final_url", "error", ...}."""
    if not is_public_url(url):
        return _fetch_failure(url, "non-public url")
    url = normalize_url(url)
    deadline = time.monotonic() + FETCH_TOTAL_TIMEOUT
    sess = _fetch_session()
    last_err = "unknown"
    for cand in _url_candidates(url):
        for attempt in range(1, WEB_FETCH_RETRIES + 1):
            if STOP_EVENT.is_set() or time.monotonic() >= deadline:
                return _fetch_failure(url, "deadline/stop")
            try:
                return _fetch_once(sess, cand, deadline)
            except FetchRetry as e:
                last_err = str(e)
                if attempt < WEB_FETCH_RETRIES:
                    wait = e.wait or min(6, 2 ** (attempt - 1) + random.uniform(0.2, 0.8))
                    if wait >= deadline - time.monotonic():
                        break
                    sleep_interruptible(wait)
            except FetchFail as e:
                last_err = str(e)
                break
            except Exception as e:  # never let one bad page kill a cycle
                last_err = "%s: %s" % (type(e).__name__, str(e)[:100])
                break
    return _fetch_failure(url, last_err)


def fetch_page(url):
    """Compatibility wrapper: cleaned text or None."""
    res = fetch_page_ex(url)
    return res["text"] if res.get("ok") else None

# ============================================================
# SOURCE CLASSIFICATION
# ============================================================
# Tier meaning (also used by MIN_CITED_SOURCE_QUALITY / MIN_FETCH_TIER):
#   10 primary government / space agency / military / archival record
#    9 peer-reviewed literature, DOI, preprint servers, journal publishers
#    8 universities, national labs, statistics offices, courts, standards bodies
#    7 official technical documentation, specifications, official datasets
#    6 high-reputation press and reference works (secondary, but edited)
#    5 community testimony: forums, Q&A, code hosts, mailing lists
#    4 generic news / unknown domain (the default)
#    3 self-publishing platforms, opinion, aggregators
#    2 tabloids, content farms, low-trust outlets
#    1 unfetchable or non-evidentiary social platforms (never fetched)

TIER_PRIMARY_GOV = 10
TIER_PEER_REVIEWED = 9
TIER_ACADEMIC = 8
TIER_TECHNICAL = 7
TIER_PRESS = 6
TIER_COMMUNITY = 5
TIER_DEFAULT = 4
TIER_SELF_PUBLISHED = 3
TIER_LOW_TRUST = 2
TIER_NON_EVIDENTIARY = 1

_DOMAIN_TIERS = {
    # --- 10: primary government / agency / archive ---------------------------
    "nasa.gov": 10, "jpl.nasa.gov": 10, "noaa.gov": 10, "nist.gov": 10, "usgs.gov": 10,
    "nih.gov": 10, "cdc.gov": 10, "energy.gov": 10, "osti.gov": 10, "archives.gov": 10,
    "govinfo.gov": 10, "congress.gov": 10, "senate.gov": 10, "house.gov": 10,
    "gao.gov": 10, "crsreports.congress.gov": 10, "federalregister.gov": 10,
    "supremecourt.gov": 10, "uscourts.gov": 10, "courtlistener.com": 10,
    "ntsb.gov": 10, "faa.gov": 10, "nrc.gov": 10, "dni.gov": 10, "cia.gov": 10,
    "nsa.gov": 10, "defense.gov": 10, "dod.gov": 10, "state.gov": 10, "fbi.gov": 10,
    "esa.int": 10, "cern.ch": 10, "who.int": 10, "iaea.org": 10, "europa.eu": 10,
    "jaxa.jp": 10, "ecmwf.int": 10, "bis.org": 10, "imf.org": 10, "worldbank.org": 10,
    "bls.gov": 10, "census.gov": 10, "federalreserve.gov": 10, "sec.gov": 10,
    "ukdefencejournal.org.uk": 4,
    # --- 9: peer-reviewed literature ----------------------------------------
    "doi.org": 9, "arxiv.org": 9, "biorxiv.org": 9, "medrxiv.org": 9, "ssrn.com": 8,
    "nature.com": 9, "science.org": 9, "sciencemag.org": 9, "cell.com": 9, "pnas.org": 9,
    "plos.org": 9, "plosone.org": 9, "elifesciences.org": 9, "springer.com": 9,
    "link.springer.com": 9, "sciencedirect.com": 9, "wiley.com": 9, "onlinelibrary.wiley.com": 9,
    "tandfonline.com": 9, "sagepub.com": 9, "jstor.org": 9, "cambridge.org": 9,
    "academic.oup.com": 9, "oup.com": 9, "ieee.org": 9, "ieeexplore.ieee.org": 9,
    "acm.org": 9, "dl.acm.org": 9, "aps.org": 9, "journals.aps.org": 9, "aip.org": 9,
    "iop.org": 9, "iopscience.iop.org": 9, "rsc.org": 9, "acs.org": 9, "pubs.acs.org": 9,
    "pubmed.ncbi.nlm.nih.gov": 9, "ncbi.nlm.nih.gov": 9, "europepmc.org": 9,
    "ui.adsabs.harvard.edu": 9, "adsabs.harvard.edu": 9, "aanda.org": 9, "iau.org": 9,
    "agu.org": 9, "copernicus.org": 9, "frontiersin.org": 8, "mdpi.com": 7,
    "semanticscholar.org": 8, "openalex.org": 8, "researchgate.net": 6, "zenodo.org": 8,
    "osf.io": 8, "hal.science": 8,
    # --- 8: academia, labs, statistics, courts, standards --------------------
    "nationalacademies.org": 8, "rand.org": 8, "brookings.edu": 8, "aaas.org": 8,
    "royalsociety.org": 8, "mpg.de": 8, "cnrs.fr": 8, "riken.jp": 8, "csiro.au": 8,
    "eso.org": 9, "stsci.edu": 9, "seti.org": 8, "lpi.usra.edu": 8, "usra.edu": 8,
    "aiaa.org": 8, "asme.org": 8, "iso.org": 8, "iec.ch": 8, "itu.int": 8,
    "ourworldindata.org": 7, "statista.com": 5,
    # --- 7: technical documentation / specifications / datasets --------------
    "ietf.org": 7, "rfc-editor.org": 7, "w3.org": 7, "unicode.org": 7, "khronos.org": 7,
    "kernel.org": 7, "python.org": 7, "docs.python.org": 7, "developer.mozilla.org": 7,
    "postgresql.org": 7, "sqlite.org": 7, "openssl.org": 7, "llvm.org": 7, "gnu.org": 7,
    "cve.org": 7, "mitre.org": 7, "nvd.nist.gov": 7, "cisa.gov": 10, "owasp.org": 7,
    "debian.org": 7, "redhat.com": 6, "docs.oracle.com": 7, "microsoft.com": 6,
    "learn.microsoft.com": 7, "developer.apple.com": 7, "android.com": 6,
    "pytorch.org": 7, "tensorflow.org": 7, "numpy.org": 7, "scipy.org": 7,
    "huggingface.co": 6, "openai.com": 6, "anthropic.com": 6, "deepmind.com": 6,
    # --- 6: edited press / reference -----------------------------------------
    "reuters.com": 6, "apnews.com": 6, "bbc.com": 6, "bbc.co.uk": 6, "npr.org": 6,
    "pbs.org": 6, "nytimes.com": 6, "washingtonpost.com": 6, "wsj.com": 6, "ft.com": 6,
    "economist.com": 6, "theguardian.com": 6, "bloomberg.com": 6, "cnbc.com": 5,
    "politico.com": 5, "axios.com": 5, "propublica.org": 6, "theatlantic.com": 5,
    "newyorker.com": 5, "scientificamerican.com": 6, "sciencenews.org": 6,
    "newscientist.com": 6, "quantamagazine.org": 6, "spectrum.ieee.org": 6,
    "arstechnica.com": 6, "theregister.com": 5, "nasaspaceflight.com": 6,
    "spacenews.com": 6, "space.com": 5, "phys.org": 5, "eurekalert.org": 5,
    "wikipedia.org": 5, "britannica.com": 6, "wikisource.org": 6, "archive.org": 6,
    "gutenberg.org": 6, "theblackvault.com": 5, "documentcloud.org": 7,
    "muckrock.com": 7, "governmentattic.org": 7, "nsarchive.gwu.edu": 9,
    # --- 5: community testimony ----------------------------------------------
    "reddit.com": 5, "news.ycombinator.com": 5, "ycombinator.com": 5,
    "stackexchange.com": 5, "stackoverflow.com": 5, "superuser.com": 5,
    "serverfault.com": 5, "askubuntu.com": 5, "mathoverflow.net": 5,
    "github.com": 5, "gitlab.com": 5, "sourceforge.net": 4, "bitbucket.org": 5,
    "groups.google.com": 5, "mail-archive.com": 5, "lore.kernel.org": 6,
    "lwn.net": 6, "metabunk.org": 5, "physicsforums.com": 5, "lesswrong.com": 5,
    "discourse.group": 5, "discuss.python.org": 5, "forum.arduino.cc": 5,
    "cloudynights.com": 5, "reddit.co": 5, "old.reddit.com": 5,
    # --- 3: self-publishing / opinion / aggregation --------------------------
    "medium.com": 3, "substack.com": 3, "linkedin.com": 3, "blogspot.com": 3,
    "wordpress.com": 3, "wix.com": 3, "tumblr.com": 3, "patheos.com": 3,
    "salon.com": 3, "cbn.com": 3, "vice.com": 3, "buzzfeed.com": 3, "msn.com": 3,
    "yahoo.com": 3, "news.google.com": 3, "flipboard.com": 3, "prnewswire.com": 3,
    "businesswire.com": 3, "globenewswire.com": 3,
    # --- 2: tabloid / low trust ----------------------------------------------
    "dailymail.co.uk": 2, "the-sun.com": 2, "mirror.co.uk": 2, "express.co.uk": 2,
    "nypost.com": 3, "rt.com": 2, "sputniknews.com": 2, "infowars.com": 1,
    "beforeitsnews.com": 1, "naturalnews.com": 1,
    # --- 1: non-evidentiary / unfetchable ------------------------------------
    "x.com": 1, "twitter.com": 1, "youtube.com": 1, "youtu.be": 1, "facebook.com": 1,
    "instagram.com": 1, "tiktok.com": 1, "pinterest.com": 1, "quora.com": 1,
    "scribd.com": 1, "coursehero.com": 1, "chegg.com": 1, "slideshare.net": 1,
    "issuu.com": 1, "academia.edu": 3, "tripadvisor.com": 1, "amazon.com": 1,
    "ebay.com": 1, "yelp.com": 1, "glassdoor.com": 1, "indeed.com": 1,
}

# Host substrings that mark community testimony regardless of the registrable domain.
_COMMUNITY_MARKERS = (
    "reddit.", "ycombinator", "stackexchange", "stackoverflow", "superuser",
    "serverfault", "askubuntu", "mathoverflow", "forum", "forums.", "boards.",
    "community.", "discuss", "discourse", "groups.google", "mail-archive",
    "lists.", "listserv", "mailman", "lore.kernel", "github.com", "gitlab.com",
    "metabunk", "physicsforums", "lesswrong", "cloudynights", "bbs.",
)

# Suffix rules applied when the registrable domain is unknown.
_SUFFIX_TIERS = (
    (".mil", 10), (".gov", 9), (".gov.uk", 9), (".gov.au", 9), (".govt.nz", 9),
    (".gc.ca", 9), (".go.jp", 9), (".gov.in", 9), (".gov.br", 9), (".gov.za", 9),
    (".edu", 8), (".ac.uk", 8), (".edu.au", 8), (".ac.jp", 8), (".ac.nz", 8),
    (".edu.sg", 8), (".ac.il", 8), (".ac.in", 8), (".edu.cn", 8), (".int", 8),
)

_INSTITUTIONAL_KINDS = {"primary-gov", "peer-reviewed", "academic", "technical", "press", "reference"}

_SOURCE_PROFILE_CACHE = {}
_SOURCE_PROFILE_LOCK = threading.Lock()


def _kind_for_tier(tier, community):
    if community:
        return "community"
    if tier >= 10:
        return "primary-gov"
    if tier == 9:
        return "peer-reviewed"
    if tier == 8:
        return "academic"
    if tier == 7:
        return "technical"
    if tier == 6:
        return "press"
    if tier == 5:
        return "reference"
    if tier == 4:
        return "unknown"
    if tier == 3:
        return "self-published"
    if tier == 2:
        return "low-trust"
    return "non-evidentiary"


def source_profile(url):
    """{'domain','host','tier','kind','community','institutional'} for a URL. Cached."""
    try:
        host = (urlparse(url).hostname or "").lower().rstrip(".")
    except Exception:
        host = ""
    if not host:
        return {"domain": "", "host": "", "tier": TIER_NON_EVIDENTIARY,
                "kind": "non-evidentiary", "community": False, "institutional": False}
    with _SOURCE_PROFILE_LOCK:
        hit = _SOURCE_PROFILE_CACHE.get(host)
    if hit is not None:
        return hit

    domain = base_domain(host)
    tier = None
    # Most specific match first: full host, then the registrable domain.
    for key in (host, domain):
        if key in _DOMAIN_TIERS:
            tier = _DOMAIN_TIERS[key]
            break
    if tier is None:
        for suffix, value in _SUFFIX_TIERS:
            if host.endswith(suffix):
                tier = value
                break
    if tier is None:
        tier = TIER_DEFAULT

    community = any(marker in host for marker in _COMMUNITY_MARKERS)
    if community and tier == TIER_DEFAULT:
        tier = TIER_COMMUNITY
    # A university-hosted mailing list is still testimony, but a .gov page is not
    # demoted just because its host contains "community.".
    if community and tier >= TIER_PRESS and not host.endswith((".gov", ".mil", ".edu")):
        pass  # keep the higher tier; `community` flag carries the testimony caveat

    kind = _kind_for_tier(tier, community)
    profile = {
        "domain": domain,
        "host": host,
        "tier": int(tier),
        "kind": kind,
        "community": bool(community),
        "institutional": kind in _INSTITUTIONAL_KINDS,
    }
    with _SOURCE_PROFILE_LOCK:
        if len(_SOURCE_PROFILE_CACHE) > 4000:
            _SOURCE_PROFILE_CACHE.clear()
        _SOURCE_PROFILE_CACHE[host] = profile
    return profile


def classify_source(url):
    """Integer quality tier 1..10 for a URL (see the table above)."""
    return source_profile(url)["tier"]


def is_community_source(url):
    return source_profile(url)["community"]


def counts_as_tier5(profile):
    """True when a source satisfies the 'engage with what people report' requirement."""
    tier = profile.get("tier", 0)
    if not (TIER_COMMUNITY <= tier < MIN_CITED_SOURCE_QUALITY):
        return False
    if TIER5_MUST_BE_NON_INSTITUTIONAL:
        return bool(profile.get("community")) or not profile.get("institutional")
    return True


# ============================================================
# TEXT ANALYSIS  (keywords, relevance, excerpts, sentences)
# ============================================================

STOPWORDS = frozenset("""
a about above after again against all am an and any are aren as at be because been
before being below between both but by can cannot could couldn did didn do does
doesn doing don down during each few for from further had hadn has hasn have haven
having he her here hers herself him himself his how i if in into is isn it its
itself just me more most mustn my myself no nor not now of off on once only or
other ought our ours ourselves out over own same shan she should shouldn so some
such than that the their theirs them themselves then there these they this those
through to too under until up very was wasn we were weren what when where which
while who whom why will with won would wouldn you your yours yourself yourselves
also may might must shall since upon whether among within without given via toward
towards across based used using make makes made new use get got go going
""".split())

_WORD_RE = re.compile(r"[a-z0-9][a-z0-9\-']*")
_NUMERIC_RE = re.compile(r"^\d[\d.,:/\-]*$")


def content_tokens(text, min_len=3):
    """Lowercase content words, stopwords and 1-2 char noise removed."""
    out = []
    for w in _WORD_RE.findall((text or "").lower()):
        if w in STOPWORDS:
            continue
        if len(w) < min_len and not w.isdigit():
            continue
        out.append(w)
    return out


def keyword_profile(*texts):
    """Weighted keywords for relevance scoring: rarer/longer/numeric terms weigh more."""
    counts = Counter()
    for text in texts:
        counts.update(content_tokens(text))
    weights = {}
    for word, _ in counts.most_common(60):
        weight = 1.0
        if len(word) >= 9:
            weight = 1.6
        elif len(word) >= 6:
            weight = 1.3
        if _NUMERIC_RE.match(word):
            weight = 1.8 if len(word) == 4 else 1.4  # years are strong signals
        weights[word] = weight
    return weights


def relevance_score(weights, text):
    """0..1 weighted share of the question's keywords present in `text`."""
    if not weights:
        return 0.0
    present = set(content_tokens(text))
    if not present:
        return 0.0
    total = sum(weights.values())
    hit = sum(w for k, w in weights.items() if k in present)
    return min(1.0, hit / total) if total else 0.0


_PARA_SPLIT_RE = re.compile(r"\n\s*\n")


def relevant_excerpt(text, weights, max_chars, head_chars=900):
    """Keep the opening plus the paragraphs that best match the question.

    Long pages usually bury the relevant passage in the middle; naive head
    truncation throws the evidence away and makes the critic flag good answers.
    """
    text = text or ""
    if len(text) <= max_chars:
        return text
    head = text[:head_chars]
    rest = text[head_chars:]
    paragraphs = [p for p in _PARA_SPLIT_RE.split(rest) if p.strip()]
    if not paragraphs:
        return shrink_middle(text, max_chars)
    scored = []
    for idx, para in enumerate(paragraphs):
        if len(para) < 40:
            continue
        score = relevance_score(weights, para[:1200])
        scored.append((score, idx, para))
    scored.sort(key=lambda t: (-t[0], t[1]))
    budget = max_chars - len(head) - 80
    chosen, used = [], 0
    for score, idx, para in scored:
        if used >= budget:
            break
        piece = para[: max(200, budget - used)]
        chosen.append((idx, piece))
        used += len(piece)
    if not chosen:
        return shrink_middle(text, max_chars)
    chosen.sort(key=lambda t: t[0])
    body = "\n\n[...]\n\n".join(p for _, p in chosen)
    return (head + "\n\n[...]\n\n" + body)[:max_chars]


_ABBREVIATIONS = frozenset("""
e.g i.e cf vs approx fig figs no nos dr mr mrs ms prof st al etc u.s u.k u.s.a jr sr
ph.d inc ltd co vol pp ca est dept univ sec sect art para ref refs eq eqs ch chap
jan feb mar apr jun jul aug sep sept oct nov dec mt gen col capt lt sgt adm
""".split())

_SENT_BOUNDARY_RE = re.compile(r"(?<=[.!?\u2026])[\"'\u201d\u2019\)\]]*\s+")


def split_sentences(text):
    """Sentence split that tolerates abbreviations, decimals and [S1] markers."""
    text = (text or "").strip()
    if not text:
        return []
    sentences, start = [], 0
    for match in _SENT_BOUNDARY_RE.finditer(text):
        end = match.start()
        chunk = text[start:match.end()]
        before = text[max(0, end - 12):end + 1].strip()
        tail = re.split(r"[\s(]", before)[-1].rstrip(".!?").lower() if before else ""
        if tail in _ABBREVIATIONS:
            continue
        # Decimals ("3.5") can never match here: the boundary pattern requires
        # whitespace after the period, so only "... 2011. Next" style breaks reach
        # this point and those are genuine sentence boundaries.
        if len(chunk.strip()) < 2:
            continue
        nxt = text[match.end():match.end() + 1]
        if nxt and not (nxt.isupper() or nxt.isdigit() or nxt in "\"'\u201c[*-\u2022#"):
            continue
        sentences.append(chunk.strip())
        start = match.end()
    if start < len(text):
        sentences.append(text[start:].strip())
    # Bullet/heading lines arrive as one blob; split them out so each is auditable.
    out = []
    for sentence in sentences:
        parts = [p.strip() for p in sentence.split("\n") if p.strip()]
        out.extend(parts if len(parts) > 1 else [sentence])
    return [s for s in out if s]


# ---------------- citation markers ----------------

_CITE_RE = re.compile(r"\[\s*[Ss]\s*(\d{1,3})\s*\]")
_CITE_GROUP_RE = re.compile(r"\[\s*[Ss]?\s*\d{1,3}(?:\s*[,;/&]+\s*(?:and\s+)?[Ss]?\s*\d{1,3})+\s*\]")
_CITE_ANY_RE = re.compile(r"\[\s*[Ss]\s*\d{1,3}(?:\s*[,;/&]+\s*(?:and\s+)?[Ss]?\s*\d{1,3})*\s*\]")


def normalize_citation_markers(text):
    """`[S1, S2]`, `[S1,2]`, `[ s3 ]`, `[S1 and S2]` -> `[S1][S2]`."""
    if not text:
        return ""

    def _expand(match):
        ids = re.findall(r"\d{1,3}", match.group(0))
        seen, out = set(), []
        for i in ids:
            n = str(int(i))
            if n not in seen:
                seen.add(n)
                out.append("[S%s]" % n)
        return "".join(out)

    text = _CITE_GROUP_RE.sub(_expand, text)
    text = _CITE_RE.sub(lambda m: "[S%d]" % int(m.group(1)), text)
    text = re.sub(r"(\[S\d{1,3}\])(?:\s*\1)+", r"\1", text)      # [S2][S2] -> [S2]
    text = re.sub(r"\s+(\[S\d{1,3}\])", r" \1", text)
    text = re.sub(r"\s+([.,;:])", r"\1", text)
    return text


def citation_ids(text):
    """Ordered unique source ids referenced by the text: ['S1', 'S4', ...]."""
    seen, out = set(), []
    for match in _CITE_RE.finditer(text or ""):
        sid = "S%d" % int(match.group(1))
        if sid not in seen:
            seen.add(sid)
            out.append(sid)
    return out


def strip_citation_markers(text):
    text = _CITE_ANY_RE.sub("", text or "")
    text = re.sub(r"[ \t]{2,}", " ", text)
    return re.sub(r"\s+([.,;:!?])", r"\1", text).strip()


def rewrite_citations_for_sft(answer, sources, mode=None):
    """Apply SFT_CITATION_MODE to the stored answer.

    The SFT prompt does not contain the source list, so raw [S1] markers teach the
    model to point at nothing. 'named' keeps attribution visible without inventing
    titles; 'strip' removes markers entirely.
    """
    mode = (mode or SFT_CITATION_MODE or "keep").lower()
    if mode == "keep" or not answer:
        return answer
    if mode == "strip":
        return strip_citation_markers(answer)

    by_id = {s["id"]: s for s in sources}

    def _name(sid):
        src = by_id.get(sid)
        if not src:
            return None
        return src.get("domain") or url_base_domain(src.get("url", "")) or None

    def _replace_run(match):
        names, seen = [], set()
        for sid in citation_ids(match.group(0)):
            name = _name(sid)
            if name and name not in seen:
                seen.add(name)
                names.append(name)
        return "[%s]" % "; ".join(names) if names else ""

    run_re = re.compile(r"(?:\[\s*[Ss]\s*\d{1,3}\s*\])+")
    out = run_re.sub(_replace_run, answer)
    out = re.sub(r"[ \t]{2,}", " ", out)
    return re.sub(r"\s+([.,;:!?])", r"\1", out).strip()


# ============================================================
# EVIDENCE ASSEMBLY
# ============================================================

def _source_header(src):
    profile = "tier %d/%s" % (src.get("tier", 0), src.get("kind", "unknown"))
    if src.get("community"):
        profile += "/testimony"
    return "[%s] %s\n      %s | %s | %s | %d chars" % (
        src["id"], src.get("title", "")[:180] or "(untitled)",
        src.get("url", "")[:200], src.get("domain", "?"), profile, len(src.get("text", "")),
    )


def build_evidence_block(sources, weights, total_chars, per_source_chars):
    """Render sources as labelled excerpts, budgeting space by relevance.

    Every source gets a floor so nothing is silently invisible to the model, and
    the remainder is distributed in proportion to how well the source matches the
    research question.
    """
    if not sources:
        return "(no sources)"
    floor = 500
    budget = max(total_chars, floor * len(sources))
    weights_sum = sum(max(0.05, s.get("relevance", 0.0)) for s in sources) or 1.0
    spare = max(0, budget - floor * len(sources))
    blocks = []
    for src in sources:
        share = max(0.05, src.get("relevance", 0.0)) / weights_sum
        allowance = int(min(per_source_chars, floor + spare * share))
        excerpt = relevant_excerpt(src.get("text", ""), weights, allowance)
        blocks.append(
            "%s\n--- EVIDENCE %s START ---\n%s\n--- EVIDENCE %s END ---"
            % (_source_header(src), src["id"], excerpt.strip(), src["id"])
        )
    return "\n\n".join(blocks)


def build_source_index(sources):
    """Compact one-line-per-source list (used where full excerpts do not fit)."""
    return "\n".join(
        "[%s] %s | %s | tier %d/%s%s"
        % (s["id"], (s.get("title") or "")[:120], s.get("domain", "?"),
           s.get("tier", 0), s.get("kind", "?"), " | TESTIMONY" if s.get("community") else "")
        for s in sources
    )


# ============================================================
# AUTOMATED GROUNDING AUDIT
# ============================================================
# Deterministic pre-check run before the critic. It cannot understand meaning,
# so it looks for the failure modes that are mechanically detectable: citations
# pointing at sources that do not exist, numbers and quotations that appear
# nowhere in the cited source, and claim sentences with no lexical connection to
# the evidence they cite. Its findings are handed to the critic (which can judge
# meaning) and also gate saving on their own.

_QUOTE_RE = re.compile(r"[\"\u201c]([^\"\u201c\u201d]{16,320})[\"\u201d]")
_NUMBER_RE = re.compile(r"(?<![\w/])(\d{1,3}(?:[,\u00a0\u202f]\d{3})+|\d+(?:\.\d+)?)\s*(%|percent)?")
_PROPER_NOUN_RE = re.compile(r"\b([A-Z][a-zA-Z0-9'\-]+(?:\s+[A-Z][a-zA-Z0-9'\-]+){1,4})")
_HEADING_RE = re.compile(r"^\s*(?:#+\s|\*\*[^*]+\*\*\s*:?\s*$|[A-Z][A-Z \-/]{6,}:?\s*$)")

HARD_FLAGS = {"unknown_source", "quote_not_verbatim", "number_absent_everywhere", "no_citations"}


def _normalize_for_match(text):
    """Lowercase, unify quotes/dashes/spaces, drop digit group separators."""
    t = (text or "").lower()
    t = t.replace("\u2019", "'").replace("\u2018", "'")
    t = t.replace("\u201c", '"').replace("\u201d", '"')
    t = re.sub(r"[\u2010-\u2015\u2212]", "-", t)
    t = re.sub(r"(?<=\d)[,\u00a0\u202f](?=\d{3}\b)", "", t)
    t = re.sub(r"[^a-z0-9%.\-'\"/: ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _normalize_number(raw):
    n = raw.replace(",", "").replace("\u00a0", "").replace("\u202f", "")
    if "." in n:
        n = n.rstrip("0").rstrip(".") or "0"
    return n


def _number_variants(value):
    """Forms the same quantity plausibly takes in a source document."""
    variants = {value}
    try:
        num = float(value)
    except ValueError:
        return variants
    if num.is_integer():
        i = int(num)
        variants.add(str(i))
        if abs(i) >= 1000:
            variants.add("{:,}".format(i))
            variants.add("{:,}".format(i).replace(",", " "))
        if abs(i) >= 1_000_000 and i % 100_000 == 0:
            variants.add(("%g" % (i / 1_000_000)) + " million")
        if abs(i) >= 1_000_000_000 and i % 100_000_000 == 0:
            variants.add(("%g" % (i / 1_000_000_000)) + " billion")
    else:
        variants.add(("%g" % num))
        variants.add(("%.1f" % num).rstrip("0").rstrip("."))
    return {v for v in variants if v}


def _is_claim_sentence(sentence):
    plain = strip_citation_markers(sentence)
    if _HEADING_RE.match(plain):
        return False
    if len(plain) < 40:
        return False
    return len(content_tokens(plain)) >= 6


def _contains_number(haystack, value):
    for variant in _number_variants(value):
        needle = _normalize_for_match(variant)
        if needle and needle in haystack:
            return True
    return False


def grounding_audit(answer, sources):
    """Deterministic evidence audit. Returns a report dict; never raises."""
    report = {
        "n_sentences": 0, "n_claim_sentences": 0, "n_cited_sentences": 0,
        "cited_ids": [], "unknown_ids": [], "unused_ids": [],
        "flags": [], "hard_flags": 0, "flagged_sentences": 0,
        "flagged_fraction": 0.0, "citation_density": 0.0,
        "distinct_sources": 0, "verdict": "clean",
    }
    try:
        answer = answer or ""
        by_id, norm_text, norm_tokens = {}, {}, {}
        for src in sources:
            by_id[src["id"]] = src
            norm = _normalize_for_match(src.get("text", ""))
            norm_text[src["id"]] = norm
            norm_tokens[src["id"]] = set(content_tokens(src.get("text", "")))
        all_norm = " \n ".join(norm_text.values())

        sentences = split_sentences(answer)
        report["n_sentences"] = len(sentences)
        flagged_idx, flags = set(), []

        def flag(index, kind, detail, sentence):
            flags.append({
                "kind": kind, "detail": detail[:220],
                "sentence": strip_citation_markers(sentence)[:240],
            })
            flagged_idx.add(index)

        all_ids = citation_ids(answer)
        report["cited_ids"] = all_ids
        report["unknown_ids"] = [i for i in all_ids if i not in by_id]
        report["unused_ids"] = [s["id"] for s in sources if s["id"] not in set(all_ids)]
        report["distinct_sources"] = len([i for i in all_ids if i in by_id])

        for index, sentence in enumerate(sentences):
            if not _is_claim_sentence(sentence):
                continue
            report["n_claim_sentences"] += 1
            ids = citation_ids(sentence)
            known = [i for i in ids if i in by_id]
            plain = strip_citation_markers(sentence)
            norm_sentence = _normalize_for_match(plain)

            for bad in (i for i in ids if i not in by_id):
                flag(index, "unknown_source",
                     "%s is not among the %d supplied sources" % (bad, len(sources)), sentence)

            numbers = [
                _normalize_number(m.group(1))
                for m in _NUMBER_RE.finditer(plain)
                if len(m.group(1).replace(",", "").replace(".", "")) >= 2
            ]
            quotes = [q.strip() for q in _QUOTE_RE.findall(plain) if len(q.split()) >= 5]

            if not ids:
                report["n_cited_sentences"] += 0
                if numbers or quotes:
                    flag(index, "uncited_specific_claim",
                         "numbers/quotations with no [S] citation", sentence)
                continue
            report["n_cited_sentences"] += 1
            if not known:
                continue

            cited_blob = " \n ".join(norm_text[i] for i in known)
            for value in numbers[:8]:
                if _contains_number(cited_blob, value):
                    continue
                if _contains_number(all_norm, value):
                    flag(index, "number_in_wrong_source",
                         "value %s appears in the evidence but not in %s" % (value, "/".join(known)),
                         sentence)
                else:
                    flag(index, "number_absent_everywhere",
                         "value %s appears in no supplied source" % value, sentence)

            for quote in quotes[:4]:
                needle = _normalize_for_match(quote)
                if needle and needle in cited_blob:
                    continue
                if needle and needle in all_norm:
                    flag(index, "quote_in_wrong_source",
                         "quotation found in the evidence but not in %s" % "/".join(known), sentence)
                else:
                    flag(index, "quote_not_verbatim",
                         "quotation does not appear verbatim in any source: \"%s\"" % quote[:90],
                         sentence)

            sentence_tokens = set(content_tokens(plain))
            if sentence_tokens:
                cited_tokens = set()
                for i in known:
                    cited_tokens |= norm_tokens[i]
                overlap = len(sentence_tokens & cited_tokens) / len(sentence_tokens)
                if overlap < 0.30:
                    flag(index, "weak_lexical_overlap",
                         "only %d%% of the sentence's content words occur in %s"
                         % (round(overlap * 100), "/".join(known)), sentence)

            if norm_sentence and len(norm_sentence) > 80 and norm_sentence in cited_blob:
                flag(index, "verbatim_copy",
                     "sentence is copied verbatim from the source rather than written", sentence)

        if report["n_claim_sentences"] and not all_ids:
            flags.append({"kind": "no_citations", "detail": "the answer cites nothing", "sentence": ""})
            flagged_idx.add(-1)

        report["flags"] = flags
        report["hard_flags"] = sum(1 for f in flags if f["kind"] in HARD_FLAGS)
        report["flagged_sentences"] = len(flagged_idx)
        denominator = max(1, report["n_claim_sentences"])
        report["flagged_fraction"] = round(len(flagged_idx) / denominator, 3)
        report["citation_density"] = round(report["n_cited_sentences"] / denominator, 3)
        if report["unknown_ids"] or report["hard_flags"] >= 2:
            report["verdict"] = "bad"
        elif report["flagged_fraction"] > GROUNDING_MAX_FLAGGED_FRACTION or report["hard_flags"]:
            report["verdict"] = "suspect"
        else:
            report["verdict"] = "clean"
    except Exception as e:  # an audit bug must never abort a cycle
        log.warning("grounding_audit failed: %s: %s", type(e).__name__, str(e)[:200])
        report["verdict"] = "unknown"
        report["error"] = "%s: %s" % (type(e).__name__, str(e)[:120])
    return report


def format_grounding_report(report, limit=14):
    """Human/model-readable digest of the audit for the critic and repair prompts."""
    if not report or report.get("verdict") == "unknown":
        return "(automated grounding pre-check unavailable)"
    lines = [
        "AUTOMATED GROUNDING PRE-CHECK (deterministic, may contain false positives):",
        "  verdict=%s  claim_sentences=%d  cited=%d  citation_density=%.2f  flagged=%.0f%%"
        % (report["verdict"], report["n_claim_sentences"], report["n_cited_sentences"],
           report["citation_density"], report["flagged_fraction"] * 100),
    ]
    if report["unknown_ids"]:
        lines.append("  CITES SOURCES THAT DO NOT EXIST: %s" % ", ".join(report["unknown_ids"]))
    if report["unused_ids"]:
        lines.append("  sources never cited: %s" % ", ".join(report["unused_ids"][:12]))
    if not report["flags"]:
        lines.append("  no mechanical problems detected.")
        return "\n".join(lines)
    lines.append("  findings (verify each one against the excerpts):")
    for item in report["flags"][:limit]:
        lines.append("   - [%s] %s" % (item["kind"], item["detail"]))
        if item["sentence"]:
            lines.append("     sentence: %s" % item["sentence"][:200])
    if len(report["flags"]) > limit:
        lines.append("   - ... and %d more findings" % (len(report["flags"]) - limit))
    return "\n".join(lines)


# ============================================================
# CITATION STATISTICS  (source-quality gates)
# ============================================================

def citation_stats(answer, sources):
    """Which sources an answer actually leans on, and how good they are."""
    by_id = {s["id"]: s for s in sources}
    ids = [i for i in citation_ids(answer) if i in by_id]
    cited = [by_id[i] for i in ids]
    domains = {s.get("domain", "") for s in cited if s.get("domain")}
    tiers = [int(s.get("tier", 0)) for s in cited]
    tier5 = [s for s in cited if counts_as_tier5(s)]
    return {
        "cited_ids": ids,
        "n_cited": len(cited),
        "domains": sorted(domains),
        "n_domains": len(domains),
        "best_tier": max(tiers) if tiers else 0,
        "mean_tier": round(sum(tiers) / len(tiers), 2) if tiers else 0.0,
        "n_high_tier": sum(1 for t in tiers if t >= MIN_CITED_SOURCE_QUALITY),
        "n_tier5": len(tier5),
        "tier5_domains": sorted({s.get("domain", "") for s in tier5}),
        "kinds": sorted({s.get("kind", "") for s in cited}),
    }


# ============================================================
# PROMPTS (continued)
# ============================================================

REPAIR_SYSTEM = r"""
You are AutoLab's answer-repair engineer.
You are given a research question, the verbatim EVIDENCE EXCERPTS, an answer, a
hostile critic's findings and an automated grounding report.

Rewrite the answer so that every objection is resolved.
Rules:
- Fix, do not defend. If a claim cannot be supported by the excerpts, delete it
  or downgrade it explicitly ("the supplied sources do not establish ...").
- Remove or correct every number, date and quotation that does not appear in the
  cited source. Copy survivors exactly as the source writes them.
- Repair citations: one source id per bracket, [S1][S2], never [S1, S2], placed
  at the end of the sentence it supports. Never cite a source id that was not
  supplied.
- Treat forum, Reddit, Stack Exchange and mailing-list material as testimony:
  write "users reported X" rather than "X happened".
- Keep or improve the parts the critic did not object to. Do not shorten the
  answer below 1500 characters and do not pad it with filler.
- Preserve the structure: direct answer first, then evidence, then an explicit
  statement of uncertainty or counter-evidence.
- Write in English only.

Return ONLY the corrected answer text. No preamble, no JSON, no commentary,
no list of the changes you made.
"""

CONCLUSION_SYSTEM = r"""
You are AutoLab's research log keeper.
Summarise what this research cycle established, in at most 120 words.
State: what is now supported by evidence, what remains open or disputed, and the
single strongest source. Do not invent anything beyond the supplied material.
Return ONLY the summary paragraph, no headings and no JSON.
"""

SFT_SYSTEM_PROMPT = _env_str(
    "AUTOLAB_SFT_SYSTEM",
    "You are a rigorous research assistant. Answer from evidence, "
    "separate fact from allegation and interpretation, quantify uncertainty, and "
    "state plainly when the evidence does not settle the question.",
)

PLANNER_RETRY_HINT = (
    "\nThe previous attempt was rejected as a duplicate or as too vague. Choose a "
    "materially DIFFERENT subject - a different event, document, program, dataset "
    "or time period - and make the question narrower and more concrete.\n"
)


# ============================================================
# RESEARCH PLANNER
# ============================================================

def _recent_attempt_questions(limit=PLANNER_AVOID_HISTORY):
    seen, out = set(), []
    for entry in reversed(ATTEMPTED_QUESTIONS):
        q = str(entry.get("question") or "").strip()
        key = normalize_question(q)
        if q and key not in seen:
            seen.add(key)
            out.append(q)
        if len(out) >= limit:
            break
    for question, _tokens in reversed(ACCEPTED_TOKENSETS):
        key = normalize_question(question)
        if key not in seen:
            seen.add(key)
            out.append(question)
        if len(out) >= limit:
            break
    return out[:limit]


def _recent_areas(window):
    areas = []
    for entry in reversed(ATTEMPTED_QUESTIONS):
        area = str(entry.get("area") or "").strip().lower()
        if area:
            areas.append(area)
        if len(areas) >= window:
            break
    return areas


def pick_area():
    """Suggest the next area, rotating away from whatever the planner is stuck on."""
    recent = _recent_areas(STUCK_AREA_WINDOW)
    counts = Counter(recent)
    banned = set()
    for area, n in counts.items():
        if n >= STUCK_AREA_THRESHOLD:
            banned.add(area)
            for cluster in RELATED_AREA_CLUSTERS:
                if any(area == u.lower() for u in cluster):
                    banned.update(u.lower() for u in cluster)
    pool = [a for a in CORE_AREAS if a.lower() not in banned]
    if not pool:
        pool = list(CORE_AREAS)
    if banned:
        log.info("Area rotation: avoiding %s (over-used in the last %d cycles).",
                 sorted(banned), STUCK_AREA_WINDOW)
    recent_counts = Counter(_recent_areas(AREA_SATURATION_WINDOW))
    weights = [1.0 / (1.0 + 2.0 * recent_counts.get(a.lower(), 0)) for a in pool]
    try:
        return random.choices(pool, weights=weights, k=1)[0]
    except Exception:
        return random.choice(pool)


def _looks_specific(question):
    """Cheap vagueness filter: reject 'tell me about volcanoes'-style questions."""
    q = (question or "").strip()
    if not (25 <= len(q) <= 400):
        return False
    tokens = content_tokens(q)
    if len(tokens) < 5:
        return False
    vague_openers = ("tell me about", "what is", "what are", "explain", "overview of",
                     "introduction to", "discuss", "describe")
    lowered = q.lower()
    if any(lowered.startswith(v) for v in vague_openers) and len(tokens) < 9:
        return False
    # A concrete question almost always pins something down: a name, an acronym,
    # a number, a year, or a quoted designation.
    has_anchor = bool(
        re.search(r"\b(19|20)\d{2}\b", q)
        or re.search(r"\b[A-Z]{2,}(?:-\d+)?\b", q)
        or re.search(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+", q)
        or re.search(r"\d", q)
    )
    return has_anchor or len(tokens) >= 8


def _ensure_community_query(queries, question):
    if any(any(m in q.lower() for m in COMMUNITY_QUERY_MARKERS) for q in queries):
        return queries
    core = " ".join(
        [w for w in re.findall(r"[A-Za-z0-9\-]{3,}", question) if w.lower() not in STOPWORDS][:6]
    )
    if core:
        queries.append(clean_query("%s reddit discussion" % core))
        if len(queries) < MAX_QUERIES:
            queries.append(clean_query("%s forum first-hand report" % core))
    return queries


def _normalize_plan(data, suggested_area):
    """Coerce raw planner JSON into a validated plan, or return None."""
    if not isinstance(data, dict):
        return None
    question = clean_text(data.get("research_question") or data.get("question") or "").strip()
    question = re.sub(r"\s+", " ", question).strip(" \"'")
    if not _looks_specific(question):
        return None

    raw_queries = data.get("search_queries") or data.get("queries") or []
    queries, seen = [], set()
    for q in as_str_list(raw_queries, limit=MAX_QUERIES * 2, maxlen=250):
        q = clean_query(q)
        key = q.lower()
        if len(q) >= 6 and key not in seen:
            seen.add(key)
            queries.append(q)
    if len(queries) < 2:
        base = clean_query(question)
        for extra in (base, base + " primary source document", base + " analysis evidence"):
            if extra and extra.lower() not in seen:
                seen.add(extra.lower())
                queries.append(extra)
    queries = _ensure_community_query(queries, question)[:MAX_QUERIES]

    area = clean_text(data.get("area") or "").strip() or suggested_area
    known = {a.lower(): a for a in CORE_AREAS + GENERAL_AREAS}
    area = known.get(area.lower(), area[:80])

    return {
        "area": area,
        "research_question": question,
        "search_queries": queries,
        "learning_goal": clean_text(data.get("learning_goal") or "")[:600],
        "evidence_requirements": as_str_list(data.get("evidence_requirements"), limit=8),
        "counterargument_target": clean_text(data.get("counterargument_target") or "")[:600],
        "knowledge_gap": clean_text(data.get("knowledge_gap") or "")[:600],
        "risk_notes": as_str_list(data.get("risk_notes"), limit=6),
    }


def is_duplicate_question(question):
    """Exact-normalized and near-duplicate (token Jaccard) check against history."""
    key = normalize_question(question)
    if not key:
        return "empty question"
    if key in ACCEPTED_NORMALIZED:
        return "already in the dataset"
    for entry in ATTEMPTED_QUESTIONS[-400:]:
        if normalize_question(entry.get("question", "")) == key:
            return "already attempted"
    tokens = token_set(question)
    for prior, prior_tokens in ACCEPTED_TOKENSETS[-1500:]:
        if jaccard(tokens, prior_tokens) >= NEAR_DUP_JACCARD:
            return "near-duplicate of: %s" % prior[:120]
    subject = set(content_tokens(question))
    if subject:
        hits = sum(
            1 for _prior, prior_tokens in ACCEPTED_TOKENSETS[-200:]
            if len(subject & prior_tokens) >= max(3, int(0.6 * len(subject)))
        )
        if hits >= SUBJECT_REPEAT_HITS:
            return "subject covered %d times already" % hits
    return ""


def plan_research(forced_topic=None, attempts=3):
    """Ask the model for the next research question. Returns a plan dict or None."""
    if forced_topic:
        plan = _normalize_plan(
            {"research_question": forced_topic, "area": "operator-specified",
             "search_queries": [forced_topic]},
            "operator-specified",
        )
        if plan is None:
            plan = {
                "area": "operator-specified",
                "research_question": clean_text(forced_topic).strip(),
                "search_queries": _ensure_community_query([clean_query(forced_topic)], forced_topic),
                "learning_goal": "", "evidence_requirements": [],
                "counterargument_target": "", "knowledge_gap": "", "risk_notes": [],
            }
        log.info("PLAN (operator-specified): %s", plan["research_question"])
        return plan

    suggested = pick_area()
    avoid = _recent_attempt_questions()
    stats = read_dataset_stats()
    rejected_here = []

    for attempt in range(1, attempts + 1):
        check_stop()
        avoid_block = "\n".join("- %s" % q[:200] for q in (avoid + rejected_here)[:PLANNER_AVOID_HISTORY])
        user_prompt = (
            "DATE (UTC): %s\n"
            "DATASET SO FAR: %d accepted SFT examples, %d preference pairs.\n"
            "SUGGESTED AREA: %s\n"
            "ALLOWED AREAS (pick this one or another from the list):\n%s\n\n"
            "DO NOT propose any question that repeats or paraphrases these:\n%s\n\n"
            "Produce ONE research question that public web evidence can actually settle "
            "or meaningfully constrain, plus %d-%d search queries (at least one aimed at "
            "community/forum testimony, and never using the site: operator).%s"
        ) % (
            now_iso()[:19], stats["sft_examples"], stats["preference_examples"],
            suggested, ", ".join(CORE_AREAS),
            avoid_block or "- (nothing yet)",
            max(3, min(5, MAX_QUERIES)), MAX_QUERIES,
            PLANNER_RETRY_HINT if attempt > 1 else "",
        )
        raw = call_model(
            PLANNER_SYSTEM, user_prompt,
            temperature=min(1.0, SEARCH_TEMPERATURE + 0.15 * (attempt - 1)),
            max_tokens=PLANNER_MAX_TOKENS, force_json=True, role="planner",
        )
        data = safe_json_loads(raw, fallback={}, want_keys=("research_question", "search_queries", "area"))
        plan = _normalize_plan(data, suggested)
        if plan is None:
            log.warning("Planner attempt %d/%d produced no usable question.", attempt, attempts)
            continue
        duplicate = is_duplicate_question(plan["research_question"])
        if duplicate:
            log.info("Planner attempt %d rejected (%s): %s", attempt, duplicate,
                     plan["research_question"][:120])
            rejected_here.append(plan["research_question"])
            suggested = pick_area()
            continue
        log.info("PLAN [%s]: %s", plan["area"], plan["research_question"])
        log.info("QUERIES: %s", " | ".join(plan["search_queries"]))
        return plan

    log.warning("Planner produced no novel question in %d attempts; skipping this cycle.", attempts)
    return None


# ============================================================
# SOURCE COLLECTION
# ============================================================

def _is_community_query(query):
    q = (query or "").lower()
    return any(marker in q for marker in COMMUNITY_QUERY_MARKERS)


def run_searches(plan):
    """Execute the plan's queries and return de-duplicated, pre-filtered candidates."""
    candidates, seen_urls = [], set()
    dropped = Counter()
    for query in plan["search_queries"][:MAX_QUERIES]:
        check_stop()
        try:
            results = web_search(query, RESULTS_PER_QUERY)
        except StopRequested:
            raise
        except Exception as e:
            log.warning("Search failed for %r: %s: %s", query, type(e).__name__, str(e)[:160])
            continue
        community_query = _is_community_query(query)
        for result in results:
            url = normalize_url(result.get("url", ""))
            if not url or url in seen_urls:
                dropped["duplicate"] += 1
                continue
            if not is_public_url(url):
                dropped["non-public"] += 1
                continue
            profile = source_profile(url)
            if profile["tier"] < MIN_FETCH_TIER:
                dropped["tier<%d" % MIN_FETCH_TIER] += 1
                continue
            seen_urls.add(url)
            candidates.append({
                "url": url,
                "title": result.get("title", ""),
                "snippet": result.get("snippet", ""),
                "query": query,
                "from_community_query": community_query,
                "search_backend": result.get("search_backend", ""),
                **profile,
            })
    if dropped:
        log.info("Search candidates dropped: %s", dict(dropped))
    return candidates


def rank_candidates(candidates, weights):
    """Score by question-relevance and source tier, then enforce diversity caps."""
    for candidate in candidates:
        text = "%s %s" % (candidate.get("title", ""), candidate.get("snippet", ""))
        candidate["snippet_relevance"] = relevance_score(weights, text)
        tier_score = candidate["tier"] / 10.0
        bonus = 0.0
        if candidate.get("from_community_query") and candidate.get("community"):
            bonus += 0.08
        if re.search(r"\.pdf($|\?)", candidate["url"], re.I):
            bonus += 0.04
        candidate["score"] = round(0.55 * candidate["snippet_relevance"] + 0.45 * tier_score + bonus, 4)

    candidates.sort(key=lambda c: (-c["score"], -c["tier"]))
    per_domain, ordered = Counter(), []
    for candidate in candidates:
        domain = candidate.get("domain") or candidate["url"]
        if per_domain[domain] >= MAX_SOURCES_PER_DOMAIN:
            continue
        per_domain[domain] += 1
        ordered.append(candidate)

    # Reserve slots near the front for community testimony so the fetch budget is
    # not exhausted by institutional sources before any forum page is tried.
    community = [c for c in ordered if c.get("community")][:COMMUNITY_SOURCE_QUOTA]
    if community:
        reserved = {c["url"] for c in community}
        rest = [c for c in ordered if c["url"] not in reserved]
        head, tail = rest[: max(0, MAX_FETCHED_SOURCES - len(community))], rest[MAX_FETCHED_SOURCES:]
        merged, ci = [], 0
        for i, item in enumerate(head):
            merged.append(item)
            if ci < len(community) and (i + 1) % 3 == 0:
                merged.append(community[ci])
                ci += 1
        merged.extend(community[ci:])
        merged.extend(tail)
        ordered = merged
    return ordered


def _content_fingerprint(text):
    """Cheap near-duplicate key so mirrors of the same article are not both kept."""
    normalized = re.sub(r"\s+", " ", (text or "")[:6000].lower())
    return zlib.crc32(normalized.encode("utf-8", "replace")) & 0xFFFFFFFF


def collect_sources(plan, weights, max_sources=MAX_FETCHED_SOURCES,
                    deadline_seconds=FETCH_DEADLINE_SECONDS):
    """Search, fetch and filter evidence. Returns (sources, diagnostics)."""
    diagnostics = Counter()
    candidates = run_searches(plan)
    diagnostics["candidates"] = len(candidates)
    if not candidates:
        return [], diagnostics
    ordered = rank_candidates(candidates, weights)
    # Fetch more than needed: block pages, PDFs and timeouts kill a large share.
    attempt_list = ordered[: max(max_sources * 3, max_sources + 12)]
    deadline = time.monotonic() + deadline_seconds
    accepted, fingerprints, per_domain = [], set(), Counter()
    executor = ThreadPoolExecutor(max_workers=max(1, FETCH_WORKERS))
    futures = {}
    try:
        for candidate in attempt_list:
            futures[executor.submit(fetch_page_ex, candidate["url"])] = candidate
        for future in as_completed(futures, timeout=max(1.0, deadline - time.monotonic())):
            candidate = futures[future]
            try:
                result = future.result()
            except Exception as e:
                diagnostics["fetch-error"] += 1
                log.debug("fetch raised for %s: %s", candidate["url"], str(e)[:120])
                continue
            if not result.get("ok"):
                diagnostics["fail:%s" % str(result.get("error", "?"))[:28]] += 1
                continue
            text = result["text"]
            relevance = relevance_score(weights, text[:20000])
            if relevance < MIN_SOURCE_RELEVANCE:
                diagnostics["off-topic"] += 1
                continue
            fingerprint = _content_fingerprint(text)
            if fingerprint in fingerprints:
                diagnostics["duplicate-content"] += 1
                continue
            final_url = normalize_url(result.get("final_url") or candidate["url"])
            profile = source_profile(final_url)
            domain = profile["domain"] or candidate.get("domain", "")
            if per_domain[domain] >= MAX_SOURCES_PER_DOMAIN:
                diagnostics["domain-cap"] += 1
                continue
            if profile["tier"] < MIN_FETCH_TIER:
                diagnostics["tier-after-redirect"] += 1
                continue
            fingerprints.add(fingerprint)
            per_domain[domain] += 1
            accepted.append({
                "url": final_url,
                "requested_url": candidate["url"],
                "title": candidate.get("title", "") or final_url,
                "snippet": candidate.get("snippet", ""),
                "query": candidate.get("query", ""),
                "text": text,
                "chars": len(text),
                "content_type": result.get("content_type", ""),
                "relevance": round(relevance, 4),
                "search_backend": candidate.get("search_backend", ""),
                **profile,
            })
            diagnostics["accepted"] += 1
            if len(accepted) >= max_sources or time.monotonic() >= deadline:
                break
    except FuturesTimeout:
        log.warning("Source fetch phase hit its %ds deadline with %d sources.",
                    deadline_seconds, len(accepted))
        diagnostics["deadline"] += 1
    except StopRequested:
        raise
    finally:
        try:
            executor.shutdown(wait=False, cancel_futures=True)
        except TypeError:  # Python 3.8
            executor.shutdown(wait=False)

    # Best evidence first, so [S1] is the strongest source in the prompt.
    accepted.sort(key=lambda s: (-(s["tier"]), -s["relevance"]))
    for index, source in enumerate(accepted, 1):
        source["id"] = "S%d" % index
    log.info(
        "SOURCES: %d kept from %d candidates | tiers %s | domains %d | %s",
        len(accepted), len(candidates),
        sorted({s["tier"] for s in accepted}, reverse=True),
        len({s["domain"] for s in accepted}),
        dict(diagnostics),
    )
    return accepted, diagnostics


# ============================================================
# PIPELINE STAGE: ANSWER GENERATION
# ============================================================

def _plan_header(plan):
    lines = ["RESEARCH QUESTION: %s" % plan["research_question"],
             "AREA: %s" % plan.get("area", "")]
    if plan.get("learning_goal"):
        lines.append("LEARNING GOAL: %s" % plan["learning_goal"])
    if plan.get("evidence_requirements"):
        lines.append("EVIDENCE REQUIREMENTS: %s" % "; ".join(plan["evidence_requirements"][:6]))
    if plan.get("counterargument_target"):
        lines.append("COUNTERARGUMENT TARGET: %s" % plan["counterargument_target"])
    if plan.get("knowledge_gap"):
        lines.append("KNOWLEDGE GAP: %s" % plan["knowledge_gap"])
    return "\n".join(lines)


def answer_quality_signals(answer, sources):
    """Cheap, model-free signals used to rank candidates before the critic runs."""
    stats = citation_stats(answer, sources)
    audit = grounding_audit(answer, sources)
    length_score = min(1.0, len(answer) / 2500.0)
    citation_score = min(1.0, stats["n_cited"] / max(1.0, float(MIN_CITED_SOURCES + 1)))
    domain_score = min(1.0, stats["n_domains"] / max(1.0, float(MIN_CITED_DOMAINS + 1)))
    tier_score = min(1.0, stats["best_tier"] / 10.0)
    density = min(1.0, audit["citation_density"] / 0.6) if audit["citation_density"] else 0.0
    penalty = min(1.0, audit["flagged_fraction"]) * 0.9 + min(1.0, audit["hard_flags"] / 3.0) * 0.6
    score = (0.18 * length_score + 0.22 * citation_score + 0.16 * domain_score
             + 0.20 * tier_score + 0.24 * density) - penalty * 0.5
    return {"score": round(score, 4), "stats": stats, "audit": audit, "chars": len(answer)}


def generate_candidate(plan, sources, weights, strict=False):
    """Produce one candidate answer (citations normalised). Returns '' on failure."""
    header = _plan_header(plan)
    index_block = build_source_index(sources)
    scaffold = (
        "%s\n\nSOURCE INDEX (cite by these ids only):\n%s\n\nEVIDENCE EXCERPTS:\n%s\n\n"
        "TASK:\nAnswer the research question using only the evidence above. Cite every "
        "load-bearing claim with the matching [S..] id at the end of its sentence. "
        "Treat forum and social material as testimony about what people reported, not "
        "as established fact. Close with an explicit statement of what remains "
        "uncertain or what the evidence does not settle.\n"
    )
    if strict:
        scaffold += (
            "\nTHE PREVIOUS ATTEMPT WAS TOO SHORT OR TOO THIN. This time write at least "
            "1800 characters with at least %d separately cited factual claims drawn from "
            "at least %d different sources. Do not pad with restatements.\n"
            % (MIN_CITED_CLAIMS + 1, MIN_CITED_DOMAINS)
        )
    fixed = len(CANDIDATE_SYSTEM) + len(header) + len(index_block) + len(scaffold) + len(AUTOLAB_MEMORY)
    budget = int(prompt_budget_chars(CANDIDATE_MAX_TOKENS, fixed) * 0.94)
    evidence = build_evidence_block(sources, weights, budget, MAX_SOURCE_CHARS)
    user_prompt = scaffold % (header, index_block, evidence)

    raw = call_model(
        CANDIDATE_SYSTEM, user_prompt,
        temperature=GENERATION_TEMPERATURE if not strict else max(0.2, GENERATION_TEMPERATURE - 0.2),
        max_tokens=CANDIDATE_MAX_TOKENS, role="candidate",
    )
    answer = normalize_citation_markers(clean_model_text(raw))
    # Models sometimes prefix an acknowledgement; drop it if a heading follows.
    answer = re.sub(r"^(?:sure|certainly|here(?:'s| is)[^\n]{0,60})[:.]\s*\n+", "", answer, flags=re.I)
    return answer.strip()


def produce_answer(plan, sources, weights):
    """Generate CANDIDATES_PER_TASK answers and return the strongest one."""
    best, best_signals = "", None
    for attempt in range(max(1, CANDIDATES_PER_TASK)):
        check_stop()
        answer = generate_candidate(plan, sources, weights)
        if len(answer) < RETRY_ANSWER_CHARS:
            log.warning("Candidate %d was %d chars; retrying with stricter instructions.",
                        attempt + 1, len(answer))
            retry = generate_candidate(plan, sources, weights, strict=True)
            if len(retry) > len(answer):
                answer = retry
        if not answer:
            continue
        signals = answer_quality_signals(answer, sources)
        log.info("CANDIDATE %d: %d chars, %d cited sources, density %.2f, flagged %.0f%%, score %.3f",
                 attempt + 1, signals["chars"], signals["stats"]["n_cited"],
                 signals["audit"]["citation_density"], signals["audit"]["flagged_fraction"] * 100,
                 signals["score"])
        if best_signals is None or signals["score"] > best_signals["score"]:
            best, best_signals = answer, signals
    return best, best_signals


# ============================================================
# PIPELINE STAGE: CRITIC
# ============================================================

CRITIC_SCHEMA = """{
 "accuracy_score": 0.0,
 "citation_score": 0.0,
 "relevance_score": 0.0,
 "quality_score": 0.0,
 "confidence": 0.0,
 "verdict": "accept | revise | reject",
 "issues": ["..."],
 "unsupported_claims": ["..."],
 "citation_errors": ["..."],
 "missing_counter_evidence": ["..."],
 "required_fixes": ["..."],
 "strongest_source": "S1",
 "summary": "one sentence"
}"""


def _normalize_critique(data):
    data = data if isinstance(data, dict) else {}
    verdict = str(data.get("verdict", "")).strip().lower()
    if verdict not in {"accept", "revise", "reject"}:
        verdict = ""
    critique = {
        "accuracy_score": safe_float(data.get("accuracy_score"), 0.0, 0.0, 10.0),
        "citation_score": safe_float(data.get("citation_score"), 0.0, 0.0, 10.0),
        "relevance_score": safe_float(data.get("relevance_score"), 0.0, 0.0, 10.0),
        "quality_score": safe_float(data.get("quality_score"), 0.0, 0.0, 10.0),
        "confidence": safe_float(data.get("confidence"), 0.0, 0.0, 1.0),
        "issues": as_str_list(data.get("issues"), limit=20),
        "unsupported_claims": as_str_list(data.get("unsupported_claims"), limit=20),
        "citation_errors": as_str_list(data.get("citation_errors"), limit=20),
        "missing_counter_evidence": as_str_list(data.get("missing_counter_evidence"), limit=10),
        "required_fixes": as_str_list(data.get("required_fixes"), limit=20),
        "strongest_source": clean_text(data.get("strongest_source") or "")[:20],
        "summary": clean_text(data.get("summary") or "")[:400],
        "parsed": bool(data),
    }
    # Some models return confidence on a 0-10 scale.
    if critique["confidence"] > 1.0:
        critique["confidence"] = min(1.0, critique["confidence"] / 10.0)
    if not verdict:
        mean = (critique["accuracy_score"] + critique["citation_score"]
                + critique["relevance_score"] + critique["quality_score"]) / 4.0
        verdict = "accept" if mean >= 8.0 and not critique["required_fixes"] else (
            "revise" if mean >= 5.0 else "reject")
    critique["verdict"] = verdict
    return critique


def critique_answer(plan, answer, sources, weights, audit):
    """Hostile fact-check against real excerpts plus the automated audit."""
    header = _plan_header(plan)
    index_block = build_source_index(sources)
    grounding = format_grounding_report(audit)
    scaffold = (
        "%s\n\nSOURCE INDEX:\n%s\n\n%s\n\nPROPOSED ANSWER:\n<<<ANSWER\n%s\nANSWER>>>\n\n"
        "EVIDENCE EXCERPTS:\n%s\n\n"
        "Return exactly this JSON structure, with your own honest values:\n%s\n"
    )
    fixed = (len(CRITIC_SYSTEM) + len(header) + len(index_block) + len(grounding)
             + len(answer) + len(scaffold) + len(CRITIC_SCHEMA))
    budget = min(CRITIC_EXCERPT_TOTAL_CHARS, int(prompt_budget_chars(CRITIC_MAX_TOKENS, fixed) * 0.92))
    evidence = build_evidence_block(sources, weights, budget, CRITIC_EXCERPT_CHARS_PER_SOURCE)
    user_prompt = scaffold % (header, index_block, grounding, answer, evidence, CRITIC_SCHEMA)

    raw = call_model(
        CRITIC_SYSTEM, user_prompt, temperature=CRITIC_TEMPERATURE,
        max_tokens=CRITIC_MAX_TOKENS, force_json=True, role="critic",
    )
    data = safe_json_loads(
        raw, fallback={}, want_keys=("accuracy_score", "citation_score", "required_fixes", "verdict"),
    )
    critique = _normalize_critique(data)
    if not data:
        critique["parsed"] = False
        critique["issues"] = ["critic output was unparseable"] + critique["issues"]
        critique["verdict"] = "reject"
        log.warning("Critic returned unparseable output; treating the example as rejected.")
    log.info(
        "CRITIC: verdict=%s accuracy=%.1f citation=%.1f relevance=%.1f quality=%.1f conf=%.2f "
        "issues=%d fixes=%d",
        critique["verdict"], critique["accuracy_score"], critique["citation_score"],
        critique["relevance_score"], critique["quality_score"], critique["confidence"],
        len(critique["issues"]), len(critique["required_fixes"]),
    )
    return critique


# ============================================================
# PIPELINE STAGE: REPAIR
# ============================================================

def repair_answer(plan, answer, sources, weights, critique, audit):
    """Rewrite the answer to satisfy the critic. Returns '' when repair is not usable."""
    header = _plan_header(plan)
    findings = []
    for label, key in (("REQUIRED FIXES", "required_fixes"), ("UNSUPPORTED CLAIMS", "unsupported_claims"),
                       ("CITATION ERRORS", "citation_errors"), ("ISSUES", "issues"),
                       ("MISSING COUNTER-EVIDENCE", "missing_counter_evidence")):
        items = critique.get(key) or []
        if items:
            findings.append("%s:\n%s" % (label, "\n".join("- %s" % i for i in items[:12])))
    critic_block = "\n\n".join(findings) or "CRITIC: no explicit findings."
    grounding = format_grounding_report(audit)
    index_block = build_source_index(sources)
    scaffold = (
        "%s\n\nSOURCE INDEX (cite by these ids only):\n%s\n\nCURRENT ANSWER:\n<<<ANSWER\n%s\nANSWER>>>\n\n"
        "%s\n\n%s\n\nEVIDENCE EXCERPTS:\n%s\n\nReturn ONLY the corrected answer.\n"
    )
    fixed = (len(REPAIR_SYSTEM) + len(header) + len(index_block) + len(answer)
             + len(critic_block) + len(grounding) + len(scaffold) + len(AUTOLAB_MEMORY))
    budget = min(REPAIR_EXCERPT_TOTAL_CHARS, int(prompt_budget_chars(CANDIDATE_MAX_TOKENS, fixed) * 0.92))
    evidence = build_evidence_block(sources, weights, budget, REPAIR_EXCERPT_CHARS_PER_SOURCE)
    user_prompt = scaffold % (header, index_block, answer, critic_block, grounding, evidence)

    raw = call_model(
        REPAIR_SYSTEM, user_prompt, temperature=max(0.1, GENERATION_TEMPERATURE - 0.25),
        max_tokens=CANDIDATE_MAX_TOKENS, role="repair",
    )
    repaired = normalize_citation_markers(clean_model_text(raw)).strip()
    repaired = re.sub(r"^<<<ANSWER\s*", "", repaired)
    repaired = re.sub(r"\s*ANSWER>>>$", "", repaired).strip()
    if len(repaired) < MIN_ANSWER_CHARS:
        log.warning("Repair produced %d chars (< %d); keeping the original answer.",
                    len(repaired), MIN_ANSWER_CHARS)
        return ""
    if len(repaired) < 0.55 * len(answer):
        log.warning("Repair shrank the answer from %d to %d chars; keeping the original.",
                    len(answer), len(repaired))
        return ""
    return repaired


def better_answer(original, repaired, sources):
    """Pick whichever version is genuinely better; ties go to the repaired text."""
    if not repaired:
        return original, None, None
    before = answer_quality_signals(original, sources)
    after = answer_quality_signals(repaired, sources)
    log.info("REPAIR: score %.3f -> %.3f (flagged %.0f%% -> %.0f%%, cited %d -> %d)",
             before["score"], after["score"],
             before["audit"]["flagged_fraction"] * 100, after["audit"]["flagged_fraction"] * 100,
             before["stats"]["n_cited"], after["stats"]["n_cited"])
    if after["score"] + 1e-9 >= before["score"]:
        return repaired, before, after
    return original, before, after


# ============================================================
# PIPELINE STAGE: CURATOR
# ============================================================

CURATOR_SCHEMA = """{
 "accept": true,
 "learning_value": 0.0,
 "reasons": ["..."],
 "risk_flags": ["..."],
 "category": "short label",
 "difficulty": "easy | medium | hard"
}"""


def curate_example(plan, answer, sources, critique, audit, stats):
    """Final accept/reject decision on the finished example."""
    source_summary = "\n".join(
        "- [%s] %s | tier %d/%s | relevance %.2f | %s"
        % (s["id"], s["domain"], s["tier"], s["kind"], s["relevance"], (s["title"] or "")[:90])
        for s in sources
    )
    user_prompt = (
        "%s\n\nSOURCES USED:\n%s\n\nCITATION PROFILE: %d cited sources across %d domains, "
        "best tier %d, mean tier %.1f, community/testimony sources cited: %d\n\n"
        "CRITIC: verdict=%s accuracy=%.1f citation=%.1f relevance=%.1f quality=%.1f confidence=%.2f\n"
        "CRITIC ISSUES: %s\nCRITIC REQUIRED FIXES: %s\n\n"
        "AUTOMATED GROUNDING: verdict=%s flagged=%.0f%% hard_flags=%d citation_density=%.2f\n\n"
        "FINAL ANSWER:\n<<<ANSWER\n%s\nANSWER>>>\n\n"
        "Decide whether this belongs in a training dataset for a rigorous research "
        "assistant. Return exactly this JSON:\n%s\n"
    ) % (
        _plan_header(plan), source_summary or "(none)",
        stats["n_cited"], stats["n_domains"], stats["best_tier"], stats["mean_tier"], stats["n_tier5"],
        critique["verdict"], critique["accuracy_score"], critique["citation_score"],
        critique["relevance_score"], critique["quality_score"], critique["confidence"],
        "; ".join(critique["issues"][:8]) or "none",
        "; ".join(critique["required_fixes"][:8]) or "none",
        audit["verdict"], audit["flagged_fraction"] * 100, audit["hard_flags"], audit["citation_density"],
        shrink_middle(answer, 9000), CURATOR_SCHEMA,
    )
    raw = call_model(
        CURATOR_SYSTEM, user_prompt, temperature=CURATOR_TEMPERATURE,
        max_tokens=CURATOR_MAX_TOKENS, force_json=True, role="curator",
    )
    data = safe_json_loads(raw, fallback={}, want_keys=("accept", "reasons", "learning_value"))
    decision = {
        "accept": safe_bool(data.get("accept"), False),
        "learning_value": safe_float(data.get("learning_value"), 0.0, 0.0, 10.0),
        "reasons": as_str_list(data.get("reasons"), limit=12),
        "risk_flags": as_str_list(data.get("risk_flags"), limit=10),
        "category": clean_text(data.get("category") or "")[:80],
        "difficulty": clean_text(data.get("difficulty") or "")[:20].lower(),
        "parsed": bool(data),
    }
    if not data:
        decision["reasons"] = ["curator output was unparseable"]
        decision["accept"] = False
    log.info("CURATOR: accept=%s learning_value=%.1f reasons=%s",
             decision["accept"], decision["learning_value"],
             "; ".join(decision["reasons"][:3]) or "-")
    return decision


# ============================================================
# PIPELINE STAGE: CONCLUSION
# ============================================================

def write_conclusion(plan, answer, sources, critique):
    if not WRITE_CONCLUSIONS:
        return ""
    try:
        user_prompt = (
            "RESEARCH QUESTION: %s\n\nSOURCES: %s\n\nFINAL ANSWER (excerpt):\n%s\n\n"
            "CRITIC SUMMARY: %s\n\nWrite the log entry."
        ) % (
            plan["research_question"],
            ", ".join("%s (%s, tier %d)" % (s["id"], s["domain"], s["tier"]) for s in sources[:10]),
            shrink_middle(answer, 5000),
            critique.get("summary") or "-",
        )
        text = clean_model_text(call_model(
            CONCLUSION_SYSTEM, user_prompt, temperature=0.3,
            max_tokens=CONCLUSION_MAX_TOKENS, role="conclusion",
        ))
        return text.strip()[:2000]
    except ModelError as e:
        log.warning("Conclusion step failed: %s", str(e)[:160])
        return ""


# ============================================================
# ACCEPTANCE GATES
# ============================================================

ACCEPTED_AREAS = []          # areas of the most recent accepted examples
STRICT_GROUNDING = _env_bool("AUTOLAB_STRICT_GROUNDING", True)
SAVE_SOURCE_TEXT = _env_bool("AUTOLAB_SAVE_SOURCE_TEXT", False)


def refresh_accepted_areas(window=AREA_SATURATION_WINDOW * 4):
    """Recover the recent accepted-area history from the dataset tail."""
    global ACCEPTED_AREAS
    areas = []
    for line in read_tail_lines(SFT_PATH, window):
        try:
            record = json.loads(line)
            area = str(((record or {}).get("meta") or {}).get("area") or "").strip().lower()
        except Exception:
            continue
        if area:
            areas.append(area)
    ACCEPTED_AREAS = areas[-window:]


def area_is_saturated(area):
    recent = ACCEPTED_AREAS[-AREA_SATURATION_WINDOW:]
    hits = sum(1 for a in recent if a == (area or "").strip().lower())
    return hits >= AREA_SATURATION_LIMIT, hits


def evaluate_gates(plan, answer, sources, critique, decision, audit, stats):
    """Apply every deterministic quality gate. Returns (accept, reasons)."""
    reasons = []

    # --- shape -------------------------------------------------------------
    if len(answer) < MIN_ANSWER_CHARS:
        reasons.append("answer too short: %d < %d chars" % (len(answer), MIN_ANSWER_CHARS))
    if audit["n_cited_sentences"] < MIN_CITED_CLAIMS:
        reasons.append("only %d cited claim sentences (need %d)"
                       % (audit["n_cited_sentences"], MIN_CITED_CLAIMS))

    # --- critic ------------------------------------------------------------
    if not critique.get("parsed", False):
        reasons.append("critic:unparseable")
    if critique["verdict"] == "reject":
        reasons.append("critic:verdict-reject")
    if critique["accuracy_score"] < MIN_ACCURACY_SCORE:
        reasons.append("critic:accuracy %.1f < %.1f" % (critique["accuracy_score"], MIN_ACCURACY_SCORE))
    if critique["quality_score"] < MIN_QUALITY_SCORE:
        reasons.append("critic:quality %.1f < %.1f" % (critique["quality_score"], MIN_QUALITY_SCORE))
    if critique["citation_score"] < MIN_CITATION_SCORE:
        reasons.append("critic:citation %.1f < %.1f" % (critique["citation_score"], MIN_CITATION_SCORE))
    if critique["relevance_score"] < MIN_RELEVANCE_SCORE:
        reasons.append("critic:relevance %.1f < %.1f" % (critique["relevance_score"], MIN_RELEVANCE_SCORE))
    if critique["confidence"] < MIN_CONFIDENCE:
        reasons.append("critic:confidence %.2f < %.2f" % (critique["confidence"], MIN_CONFIDENCE))
    if len(critique["required_fixes"]) > MAX_REQUIRED_FIXES:
        reasons.append("critic:%d required fixes (max %d)"
                       % (len(critique["required_fixes"]), MAX_REQUIRED_FIXES))
    if len(critique["issues"]) > MAX_ISSUES:
        reasons.append("critic:%d issues (max %d)" % (len(critique["issues"]), MAX_ISSUES))
    if critique["unsupported_claims"]:
        reasons.append("critic:%d unsupported claims" % len(critique["unsupported_claims"]))

    # --- source quality ----------------------------------------------------
    if stats["n_cited"] < MIN_CITED_SOURCES:
        reasons.append("sources:only %d cited (need %d)" % (stats["n_cited"], MIN_CITED_SOURCES))
    if stats["n_domains"] < MIN_CITED_DOMAINS:
        reasons.append("sources:only %d distinct domains (need %d)"
                       % (stats["n_domains"], MIN_CITED_DOMAINS))
    if stats["best_tier"] < MIN_CITED_SOURCE_QUALITY:
        message = "sources:best cited tier %d < %d" % (stats["best_tier"], MIN_CITED_SOURCE_QUALITY)
        if ENFORCE_MIN_CITED_SOURCE_QUALITY:
            reasons.append(message)
        else:
            log.warning("%s (not enforced)", message)
    if REQUIRE_TIER5_SOURCE and stats["n_tier5"] < MIN_TIER5_SOURCES:
        reasons.append("sources:no community/mid-tier source cited (need %d)" % MIN_TIER5_SOURCES)

    # --- automated grounding ----------------------------------------------
    if audit["unknown_ids"]:
        reasons.append("grounding:cites non-existent sources %s" % ",".join(audit["unknown_ids"][:5]))
    if audit["flagged_fraction"] > GROUNDING_MAX_FLAGGED_FRACTION:
        reasons.append("grounding:%.0f%% of claims flagged (max %.0f%%)"
                       % (audit["flagged_fraction"] * 100, GROUNDING_MAX_FLAGGED_FRACTION * 100))
    if STRICT_GROUNDING and audit["hard_flags"]:
        reasons.append("grounding:%d hard findings" % audit["hard_flags"])

    # --- curator -----------------------------------------------------------
    if not decision["accept"]:
        reasons.append("curator:reject (%s)" % ("; ".join(decision["reasons"][:2]) or "no reason given"))

    # --- novelty / dataset balance ----------------------------------------
    duplicate = is_duplicate_question(plan["research_question"])
    if duplicate:
        reasons.append("novelty:%s" % duplicate)
    saturated, hits = area_is_saturated(plan.get("area", ""))
    if saturated:
        reasons.append("novelty:area '%s' used %d times in the last %d accepted"
                       % (plan.get("area", "?"), hits, AREA_SATURATION_WINDOW))
    content_hash = sha256_text(normalize_question(plan["research_question"]) + "||" + answer[:2000])
    if content_hash in SEEN_HASHES:
        reasons.append("novelty:identical content already saved")

    return (not reasons), reasons, content_hash


# ============================================================
# DATASET WRITERS
# ============================================================

def _cycle_id(question):
    return "%s-%s" % (datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"), sha256_text(question)[:8])


def _source_meta(sources, cited_ids=()):
    cited = set(cited_ids)
    return [
        {
            "id": s["id"], "url": s["url"], "domain": s["domain"], "title": (s["title"] or "")[:200],
            "tier": s["tier"], "kind": s["kind"], "community": s["community"],
            "relevance": s["relevance"], "chars": s["chars"], "cited": s["id"] in cited,
        }
        for s in sources
    ]


def save_sft_example(plan, answer, sources, critique, decision, audit, stats, cycle_id):
    record = {
        "id": cycle_id,
        "messages": [
            {"role": "system", "content": SFT_SYSTEM_PROMPT},
            {"role": "user", "content": plan["research_question"]},
            {"role": "assistant", "content": rewrite_citations_for_sft(answer, sources)},
        ],
        "meta": {
            "created": now_iso(),
            "pipeline_version": PIPELINE_VERSION,
            "model": MODEL_NAME,
            "area": plan.get("area", ""),
            "learning_goal": plan.get("learning_goal", ""),
            "category": decision.get("category", ""),
            "difficulty": decision.get("difficulty", ""),
            "learning_value": decision.get("learning_value", 0.0),
            "citation_mode": SFT_CITATION_MODE,
            "scores": {
                "accuracy": critique["accuracy_score"], "citation": critique["citation_score"],
                "relevance": critique["relevance_score"], "quality": critique["quality_score"],
                "confidence": critique["confidence"],
            },
            "grounding": {
                "verdict": audit["verdict"], "flagged_fraction": audit["flagged_fraction"],
                "hard_flags": audit["hard_flags"], "citation_density": audit["citation_density"],
                "cited_claims": audit["n_cited_sentences"],
            },
            "citations": {
                "n_cited": stats["n_cited"], "n_domains": stats["n_domains"],
                "best_tier": stats["best_tier"], "mean_tier": stats["mean_tier"],
                "n_tier5": stats["n_tier5"], "domains": stats["domains"],
            },
            "sources": _source_meta(sources, stats["cited_ids"]),
        },
    }
    append_jsonl(SFT_PATH, record)
    return record


def save_preference_pair(plan, chosen, rejected, cycle_id, before, after, sources):
    if not chosen or not rejected or chosen.strip() == rejected.strip():
        return False
    if len(rejected) < MIN_ANSWER_CHARS // 2:
        return False
    append_jsonl(PREF_PATH, {
        "id": cycle_id,
        "prompt": plan["research_question"],
        "system": SFT_SYSTEM_PROMPT,
        "chosen": rewrite_citations_for_sft(chosen, sources),
        "rejected": rewrite_citations_for_sft(rejected, sources),
        "meta": {
            "created": now_iso(),
            "pipeline_version": PIPELINE_VERSION,
            "area": plan.get("area", ""),
            "reason": "critic-guided repair",
            "score_before": (before or {}).get("score"),
            "score_after": (after or {}).get("score"),
            "flagged_before": ((before or {}).get("audit") or {}).get("flagged_fraction"),
            "flagged_after": ((after or {}).get("audit") or {}).get("flagged_fraction"),
        },
    })
    return True


def save_record_file(directory, cycle_id, payload):
    try:
        path = Path(directory) / ("%s.json" % cycle_id)
        atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        return path
    except OSError as e:
        log.warning("Could not write record %s: %s", cycle_id, e)
        return None


def build_trace(plan, answer, sources, critique, decision, audit, stats, extra):
    trace = {
        "id": extra.get("cycle_id"),
        "created": now_iso(),
        "pipeline_version": PIPELINE_VERSION,
        "plan": plan,
        "answer": answer,
        "critique": critique,
        "curator": decision,
        "grounding": {k: v for k, v in audit.items() if k != "flags"},
        "grounding_flags": audit.get("flags", [])[:40],
        "citations": stats,
        "sources": _source_meta(sources, stats.get("cited_ids", [])),
    }
    if SAVE_SOURCE_TEXT:
        trace["source_text"] = {s["id"]: s["text"][:20000] for s in sources}
    else:
        trace["source_excerpts"] = {s["id"]: s["text"][:1200] for s in sources}
    trace.update(extra)
    return trace


def save_source_manifest(cycle_id, plan, sources, diagnostics):
    """Audit trail of what evidence the cycle actually used."""
    save_record_file(SOURCES_DIR, cycle_id, {
        "id": cycle_id,
        "time": now_iso(),
        "question": plan.get("research_question", ""),
        "area": plan.get("area", ""),
        "queries": plan.get("search_queries", []),
        "diagnostics": dict(diagnostics),
        "sources": _source_meta(sources),
    })


def record_attempt(plan, status, reasons, cycle_id, extra=None):
    entry = {
        "id": cycle_id,
        "time": now_iso(),
        "question": plan.get("research_question", ""),
        "area": plan.get("area", ""),
        "status": status,
        "reasons": [str(r)[:160] for r in (reasons or [])][:10],
    }
    if extra:
        entry.update(extra)
    ATTEMPTED_QUESTIONS.append(entry)
    del ATTEMPTED_QUESTIONS[: max(0, len(ATTEMPTED_QUESTIONS) - 1000)]   # bound memory
    try:
        save_attempted_questions(ATTEMPTED_QUESTIONS)
    except OSError as e:
        log.warning("Could not persist attempted questions: %s", e)


def save_conclusion(plan, text, sources, cycle_id, accepted):
    if not text:
        return
    try:
        append_jsonl(CONCLUSIONS_PATH, {
            "id": cycle_id,
            "time": now_iso(),
            "area": plan.get("area", ""),
            "question": plan["research_question"],
            "conclusion": text,
            "accepted": bool(accepted),
            "sources": [{"id": s["id"], "domain": s["domain"], "tier": s["tier"]} for s in sources[:12]],
        })
    except OSError as e:
        log.warning("Could not write conclusion: %s", e)


# ============================================================
# RESEARCH CYCLE
# ============================================================

def _disk_guard():
    free = free_disk_mb(BASE_DIR)
    if free < MIN_FREE_DISK_MB:
        log.error("Only %.0f MB free at %s (minimum %d MB). Stopping to protect the dataset.",
                  free, BASE_DIR, MIN_FREE_DISK_MB)
        STOP_EVENT.set()
        return False
    return True


def run_cycle(cycle_index, forced_topic=None, dry_run=False):
    """One full research cycle. Returns 'accepted' | 'rejected' | 'skipped'."""
    started = time.time()
    check_stop()
    if not _disk_guard():
        raise StopRequested()

    log.info("=" * 78)
    log.info("CYCLE %d  |  dataset: %d SFT / %d preference  |  lifetime accepted %d",
             cycle_index, dataset_count(SFT_PATH), dataset_count(PREF_PATH),
             LIFETIME.get("accepted", 0))

    plan = plan_research(forced_topic)
    if not plan:
        bump_stat("skipped", ["planner:no novel question"])
        return "skipped"
    cycle_id = _cycle_id(plan["research_question"])
    weights = keyword_profile(
        plan["research_question"], plan.get("learning_goal", ""),
        " ".join(plan.get("evidence_requirements", [])),
    )

    sources, diagnostics = collect_sources(plan, weights)
    if len(sources) < MIN_SOURCES_FOR_CYCLE:
        reasons = ["sources:only %d usable (need %d)" % (len(sources), MIN_SOURCES_FOR_CYCLE)]
        log.warning("%s - skipping this question.", reasons[0])
        record_attempt(plan, "skipped", reasons, cycle_id, {"diagnostics": dict(diagnostics)})
        bump_stat("skipped", reasons)
        return "skipped"

    if not dry_run:
        save_source_manifest(cycle_id, plan, sources, diagnostics)

    answer, signals = produce_answer(plan, sources, weights)
    if not answer or len(answer) < MIN_ANSWER_CHARS // 2:
        reasons = ["generation:answer empty or unusably short (%d chars)" % len(answer or "")]
        log.warning(reasons[0])
        record_attempt(plan, "skipped", reasons, cycle_id)
        bump_stat("skipped", reasons)
        return "skipped"

    original_answer = answer
    audit = (signals or {}).get("audit") or grounding_audit(answer, sources)
    critique = critique_answer(plan, answer, sources, weights, audit)

    repair_before = repair_after = None
    rounds = 0
    while rounds < MAX_REPAIR_ROUNDS and not STOP_EVENT.is_set():
        needs_repair = (
            critique["verdict"] != "accept"
            or critique["required_fixes"]
            or critique["unsupported_claims"]
            or audit["verdict"] in ("suspect", "bad")
        )
        if not needs_repair:
            break
        rounds += 1
        log.info("REPAIR round %d/%d", rounds, MAX_REPAIR_ROUNDS)
        candidate = repair_answer(plan, answer, sources, weights, critique, audit)
        chosen, before, after = better_answer(answer, candidate, sources)
        if chosen == answer:
            break
        repair_before = repair_before or before
        repair_after = after
        answer = chosen
        audit = grounding_audit(answer, sources)
        critique = critique_answer(plan, answer, sources, weights, audit)

    stats = citation_stats(answer, sources)
    decision = curate_example(plan, answer, sources, critique, audit, stats)
    accepted, reasons, content_hash = evaluate_gates(
        plan, answer, sources, critique, decision, audit, stats
    )
    elapsed = time.time() - started

    if dry_run:
        log.info("DRY RUN: would have %s (%s)", "ACCEPTED" if accepted else "REJECTED",
                 "; ".join(reasons[:4]) or "all gates passed")
        return "accepted" if accepted else "rejected"

    extra = {
        "cycle_id": cycle_id,
        "cycle_index": cycle_index,
        "elapsed_seconds": round(elapsed, 1),
        "accepted": accepted,
        "reject_reasons": reasons,
        "repair_rounds": rounds,
        "search_diagnostics": dict(diagnostics),
        "candidate_signals": {k: v for k, v in (signals or {}).items() if k in ("score", "chars")},
    }

    if accepted:
        # The dataset line is written first: if that write fails the cycle is an
        # error, and no in-memory index claims an example that is not on disk.
        save_sft_example(plan, answer, sources, critique, decision, audit, stats, cycle_id)
        SEEN_HASHES.add(content_hash)
        try:
            save_seen(SEEN_HASHES)
        except OSError as e:
            log.warning("Could not persist seen hashes: %s", e)
        ACCEPTED_NORMALIZED.add(normalize_question(plan["research_question"]))
        ACCEPTED_TOKENSETS.append((plan["research_question"], token_set(plan["research_question"])))
        del ACCEPTED_TOKENSETS[: max(0, len(ACCEPTED_TOKENSETS) - 5000)]
        ACCEPTED_AREAS.append((plan.get("area") or "").strip().lower())
        del ACCEPTED_AREAS[: max(0, len(ACCEPTED_AREAS) - AREA_SATURATION_WINDOW * 4)]
        if rounds and original_answer != answer:
            save_preference_pair(plan, answer, original_answer, cycle_id,
                                 repair_before, repair_after, sources)
        conclusion = write_conclusion(plan, answer, sources, critique)
        save_conclusion(plan, conclusion, sources, cycle_id, True)
        extra["conclusion"] = conclusion
        save_record_file(RESEARCH_DIR, cycle_id,
                         build_trace(plan, answer, sources, critique, decision, audit, stats, extra))
        record_attempt(plan, "accepted", [], cycle_id)
        bump_stat("accepted")
        log.info("ACCEPTED in %.0fs: %s", elapsed, plan["research_question"][:140])
    else:
        save_record_file(REJECTED_DIR, cycle_id,
                         build_trace(plan, answer, sources, critique, decision, audit, stats, extra))
        record_attempt(plan, "rejected", reasons, cycle_id)
        bump_stat("rejected", reasons)
        log.info("REJECTED in %.0fs (%s): %s", elapsed, "; ".join(reasons[:4]),
                 plan["research_question"][:110])
    return "accepted" if accepted else "rejected"


# ============================================================
# SINGLE-INSTANCE LOCK
# ============================================================

def _pid_alive(pid):
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True      # exists, owned by someone else
    except OSError:
        return True
    return True


class InstanceLock:
    """Cooperative lock so two autolab processes cannot share one dataset.

    A lock left behind by a killed process is detected (same host, dead pid) and
    reclaimed instead of blocking the run forever.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.acquired = False

    def _payload(self):
        return json.dumps({
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "started": now_iso(),
            "argv": " ".join(sys.argv[:6]),
        }, ensure_ascii=False)

    def acquire(self):
        for attempt in (1, 2):
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(self._payload())
                self.acquired = True
                atexit.register(self.release)
                return True
            except FileExistsError:
                holder = {}
                try:
                    holder = json.loads(self.path.read_text(encoding="utf-8", errors="replace"))
                except Exception:
                    holder = {}
                pid = int(safe_float(holder.get("pid"), 0))
                same_host = holder.get("host") == socket.gethostname()
                if attempt == 1 and (not holder or (same_host and not _pid_alive(pid))):
                    log.warning("Removing a stale lock from pid %s (%s).", pid or "?",
                                holder.get("started", "unknown start"))
                    try:
                        self.path.unlink()
                        continue
                    except OSError as e:
                        log.error("Could not remove the stale lock: %s", e)
                        return False
                log.error("Another autolab instance holds %s (pid %s on %s, started %s).",
                          self.path, pid or "?", holder.get("host", "?"), holder.get("started", "?"))
                return False
            except OSError as e:
                log.warning("Could not create the lock file (%s); continuing without it.", e)
                return True
        return False

    def release(self):
        if not self.acquired:
            return
        self.acquired = False
        try:
            if self.path.exists():
                holder = {}
                try:
                    holder = json.loads(self.path.read_text(encoding="utf-8", errors="replace"))
                except Exception:
                    pass
                if not holder or int(safe_float(holder.get("pid"), 0)) == os.getpid():
                    self.path.unlink()
        except OSError:
            pass


# ============================================================
# SIGNALS
# ============================================================

_SIGNAL_COUNT = Counter()


def _handle_signal(signum, _frame):
    _SIGNAL_COUNT[signum] += 1
    name = getattr(signal.Signals(signum), "name", str(signum)) if hasattr(signal, "Signals") else str(signum)
    if _SIGNAL_COUNT[signum] >= 2:
        sys.stderr.write("\n%s again - exiting immediately.\n" % name)
        os._exit(130)
    sys.stderr.write("\n%s received - finishing the current step, then stopping.\n" % name)
    STOP_EVENT.set()


def install_signal_handlers():
    for name in ("SIGINT", "SIGTERM", "SIGHUP", "SIGBREAK"):
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        try:
            signal.signal(sig, _handle_signal)
        except (ValueError, OSError, RuntimeError):
            pass  # not the main thread, or unsupported on this platform


# ============================================================
# PREFLIGHT
# ============================================================

def _probe_writable(directory):
    try:
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=str(directory), prefix=".probe-", delete=True):
            pass
        return True, ""
    except Exception as e:
        return False, "%s: %s" % (type(e).__name__, str(e)[:120])


def preflight(run_search_probe=True, run_model_probe=True):
    """Verify everything the loop depends on. Returns True when the run can start."""
    ok = True
    print("=" * 78)
    print("AUTOLAB %s - preflight" % PIPELINE_VERSION)
    print("=" * 78)
    print("base dir        : %s" % BASE_DIR)
    print("model endpoint  : %s" % LLAMA_BASE_URL)
    print("model name      : %s" % MODEL_NAME)
    print("memory file     : %s" % ("(none)" if MEMORY_PATH is None else "%s%s" % (MEMORY_PATH, "" if MEMORY_PATH.is_file() else "  (missing)")))
    print("search provider : %s" % resolved_search_provider())
    print("python          : %s" % sys.version.split()[0])

    print("\n[1] directories")
    for directory in (DATASET_DIR, RESEARCH_DIR, SOURCES_DIR, REJECTED_DIR, EVAL_DIR, LOG_DIR):
        writable, error = _probe_writable(directory)
        print("    %-14s %s%s" % (directory.name, "ok" if writable else "FAILED", "  " + error if error else ""))
        ok = ok and writable

    free = free_disk_mb(BASE_DIR)
    enough = free >= MIN_FREE_DISK_MB
    print("    free space     %.0f MB (minimum %d MB) %s" % (free, MIN_FREE_DISK_MB, "ok" if enough else "TOO LOW"))
    ok = ok and enough

    print("\n[2] optional dependencies")
    print("    pypdf          %s" % ("ok" if PdfReader else "missing - PDF sources will be skipped"))
    print("    trafilatura    %s" % ("ok" if trafilatura else "missing - falling back to BeautifulSoup"))
    print("    ddgs           %s" % (DDGS_PACKAGE or "missing - only Serper can be used"))
    try:
        importlib.import_module("lxml")
        print("    lxml           ok")
    except Exception:
        print("    lxml           missing - html.parser will be used (slower)")

    print("\n[3] search")
    provider = resolved_search_provider()
    if provider == "ddgs" and DDGS is None and not SERPER_API_KEY:
        print("    FAILED: no search backend available (pip install ddgs, or set SERPER_API_KEY)")
        ok = False
    elif run_search_probe:
        try:
            results = web_search("NASA exoplanet archive confirmed planets", 5)
            print("    probe query returned %d results via %s" % (len(results), provider))
            for r in results[:3]:
                print("      - [%d] %s" % (classify_source(r["url"]), r["url"][:90]))
            if not results:
                print("    WARNING: the search probe returned nothing; the loop cannot collect evidence.")
                ok = False
        except Exception as e:
            print("    FAILED: %s: %s" % (type(e).__name__, str(e)[:160]))
            ok = False
    else:
        print("    skipped")

    print("\n[4] model server")
    if not server_is_up():
        print("    FAILED: no answer from %s" % LLAMA_MODELS_URL)
        print("    start llama-server, e.g.:")
        print("      llama-server -m model.gguf -c 32768 --host 127.0.0.1 --port %s"
              % (urlparse(LLAMA_BASE_URL).port or 8080))
        ok = False
    else:
        print("    /v1/models      ok")
        n_ctx = detect_server_context()
        print("    context window  %s" % (n_ctx or "unknown (set AUTOLAB_CTX)"))
        if n_ctx:
            _MODEL_STATE["n_ctx"] = n_ctx
        if run_model_probe:
            try:
                started = time.time()
                reply = call_model("Reply with the single word: ready.", "ready?",
                                   temperature=0.0, max_tokens=16, retries=1, role="probe")
                print("    completion      ok (%.1fs, %r)" % (time.time() - started, reply[:40]))
            except Exception as e:
                print("    completion      FAILED: %s: %s" % (type(e).__name__, str(e)[:160]))
                ok = False

    print("\n[5] dataset")
    stats = read_dataset_stats()
    print("    sft.jsonl       %d examples" % stats["sft_examples"])
    print("    preferences     %d pairs" % stats["preference_examples"])
    print("    seen hashes     %d" % len(SEEN_HASHES))
    print("    attempted       %d" % len(ATTEMPTED_QUESTIONS))
    print("\nPREFLIGHT %s" % ("PASSED" if ok else "FAILED"))
    print("=" * 78)
    return ok


# ============================================================
# DATASET VERIFICATION AND STATISTICS
# ============================================================

def verify_dataset(fix=False):
    """Check every JSONL record; optionally rewrite the file without the bad lines."""
    problems = repaired = 0
    for path, required in ((SFT_PATH, ("messages",)), (PREF_PATH, ("prompt", "chosen", "rejected")),
                           (CONCLUSIONS_PATH, ("question",))):
        if not path.exists():
            print("%-24s missing (nothing to check)" % path.name)
            continue
        good, bad, seen_ids = [], 0, set()
        duplicates = 0
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except Exception as e:
                    bad += 1
                    print("  %s:%d unparseable (%s)" % (path.name, lineno, str(e)[:80]))
                    continue
                if not isinstance(record, dict) or any(k not in record for k in required):
                    bad += 1
                    print("  %s:%d missing required keys %s" % (path.name, lineno, required))
                    continue
                if path == SFT_PATH:
                    messages = record.get("messages")
                    if (not isinstance(messages, list) or len(messages) < 2
                            or not all(isinstance(m, dict) and m.get("content") for m in messages)):
                        bad += 1
                        print("  %s:%d malformed messages" % (path.name, lineno))
                        continue
                    key = normalize_question(first_user_message(record))
                    if key and key in seen_ids:
                        duplicates += 1
                    seen_ids.add(key)
                good.append(record)
        problems += bad
        print("%-24s %d valid, %d invalid, %d duplicate questions"
              % (path.name, len(good), bad, duplicates))
        if fix and bad:
            backup = path.with_name(path.name + ".backup-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
            shutil.copy2(str(path), str(backup))
            atomic_write_text(path, "".join(
                json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in good))
            repaired += bad
            print("  repaired %s (%d lines dropped, backup: %s)" % (path.name, bad, backup.name))
    if fix and repaired:
        print("\n%d problem(s) found, %d repaired." % (problems, repaired))
        return problems == repaired
    print("\n%d problem(s) found." % problems)
    return problems == 0


def print_stats():
    stats = read_dataset_stats()
    print("=" * 78)
    print("AUTOLAB dataset at %s" % BASE_DIR)
    print("=" * 78)
    print("sft examples      : %d" % stats["sft_examples"])
    print("preference pairs  : %d" % stats["preference_examples"])
    print("attempted (kept)  : %d" % len(ATTEMPTED_QUESTIONS))
    print("seen content hash : %d" % len(SEEN_HASHES))
    print("\nlifetime          : %s" % json.dumps(
        {k: v for k, v in LIFETIME.items() if k != "reasons"}, sort_keys=True))
    cycles = max(1, LIFETIME.get("cycles", 0))
    print("acceptance rate   : %.1f%%" % (100.0 * LIFETIME.get("accepted", 0) / cycles))

    areas, tiers, values = Counter(), Counter(), []
    for line in read_tail_lines(SFT_PATH, 5000):
        try:
            meta = (json.loads(line) or {}).get("meta") or {}
        except Exception:
            continue
        areas[meta.get("area", "?")] += 1
        citations = meta.get("citations") or {}
        if citations.get("best_tier") is not None:
            tiers[int(safe_float(citations.get("best_tier"), 0))] += 1
        values.append(safe_float(meta.get("learning_value"), 0.0))
    if areas:
        print("\ntop areas:")
        for area, n in areas.most_common(12):
            print("   %-48s %d" % (area[:48], n))
    if tiers:
        print("\nbest cited source tier distribution:")
        for tier in sorted(tiers, reverse=True):
            print("   tier %-2d %s %d" % (tier, "#" * min(40, tiers[tier]), tiers[tier]))
    if values:
        print("\nmean curator learning value: %.2f" % (sum(values) / len(values)))
    reasons = LIFETIME.get("reasons") or {}
    if reasons:
        print("\ntop rejection reasons:")
        for reason, n in sorted(reasons.items(), key=lambda kv: -kv[1])[:15]:
            print("   %-58s %d" % (reason[:58], n))
    print("=" * 78)


# ============================================================
# MAIN LOOP
# ============================================================

def write_evaluation_snapshot(cycles, started):
    """Periodic health report: acceptance rate, source mix, top rejection causes.

    This is what tells you whether the gates are too tight, the search backend has
    degraded, or the model has started drifting - without reading the log by hand.
    """
    areas, tiers, kinds = Counter(), Counter(), Counter()
    values, densities, flagged = [], [], []
    for line in read_tail_lines(SFT_PATH, 500):
        try:
            meta = (json.loads(line) or {}).get("meta") or {}
        except Exception:
            continue
        areas[meta.get("area", "?")] += 1
        citations = meta.get("citations") or {}
        grounding = meta.get("grounding") or {}
        tiers[int(safe_float(citations.get("best_tier"), 0))] += 1
        for source in meta.get("sources") or []:
            if source.get("cited"):
                kinds[source.get("kind", "?")] += 1
        values.append(safe_float(meta.get("learning_value"), 0.0))
        densities.append(safe_float(grounding.get("citation_density"), 0.0))
        flagged.append(safe_float(grounding.get("flagged_fraction"), 0.0))

    def _mean(xs):
        return round(sum(xs) / len(xs), 3) if xs else 0.0

    snapshot = {
        "time": now_iso(),
        "pipeline_version": PIPELINE_VERSION,
        "model": MODEL_NAME,
        "run": {
            "cycles": cycles,
            "elapsed": _human_duration(time.time() - started),
            "accepted": RUN_COUNTS["accepted"], "rejected": RUN_COUNTS["rejected"],
            "skipped": RUN_COUNTS["skipped"], "errors": RUN_COUNTS["errors"],
            "acceptance_rate": round(RUN_COUNTS["accepted"] / max(1, cycles), 3),
        },
        "lifetime": {k: v for k, v in LIFETIME.items() if k != "reasons"},
        "dataset": read_dataset_stats(),
        "recent_examples": {
            "areas": dict(areas.most_common(15)),
            "best_tier_histogram": {str(k): v for k, v in sorted(tiers.items(), reverse=True)},
            "cited_source_kinds": dict(kinds.most_common()),
            "mean_learning_value": _mean(values),
            "mean_citation_density": _mean(densities),
            "mean_flagged_fraction": _mean(flagged),
        },
        "top_rejection_reasons": dict(
            sorted((LIFETIME.get("reasons") or {}).items(), key=lambda kv: -kv[1])[:20]
        ),
        "search_backends": {
            "provider": resolved_search_provider(),
            "successes": dict(_BACKENDS.successes),
            "blocks": dict(_BACKENDS.block_count),
            "cooling_down": [b for b in WEB_SEARCH_BACKENDS if not _BACKENDS.available(b)],
        },
    }
    save_record_file(EVAL_DIR, "eval-%s" % datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"), snapshot)
    log.info("EVAL snapshot: accepted %d/%d this run (%.0f%%), mean flagged %.2f, provider %s",
             RUN_COUNTS["accepted"], cycles, 100.0 * RUN_COUNTS["accepted"] / max(1, cycles),
             snapshot["recent_examples"]["mean_flagged_fraction"], snapshot["search_backends"]["provider"])
    return snapshot


def _human_duration(seconds):
    seconds = int(max(0, seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return "%dh%02dm" % (hours, minutes)
    if minutes:
        return "%dm%02ds" % (minutes, secs)
    return "%ds" % secs


def main_loop(max_cycles=0, forced_topic=None, dry_run=False, max_runtime=None, max_examples=None):
    started = time.time()
    max_runtime = MAX_RUNTIME_SECONDS if max_runtime is None else max_runtime
    max_examples = MAX_EXAMPLES if max_examples is None else max_examples
    cycles = failures = skips = 0
    log.info("Run limits: cycles=%s runtime=%s examples=%s",
             max_cycles or "unlimited",
             _human_duration(max_runtime) if max_runtime else "unlimited",
             max_examples or "unlimited")

    while not STOP_EVENT.is_set():
        if max_cycles and cycles >= max_cycles:
            log.info("Reached the cycle limit (%d).", max_cycles)
            break
        elapsed = time.time() - started
        if max_runtime and elapsed >= max_runtime:
            log.info("Reached the runtime limit (%s).", _human_duration(max_runtime))
            break
        if max_examples and dataset_count(SFT_PATH) >= max_examples:
            log.info("Reached the dataset target (%d examples).", max_examples)
            break

        cycles += 1
        topic = forced_topic if cycles == 1 else None
        try:
            outcome = run_cycle(cycles, forced_topic=topic, dry_run=dry_run)
            failures = 0
            if outcome == "skipped":
                skips += 1
                if skips >= 5:
                    # Nothing is being produced: usually a dead search backend or a
                    # saturated question space. Slow down instead of burning the
                    # model on cycles that cannot finish.
                    pause = min(FAILURE_BACKOFF_MAX, 30 * (skips - 4))
                    log.warning("%d cycles in a row produced nothing usable; pausing %ds. "
                                "Check the search provider and the novelty gates.", skips, pause)
                    sleep_interruptible(pause)
            else:
                skips = 0
        except StopRequested:
            log.info("Stop requested; ending the run.")
            break
        except KeyboardInterrupt:
            STOP_EVENT.set()
            break
        except ModelError as e:
            failures += 1
            log.error("Model failure in cycle %d (%d/%d): %s",
                      cycles, failures, MAX_CONSECUTIVE_FAILURES, str(e)[:300])
            bump_stat("errors", ["model:%s" % str(e)[:60]])
            if not STOP_EVENT.is_set():
                wait_for_server(min(SERVER_WAIT_SECONDS, 60 * failures))
        except Exception as e:
            failures += 1
            log.exception("Unexpected failure in cycle %d (%d/%d): %s: %s",
                          cycles, failures, MAX_CONSECUTIVE_FAILURES, type(e).__name__, str(e)[:200])
            bump_stat("errors", ["%s" % type(e).__name__])

        if failures:
            if failures >= MAX_CONSECUTIVE_FAILURES:
                log.error("Aborting: %d consecutive failures.", failures)
                break
            backoff = min(FAILURE_BACKOFF_MAX, 5 * (2 ** min(failures, 6)))
            log.info("Backing off %ss before the next cycle.", backoff)
            sleep_interruptible(backoff)

        if cycles % STATS_EVERY == 0:
            try:
                save_stats()
                write_evaluation_snapshot(cycles, started)
                prune_dir(RESEARCH_DIR, MAX_RECORD_FILES)
                prune_dir(REJECTED_DIR, MAX_RECORD_FILES)
                prune_dir(SOURCES_DIR, MAX_RECORD_FILES)
                prune_dir(EVAL_DIR, 200)
            except Exception as e:
                log.warning("Housekeeping failed: %s", str(e)[:160])
        if not STOP_EVENT.is_set():
            sleep_interruptible(SLEEP_BETWEEN_CYCLES)

    try:
        save_stats()
        if cycles and not dry_run:
            write_evaluation_snapshot(cycles, started)
    except Exception as e:
        log.warning("Could not write the final snapshot: %s", str(e)[:160])
    elapsed = time.time() - started
    log.info("=" * 78)
    log.info("RUN COMPLETE: %d cycles in %s | accepted %d, rejected %d, skipped %d, errors %d",
             cycles, _human_duration(elapsed), RUN_COUNTS["accepted"], RUN_COUNTS["rejected"],
             RUN_COUNTS["skipped"], RUN_COUNTS["errors"])
    log.info("Dataset now: %d SFT examples, %d preference pairs.",
             dataset_count(SFT_PATH), dataset_count(PREF_PATH))
    return 0


# ============================================================
# CLI
# ============================================================

def build_arg_parser():
    parser = argparse.ArgumentParser(
        prog="autolab",
        description="AutoLab %s - autonomous research -> critique -> repair -> curate "
                    "dataset builder." % PIPELINE_VERSION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Environment variables are documented at the top of this file.",
    )
    mode = parser.add_argument_group("modes")
    mode.add_argument("--once", action="store_true", help="run exactly one research cycle")
    mode.add_argument("--cycles", type=int, default=0, metavar="N", help="stop after N cycles")
    mode.add_argument("--check", action="store_true", help="preflight checks only, then exit")
    mode.add_argument("--selftest", action="store_true",
                      help="offline unit tests plus one cycle against a stub server")
    mode.add_argument("--dry-run", action="store_true",
                      help="run the full pipeline but write nothing to the dataset")
    mode.add_argument("--verify", action="store_true", help="validate the existing dataset files")
    mode.add_argument("--fix", action="store_true", help="with --verify: drop invalid lines (keeps a backup)")
    mode.add_argument("--stats", action="store_true", help="print dataset statistics and exit")

    target = parser.add_argument_group("targets")
    target.add_argument("--base-dir", metavar="PATH", help="output root (overrides AUTOLAB_BASE_DIR)")
    target.add_argument("--url", metavar="URL", help="llama-server base URL")
    target.add_argument("--model", metavar="NAME", help="model id sent to the server")
    target.add_argument("--memory", metavar="PATH", help="persona/memory excerpt file")
    target.add_argument("--ctx", type=int, metavar="N", help="context window override")
    target.add_argument("--provider", choices=("auto", "serper", "ddgs"), help="search provider")
    target.add_argument("--topic", metavar="TEXT", help="force the first cycle to research this question")

    limits = parser.add_argument_group("limits")
    limits.add_argument("--max-hours", type=float, metavar="H", help="wall-clock limit for the run")
    limits.add_argument("--max-examples", type=int, metavar="N", help="stop at this dataset size")
    limits.add_argument("--no-lock", action="store_true", help="skip the single-instance lock")

    noise = parser.add_argument_group("output")
    noise.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    noise.add_argument("-q", "--quiet", action="store_true", help="warnings and errors only")
    noise.add_argument("--version", action="version", version="autolab %s" % PIPELINE_VERSION)
    return parser


def apply_cli_overrides(args):
    global MODEL_NAME, MEMORY_PATH, CONTEXT_SIZE, SEARCH_PROVIDER
    if args.base_dir:
        configure_paths(args.base_dir)
    if args.url:
        configure_endpoint(args.url)
    if args.model:
        MODEL_NAME = args.model
    if args.memory:
        MEMORY_PATH = Path(args.memory)
    if args.ctx:
        CONTEXT_SIZE = max(4096, int(args.ctx))
        _MODEL_STATE["n_ctx"] = CONTEXT_SIZE
    if args.provider:
        SEARCH_PROVIDER = args.provider


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    apply_cli_overrides(args)

    if args.selftest:
        setup_logging(console_only=True)
        log.setLevel(logging.DEBUG if args.verbose else logging.WARNING)
        return run_selftest(verbose=args.verbose)

    ensure_dirs()
    setup_logging()
    if args.verbose:
        log.setLevel(logging.DEBUG)
    elif args.quiet:
        log.setLevel(logging.WARNING)
    install_signal_handlers()
    load_state()
    refresh_accepted_areas()

    if args.stats:
        print_stats()
        return 0
    if args.verify:
        return 0 if verify_dataset(fix=args.fix) else 1
    if args.check:
        return 0 if preflight() else 3

    lock = InstanceLock(LOCK_PATH)
    if not args.no_lock and not lock.acquire():
        return 4

    try:
        load_memory()
        log.info("AutoLab %s starting | base=%s | endpoint=%s | model=%s",
                 PIPELINE_VERSION, BASE_DIR, LLAMA_BASE_URL, MODEL_NAME)
        if not wait_for_server():
            log.error("llama-server did not become available at %s within %ds.",
                      LLAMA_BASE_URL, SERVER_WAIT_SECONDS)
            return 3
        init_context_window()
        provider = resolved_search_provider()
        if provider == "ddgs" and DDGS is None:
            log.error("No search backend is installed (pip install ddgs) and no SERPER_API_KEY is set.")
            return 3
        max_runtime = int(args.max_hours * 3600) if args.max_hours else None
        return main_loop(
            max_cycles=1 if args.once else max(0, args.cycles),
            forced_topic=args.topic,
            dry_run=args.dry_run,
            max_runtime=max_runtime,
            max_examples=args.max_examples,
        )
    except KeyboardInterrupt:
        log.info("Interrupted.")
        return 0
    finally:
        lock.release()


# ============================================================
# SELF-TEST
# ============================================================
# `--selftest` runs entirely offline: deterministic tests of the pure functions,
# then two complete research cycles against an in-process stub of llama-server
# with a stubbed search/fetch layer. It is the fastest way to confirm that a
# change did not break the pipeline without burning GPU time or network quota.

_STUB_STATE = {"mode": "good"}

_STUB_SOURCES = {
    "https://www.nasa.gov/mission/kepler/overview": {
        "title": "Kepler Mission Overview",
        "text": (
            "The Kepler space telescope launched on March 7, 2009 and monitored about 150,000 "
            "main-sequence stars in a fixed field of view in the constellation Cygnus. The "
            "photometer detected planetary transits by measuring brightness dips of roughly 84 "
            "parts per million for an Earth-size planet crossing a Sun-like star. Mission "
            "operations ended on October 30, 2018 after the spacecraft exhausted its hydrazine "
            "fuel. NASA has confirmed 2662 exoplanets from Kepler data, and thousands of "
            "additional candidate signals remain under review by the science team. The mission "
            "operated in two phases, the original Kepler survey and the extended K2 campaign "
            "that used solar radiation pressure for pointing stability after two reaction wheels "
            "failed. Data products including light curves and target pixel files are archived at "
            "the Mikulski Archive for Space Telescopes and remain publicly available."
        ) * 2,
    },
    "https://arxiv.org/abs/1510.01234": {
        "title": "Occurrence rates of small planets from the Kepler sample",
        "text": (
            "We report an occurrence rate analysis of small planets derived from the Kepler "
            "sample. Using a vetted sample of stars and an injection recovery pipeline, we find "
            "that approximately 24 percent of Sun-like stars host an Earth-size planet within "
            "the habitable zone, with substantial systematic uncertainty arising from reliability "
            "corrections. The false positive rate for candidate signals remains a dominant source "
            "of error, and different vetting choices shift the inferred occurrence rate by a "
            "factor of two. We caution that the habitable zone boundaries themselves are model "
            "dependent, so the occurrence rate estimate should not be read as a measurement of "
            "habitability. Independent reanalyses of the same catalog report values between 10 "
            "and 50 percent depending on the completeness treatment."
        ) * 2,
    },
    "https://old.reddit.com/r/astronomy/comments/abc123/kepler_light_curves": {
        "title": "Working with Kepler light curves - r/astronomy",
        "text": (
            "Amateur observers on r/astronomy reported that the archived Kepler light curves are "
            "harder to process than expected. Several users described difficulty removing "
            "systematics from the long cadence data, and one commenter said a full detrending run "
            "took three days on a laptop. Others replied that the community pipelines handle the "
            "systematics better than hand-written scripts, and a few posters disputed the claimed "
            "occurrence rate numbers, saying the completeness corrections are poorly explained "
            "outside the original papers. The thread is testimony about processing experience "
            "rather than a peer-reviewed result."
        ) * 2,
    },
    "https://www.bbc.co.uk/news/science-kepler-retirement": {
        "title": "Kepler telescope retires after nine years",
        "text": (
            "The Kepler space telescope has been retired after nine years of operations, NASA "
            "announced. The observatory ran out of fuel in October 2018 and was placed in a safe "
            "orbit trailing the Earth. Astronomers said the mission transformed the study of "
            "planets around other stars and left a data archive that researchers expect to mine "
            "for another decade. Scientists interviewed for this article emphasised that many of "
            "the candidate detections still require follow-up observations before they can be "
            "counted as confirmed planets."
        ) * 2,
    },
}

_STUB_GOOD_ANSWER = (
    "Kepler's confirmed planet count and the habitable-zone occurrence rate come from two "
    "different kinds of evidence, and only the first is a settled number. NASA reports 2662 "
    "confirmed exoplanets from Kepler data, with thousands of additional candidate signals still "
    "under review by the science team [S1]. Those detections were made by a photometer that "
    "measured brightness dips of roughly 84 parts per million for an Earth-size planet crossing a "
    "Sun-like star, across about 150,000 main-sequence stars in Cygnus [S1]. The spacecraft "
    "exhausted its hydrazine fuel and operations ended on October 30, 2018, which is why the "
    "catalogue is now static and further confirmations come from reanalysis rather than new "
    "photometry [S1]. Press coverage of the retirement stressed that many candidate detections "
    "still require follow-up observations before they can be counted as confirmed planets, so "
    "the confirmed total and the candidate total should never be merged [S4].\n\n"
    "The occurrence rate is far less settled. An occurrence rate analysis of the small planet "
    "sample finds that approximately 24 percent of Sun-like stars host an Earth-size planet "
    "within the habitable zone, but the authors attach substantial systematic uncertainty from "
    "reliability corrections [S2]. Independent reanalyses of the same catalog report values "
    "between 10 and 50 percent depending on the completeness treatment, and different vetting "
    "choices shift the inferred occurrence rate by a factor of two [S2]. Community discussion "
    "adds a caveat about reproducibility rather than about physics: several users described "
    "difficulty removing systematics from the long cadence data, and a few posters disputed the "
    "claimed occurrence rate numbers because the completeness corrections are poorly explained "
    "outside the original papers [S3]. That is testimony about processing experience, not "
    "independent evidence that the published rate is wrong [S3].\n\n"
    "What the supplied sources do not establish is a single occurrence rate estimate confirmed by "
    "a second independent sample; reliability corrections and the false positive rate remain the "
    "dominant source of error, and the habitable zone boundaries are themselves model dependent "
    "[S2]."
)

_STUB_BAD_ANSWER = (
    "The Kepler mission definitively proved that 91 percent of stars host habitable worlds, a "
    "result confirmed by every subsequent survey [S9]. Researchers stated that \"the habitable "
    "zone question is now completely closed and requires no further study whatsoever\" [S1]. "
    "The telescope confirmed 41337 planets during its operations, which ended in 2031 after a "
    "fuel leak [S1]. Reddit users have proven that the completeness corrections are fraudulent "
    "and the published papers should be retracted immediately [S3]. There is no remaining "
    "uncertainty about any of these figures, and the consensus among all astronomers is "
    "unanimous on every point raised here [S2]. The occurrence rate of 91 percent is the single "
    "most replicated finding in the history of exoplanet science [S9]."
)

_STUB_REPAIRED_BAD = (
    "The supplied sources do not support a 91 percent figure of any kind. NASA reports 2662 "
    "confirmed exoplanets from Kepler data, with thousands of additional candidate signals still "
    "under review [S1]. Mission operations ended on October 30, 2018 after the spacecraft "
    "exhausted its hydrazine fuel [S1]. An occurrence rate analysis of the small planet sample "
    "finds that approximately 24 percent of Sun-like stars host an Earth-size planet within the "
    "habitable zone, with substantial systematic uncertainty arising from reliability corrections "
    "[S2]. Several users described difficulty removing systematics from the long cadence data, "
    "which is testimony about processing experience rather than a peer-reviewed result [S3]. "
    "The evidence here does not settle the occurrence rate: independent reanalyses of the same "
    "catalog report values between 10 and 50 percent depending on the completeness treatment [S2]."
)


def _stub_plan(mode):
    if mode == "good":
        question = ("How many exoplanets did the Kepler mission confirm by the end of operations, "
                    "and what habitable-zone occurrence rate do Kepler analyses actually support?")
    else:
        question = ("What did the Kepler extended K2 campaign change about pointing stability "
                    "after the reaction wheel failures of 2013?")
    return json.dumps({
        "area": "exoplanets and habitability",
        "research_question": question,
        "search_queries": [
            "Kepler confirmed exoplanet count NASA",
            "Kepler habitable zone occurrence rate uncertainty",
            "Kepler light curve processing reddit discussion",
        ],
        "learning_goal": "Separate a catalogue count from a modelled occurrence rate.",
        "evidence_requirements": ["primary NASA figure", "peer-reviewed occurrence analysis"],
        "counterargument_target": "that the occurrence rate is a measured quantity",
        "knowledge_gap": "how large the systematic uncertainty is",
        "risk_notes": ["community posts are testimony only"],
    })


def _stub_critique(mode):
    if mode == "good":
        return json.dumps({
            "accuracy_score": 9.2, "citation_score": 9.0, "relevance_score": 9.4,
            "quality_score": 9.1, "confidence": 0.9, "verdict": "accept",
            "issues": [], "unsupported_claims": [], "citation_errors": [],
            "missing_counter_evidence": [], "required_fixes": [],
            "strongest_source": "S1", "summary": "Well grounded and appropriately hedged.",
        })
    return json.dumps({
        "accuracy_score": 1.5, "citation_score": 1.0, "relevance_score": 4.0,
        "quality_score": 2.0, "confidence": 0.2, "verdict": "reject",
        "issues": ["fabricated statistics", "cites a source that does not exist",
                   "false certainty", "quotation is invented", "treats forum posts as proof"],
        "unsupported_claims": ["91 percent of stars host habitable worlds", "41337 planets"],
        "citation_errors": ["S9 is not a supplied source"],
        "missing_counter_evidence": ["systematic uncertainty in occurrence rates"],
        "required_fixes": ["remove fabricated numbers", "remove the invented quotation",
                           "remove S9", "stop treating testimony as proof"],
        "strongest_source": "S1", "summary": "Fabricated throughout.",
    })


def _stub_curation(mode):
    if mode == "good":
        return json.dumps({
            "accept": True, "learning_value": 8.5,
            "reasons": ["specific question", "evidence separates count from model estimate"],
            "risk_flags": [], "category": "exoplanet science", "difficulty": "medium",
        })
    return json.dumps({
        "accept": False, "learning_value": 0.0,
        "reasons": ["fabricated content", "citations do not resolve"],
        "risk_flags": ["hallucination"], "category": "exoplanet science", "difficulty": "hard",
    })


def _stub_reply(system, user):
    mode = _STUB_STATE["mode"]
    system = system or ""
    if "research planner" in system:
        return _stub_plan(mode)
    if "research-answer generator" in system:
        return _STUB_GOOD_ANSWER if mode == "good" else _STUB_BAD_ANSWER
    if "fact-checking critic" in system:
        return _stub_critique(mode)
    if "answer-repair engineer" in system:
        return _STUB_REPAIRED_BAD
    if "dataset curator" in system:
        return _stub_curation(mode)
    if "research log keeper" in system:
        return "Kepler's confirmed count is a catalogue fact; the habitable-zone occurrence rate is a model-dependent estimate with large systematic uncertainty."
    return "ready"


def _start_stub_server():
    """Start an in-process OpenAI-compatible stub and return (server, base_url)."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "AutoLabStub/1.0"

        def log_message(self, *_args):
            pass

        def _send(self, code, payload):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.startswith("/v1/models"):
                self._send(200, {"object": "list", "data": [{"id": MODEL_NAME, "object": "model"}]})
            elif self.path.startswith("/props"):
                self._send(200, {"default_generation_settings": {"n_ctx": 32768}})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            try:
                payload = json.loads(raw.decode("utf-8", "replace"))
            except Exception:
                self._send(400, {"error": {"message": "bad json"}})
                return
            messages = payload.get("messages") or []
            system = next((m.get("content", "") for m in messages if m.get("role") == "system"), "")
            user = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")
            content = _stub_reply(system, user)
            self._send(200, {
                "id": "stub", "object": "chat.completion", "model": payload.get("model", "stub"),
                "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": estimate_tokens(system + user),
                          "completion_tokens": estimate_tokens(content)},
            })

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
    thread.start()
    return server, "http://127.0.0.1:%d" % server.server_address[1]


def _stub_web_search(query, max_results=RESULTS_PER_QUERY):
    results = []
    for url, data in _STUB_SOURCES.items():
        results.append({"title": data["title"], "url": url,
                        "snippet": data["text"][:200], "search_backend": "stub"})
    return results[:max_results]


def _stub_fetch_page_ex(url):
    data = _STUB_SOURCES.get(normalize_url(url)) or _STUB_SOURCES.get(url)
    if not data:
        return _fetch_failure(url, "stub: unknown url")
    return {"ok": True, "text": data["text"], "final_url": url,
            "content_type": "html", "bytes": len(data["text"]), "error": ""}


class _SelfTest:
    def __init__(self, verbose=False):
        self.passed = 0
        self.failures = []
        self.verbose = verbose

    def check(self, name, condition, detail=""):
        if condition:
            self.passed += 1
            if self.verbose:
                print("    ok   %s" % name)
        else:
            self.failures.append("%s%s" % (name, (" - " + str(detail)[:200]) if detail else ""))
            print("    FAIL %s%s" % (name, (" - " + str(detail)[:200]) if detail else ""))

    def equal(self, name, got, expected):
        self.check(name, got == expected, "got %r, expected %r" % (got, expected))


def _unit_tests(t):
    print("\n[1] pure functions")
    # --- URL handling ------------------------------------------------------
    t.equal("normalize_url strips tracking + fragment",
            normalize_url("HTTPS://Example.COM:443/a/b/?utm_source=x&id=7#frag"),
            "https://example.com/a/b?id=7")
    t.equal("normalize_url keeps non-default port",
            normalize_url("http://example.com:8080/x/"), "http://example.com:8080/x")
    t.check("is_public_url rejects localhost", not is_public_url("http://localhost/x"))
    t.check("is_public_url rejects loopback ip", not is_public_url("http://127.0.0.1:8080/x"))
    t.check("is_public_url rejects private range", not is_public_url("http://10.1.2.3/x"))
    t.check("is_public_url rejects link-local", not is_public_url("http://169.254.169.254/latest/meta-data"))
    t.check("is_public_url rejects ipv6 loopback", not is_public_url("http://[::1]/x"))
    t.check("is_public_url rejects file scheme", not is_public_url("file:///etc/passwd"))
    t.check("is_public_url rejects intranet suffix", not is_public_url("http://host.internal/x"))
    t.check("is_public_url accepts a normal host", is_public_url("https://www.nasa.gov/x"))
    t.equal("base_domain multi-part suffix", base_domain("www.bbc.co.uk"), "bbc.co.uk")
    t.equal("base_domain subdomain", base_domain("news.mit.edu"), "mit.edu")

    # --- source tiering ----------------------------------------------------
    t.equal("tier nasa.gov", classify_source("https://www.nasa.gov/a"), 10)
    t.equal("tier arxiv", classify_source("https://arxiv.org/abs/1"), 9)
    t.equal("tier unknown .edu", classify_source("https://cs.someuni.edu/paper"), 8)
    t.equal("tier reddit", classify_source("https://old.reddit.com/r/x/1"), 5)
    t.equal("tier medium", classify_source("https://medium.com/@a/b"), 3)
    t.equal("tier x.com", classify_source("https://x.com/a/status/1"), 1)
    t.check("reddit is community", is_community_source("https://www.reddit.com/r/x"))
    t.check("nasa is not community", not is_community_source("https://www.nasa.gov/x"))
    t.check("reddit satisfies the tier-5 requirement",
            counts_as_tier5(source_profile("https://www.reddit.com/r/x")))
    t.check("nasa does not satisfy the tier-5 requirement",
            not counts_as_tier5(source_profile("https://www.nasa.gov/x")))

    # --- JSON recovery -----------------------------------------------------
    t.equal("json from code fence", safe_json_loads('```json\n{"a": 1}\n```').get("a"), 1)
    t.equal("json with prose around it",
            safe_json_loads('Sure! Here it is:\n{"a": 2}\nHope that helps.').get("a"), 2)
    t.equal("json picks the object with the wanted keys",
            safe_json_loads('{"x":1}\n{"accuracy_score":9}', want_keys=("accuracy_score",)
                            ).get("accuracy_score"), 9)
    t.equal("json trailing comma", safe_json_loads('{"a": 1,}').get("a"), 1)
    t.equal("python literals", safe_json_loads('{"a": True, "b": None}').get("a"), True)
    t.equal("truncated json recovered",
            safe_json_loads('{"a": 1, "b": "unterminated').get("a"), 1)
    t.equal("bad input returns the fallback", safe_json_loads("no json here", fallback={"f": 1}), {"f": 1})
    t.equal("think block stripped",
            clean_model_text("<think>secret reasoning</think>Answer here"), "Answer here")
    t.equal("unclosed think block stripped",
            clean_model_text("Answer<think>never closed"), "Answer")
    t.equal("special tokens stripped",
            clean_model_text("Hi<|im_end|><end_of_turn>"), "Hi")

    # --- citations ---------------------------------------------------------
    t.equal("grouped citation split", normalize_citation_markers("Claim [S1, S2]."), "Claim [S1][S2].")
    t.equal("numeric-only group split", normalize_citation_markers("Claim [S1,2]."), "Claim [S1][S2].")
    t.equal("spaced marker normalised", normalize_citation_markers("Claim [ s3 ]."), "Claim [S3].")
    t.equal("duplicate markers collapsed", normalize_citation_markers("Claim [S2][S2]."), "Claim [S2].")
    t.equal("citation ids extracted", citation_ids("a [S3] b [S1] c [S3]"), ["S3", "S1"])
    t.equal("markers stripped", strip_citation_markers("Fact [S1][S2]."), "Fact.")

    # --- sentences ---------------------------------------------------------
    sentences = split_sentences("Dr. Smith found 3.5 kg of material. He reported it in 2011. Yes.")
    t.equal("abbreviations and decimals do not split sentences", len(sentences), 3)

    # --- text utilities ----------------------------------------------------
    t.check("shrink_middle respects the budget", len(shrink_middle("x" * 5000, 1000)) <= 1000)
    t.check("shrink_middle keeps both ends",
            shrink_middle("START" + "x" * 5000 + "END", 1000).startswith("START"))
    excerpt = relevant_excerpt(
        "intro text here. " + ("filler paragraph. " * 200) + "\n\nthe kepler occurrence rate is 24 percent.\n\n"
        + ("more filler. " * 200), keyword_profile("kepler occurrence rate"), 1200)
    t.check("relevant_excerpt finds the matching paragraph", "kepler occurrence rate" in excerpt.lower())
    t.check("block page detected", looks_like_block_page("Please enable JavaScript and cookies to continue"))
    t.check("normal text is not a block page",
            not looks_like_block_page("The mission launched in 2009 and returned data for years. " * 20))
    t.equal("text_problem flags short text", text_problem("tiny"), "too short")
    t.equal("clean text keeps normal characters", clean_text("a\x00b\ufeffc"), "abc")

    # --- planner helpers ---------------------------------------------------
    t.check("vague question rejected", not _looks_specific("Tell me about volcanoes"))
    t.check("specific question accepted",
            _looks_specific("What did the 2004 USS Nimitz FLIR1 video actually record according to "
                            "the Navy's released analysis?"))
    plan = _normalize_plan({"research_question": "What did the AATIP program fund between 2008 and 2012 "
                                                 "according to released DoD contracts?",
                            "search_queries": ["AATIP contracts DoD"]}, "intelligence history")
    t.check("plan normalisation adds a community query",
            plan is not None and any(any(m in q.lower() for m in COMMUNITY_QUERY_MARKERS)
                                     for q in plan["search_queries"]))
    t.check("plan normalisation rejects vagueness",
            _normalize_plan({"research_question": "What is space?"}, "space science") is None)
    t.check("clean_query removes the site: operator", "site:" not in clean_query("site:arxiv.org kepler"))

    # --- ranking -----------------------------------------------------------
    weights = keyword_profile("kepler occurrence rate habitable zone")
    candidates = [
        {"url": "https://a.example.com/1", "title": "kepler occurrence rate", "snippet": "habitable zone",
         **source_profile("https://a.example.com/1")},
        {"url": "https://a.example.com/2", "title": "kepler", "snippet": "zone",
         **source_profile("https://a.example.com/2")},
        {"url": "https://a.example.com/3", "title": "kepler", "snippet": "rate",
         **source_profile("https://a.example.com/3")},
        {"url": "https://a.example.com/4", "title": "kepler", "snippet": "rate",
         **source_profile("https://a.example.com/4")},
        {"url": "https://b.example.org/1", "title": "kepler occurrence", "snippet": "rate",
         **source_profile("https://b.example.org/1")},
    ]
    ranked = rank_candidates(candidates, weights)
    per_domain = Counter(c["domain"] for c in ranked)
    t.check("per-domain cap enforced", per_domain["example.com"] <= MAX_SOURCES_PER_DOMAIN,
            dict(per_domain))


def _grounding_tests(t):
    print("\n[2] grounding audit")
    sources = []
    for index, (url, data) in enumerate(_STUB_SOURCES.items(), 1):
        profile = source_profile(url)
        sources.append({"id": "S%d" % index, "url": url, "title": data["title"],
                        "text": data["text"], "chars": len(data["text"]),
                        "relevance": 0.5, "snippet": "", "query": "", **profile})

    good = grounding_audit(_STUB_GOOD_ANSWER, sources)
    t.equal("good answer has no unknown source ids", good["unknown_ids"], [])
    t.equal("good answer has no hard findings", good["hard_flags"], 0)
    t.check("good answer stays under the flag threshold",
            good["flagged_fraction"] <= GROUNDING_MAX_FLAGGED_FRACTION,
            "%.2f: %s" % (good["flagged_fraction"], [f["kind"] for f in good["flags"]]))
    t.equal("good answer audits clean", good["verdict"], "clean")
    t.check("good answer cites enough claims", good["n_cited_sentences"] >= MIN_CITED_CLAIMS,
            good["n_cited_sentences"])

    bad = grounding_audit(_STUB_BAD_ANSWER, sources)
    t.equal("fabricated source id detected", bad["unknown_ids"], ["S9"])
    kinds = {f["kind"] for f in bad["flags"]}
    t.check("invented numbers detected", "number_absent_everywhere" in kinds, kinds)
    t.check("invented quotation detected", "quote_not_verbatim" in kinds, kinds)
    t.equal("fabricated answer audits bad", bad["verdict"], "bad")

    stats = citation_stats(_STUB_GOOD_ANSWER, sources)
    t.check("citation stats count distinct sources", stats["n_cited"] >= 3, stats["n_cited"])
    t.check("citation stats count distinct domains", stats["n_domains"] >= MIN_CITED_DOMAINS,
            stats["domains"])
    t.check("citation stats find a high-tier source", stats["best_tier"] >= MIN_CITED_SOURCE_QUALITY,
            stats["best_tier"])
    t.check("citation stats find community testimony", stats["n_tier5"] >= MIN_TIER5_SOURCES,
            stats["tier5_domains"])

    named = rewrite_citations_for_sft(_STUB_GOOD_ANSWER, sources, "named")
    t.check("named citation mode writes domains", "[nasa.gov]" in named, named[:200])
    t.check("named citation mode removes [S ids", "[S1]" not in named)
    stripped = rewrite_citations_for_sft(_STUB_GOOD_ANSWER, sources, "strip")
    t.check("strip citation mode removes markers", "[S" not in stripped)
    t.check("strip citation mode keeps the prose", "2662" in stripped)


def _io_tests(t, tmp):
    print("\n[3] state and file handling")
    configure_paths(tmp)
    ensure_dirs()

    atomic_write_text(DATASET_DIR / "probe.json", json.dumps({"a": 1}))
    t.equal("atomic write round-trips",
            json.loads((DATASET_DIR / "probe.json").read_text(encoding="utf-8"))["a"], 1)

    path = DATASET_DIR / "lines.jsonl"
    for i in range(5):
        append_jsonl(path, {"i": i})
    t.equal("append_jsonl line count", dataset_count(path), 5)
    t.equal("read_tail_lines returns the last records",
            [json.loads(line)["i"] for line in read_tail_lines(path, 2)], [3, 4])

    with path.open("a", encoding="utf-8") as f:     # simulate a killed process
        f.write('{"i": 99, "partial"')
    append_jsonl(path, {"i": 6})
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    t.equal("partial line is isolated, not merged", json.loads(lines[-1])["i"], 6)

    corrupt = DATASET_DIR / "corrupt.json"
    corrupt.write_text("{not json", encoding="utf-8")
    t.equal("corrupt state file falls back to the default",
            load_json_file(corrupt, {"default": True}, dict), {"default": True})
    t.check("corrupt state file is quarantined", not corrupt.exists())

    t.check("free_disk_mb returns a number", free_disk_mb(tmp) > 0)
    lock = InstanceLock(Path(tmp) / "test.lock")
    t.check("lock acquires", lock.acquire())
    second = InstanceLock(Path(tmp) / "test.lock")
    t.check("second lock is refused", not second.acquire())
    lock.release()
    t.check("lock releases", second.acquire())
    second.release()


def _pipeline_test(t, tmp):
    print("\n[4] end-to-end cycles against a stub server")
    server, url = _start_stub_server()
    saved = {"web_search": globals()["web_search"], "fetch_page_ex": globals()["fetch_page_ex"]}
    globals()["web_search"] = _stub_web_search
    globals()["fetch_page_ex"] = _stub_fetch_page_ex
    try:
        configure_endpoint(url)
        configure_paths(tmp)
        ensure_dirs()
        load_state()
        refresh_accepted_areas()
        t.check("stub server answers /v1/models", server_is_up())
        init_context_window()
        t.equal("context window read from /props", _MODEL_STATE["n_ctx"], 32768)

        _STUB_STATE["mode"] = "good"
        outcome = run_cycle(1)
        t.equal("a well-grounded cycle is accepted", outcome, "accepted")
        t.equal("one sft example written", dataset_count(SFT_PATH), 1)
        record = json.loads(read_tail_lines(SFT_PATH, 1)[0])
        t.equal("sft record has three messages", len(record["messages"]), 3)
        t.equal("sft user message is the research question",
                record["messages"][1]["content"], ATTEMPTED_QUESTIONS[-1]["question"])
        t.check("sft record carries source metadata", len(record["meta"]["sources"]) >= 3)
        t.check("sft record carries grounding metadata",
                record["meta"]["grounding"]["verdict"] == "clean")
        t.check("a research trace file was written", any(Path(RESEARCH_DIR).glob("*.json")))
        t.check("a conclusion was logged", dataset_count(CONCLUSIONS_PATH) == 1)

        _STUB_STATE["mode"] = "bad"
        outcome = run_cycle(2)
        t.equal("a fabricated cycle is rejected", outcome, "rejected")
        t.equal("no extra sft example written", dataset_count(SFT_PATH), 1)
        t.check("a rejection trace file was written", any(Path(REJECTED_DIR).glob("*.json")))
        rejected = json.loads(sorted(Path(REJECTED_DIR).glob("*.json"))[-1].read_text(encoding="utf-8"))
        t.check("rejection reasons were recorded", bool(rejected.get("reject_reasons")),
                rejected.get("reject_reasons"))

        _STUB_STATE["mode"] = "good"
        t.check("the accepted question is now a known duplicate",
                bool(is_duplicate_question(record["messages"][1]["content"])))
        save_stats()
        t.check("stats file written", STATS_PATH.exists())
        t.check("lifetime counters updated", LIFETIME["accepted"] >= 1 and LIFETIME["rejected"] >= 1)
    finally:
        globals().update(saved)
        _STUB_STATE["mode"] = "good"
        try:
            server.shutdown()
            server.server_close()
        except Exception:
            pass


def run_selftest(verbose=False):
    print("=" * 78)
    print("AUTOLAB %s - self-test (offline)" % PIPELINE_VERSION)
    print("=" * 78)
    t = _SelfTest(verbose=verbose)
    started = time.time()
    base_state = (BASE_DIR, LLAMA_BASE_URL)
    tmp = tempfile.mkdtemp(prefix="autolab-selftest-")
    try:
        _unit_tests(t)
        _grounding_tests(t)
        _io_tests(t, tmp)
        _pipeline_test(t, tmp)
    except Exception as e:
        t.failures.append("self-test crashed: %s: %s" % (type(e).__name__, e))
        import traceback
        traceback.print_exc()
    finally:
        configure_paths(base_state[0])
        configure_endpoint(base_state[1])
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "=" * 78)
    print("%d checks passed, %d failed, in %.1fs" % (t.passed, len(t.failures), time.time() - started))
    for failure in t.failures:
        print("  FAILED: %s" % failure)
    print("=" * 78)
    return 0 if not t.failures else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        sys.stderr.write("\nInterrupted.\n")
        raise SystemExit(130)
