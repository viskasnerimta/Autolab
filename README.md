# AutoLab

An autonomous **research → critique → repair → curate** loop that builds a supervised fine-tuning (SFT) and preference dataset from public web evidence, using a **local [llama.cpp](https://github.com/ggml-org/llama.cpp) server** as the model. It is a single Python file with no framework dependencies.

> **Status: experimental.** The offline self-test (96 checks, no model required) passes, but the quality of the generated data depends heavily on your model, your search provider and your thresholds. Read the [Limitations](#limitations) section before training on anything it produces.

## How it works

Each cycle runs these stages:

```
plan a question  ->  web search  ->  fetch + rank evidence  ->  write a cited answer
   ->  automated grounding audit  ->  hostile critic  ->  repair  ->  re-audit
   ->  curator  ->  quality gates  ->  sft.jsonl / preferences.jsonl / research_conclusions.jsonl
```

- **Planner** picks a concrete research question (a specific event, document, paper, dataset, failure mode...) and generates search queries, avoiding questions it has already attempted.
- **Evidence collection** searches (Serper or DuckDuckGo-style scraping via `ddgs`), fetches HTML and PDF pages in parallel, scores sources by domain quality and relevance, and caps how many come from one domain.
- **Answer generation** writes an answer that must cite sources as `[S1]`, `[S2]`, ...
- **Grounding audit** is an automated check that cited source ids exist, that numbers in the answer appear in the sources, and that quoted text is verbatim. It is a heuristic filter, not proof of correctness.
- **Critic / repair / curator** stages review the answer against evidence excerpts, repair it once (configurable), and decide whether it passes the quality gates.
- Rejected cycles are kept, with reasons, in `rejected/` so you can see why.

## Requirements

- Python 3.8+
- A running `llama-server` from llama.cpp (needs the OpenAI-compatible `/v1/chat/completions` and `/v1/models` endpoints; `/props` is used to read the context size)
- Internet access for search and page fetching

```bash
pip install -r requirements.txt
```

Only `requests` and `beautifulsoup4` are strictly required. `pypdf` (PDF sources), `trafilatura` (better page extraction), `lxml` (faster parsing) and `ddgs` (free search backends) are optional and degrade gracefully.

## Quick start

**1. Start your model server** (adjust the model path and flags for your hardware):

```bash
llama-server \
  -m /path/to/model.gguf \
  --host 127.0.0.1 --port 8080 \
  -c 32768 -np 1 \
  -ngl 99 -fa on \
  --jinja
```

- `-c` is the context window. Each cycle packs web evidence into prompts, so **32k or more is recommended**. AutoLab reads the real value from the server and budgets its prompts to fit.
- `-np 1` gives the full context to a single request (AutoLab sends one model request at a time).
- To fit a longer context on a limited GPU, quantize the KV cache (`-ctk q8_0 -ctv q8_0`, or `q4_0` for more savings at some quality cost), lower `-ngl` to move layers to the CPU, or add `--no-kv-offload` to keep the KV cache in system RAM (slower). If a flag is rejected by your llama.cpp build, check `llama-server --help`.
- If your model has a "thinking" mode that slows every call, disable it (see `AUTOLAB_EXTRA_PARAMS` below).

**2. (Optional) configure search.** For reliable results, get a [Serper.dev](https://serper.dev) key:

```bash
export SERPER_API_KEY="your-key"
```

Without a key, AutoLab falls back to free scraping backends through `ddgs`, which can return thin or empty results and may be rate-limited.

**3. Verify, then run in stages:**

```bash
python autolab.py --selftest      # offline tests, no model needed
python autolab.py --check         # preflight: folders, search, model server
python autolab.py --dry-run --once   # one full cycle, nothing saved to the dataset
python autolab.py --once          # one real cycle; inspect the output
python autolab.py                 # continuous mode (stops after 4 hours by default)
```

Common options:

```bash
python autolab.py \
  --url http://127.0.0.1:8080 \
  --model my-model \
  --ctx 32768 \
  --memory my_persona.md \
  --provider serper \
  --max-hours 12 --max-examples 500
```

## Command-line flags

| Flag | Meaning |
|---|---|
| `--once` / `--cycles N` | run one cycle / stop after N cycles |
| `--check` | preflight checks only, then exit |
| `--selftest` | offline tests plus a stub-server dry run |
| `--dry-run` | run full cycles but write nothing to the dataset |
| `--verify` / `--fix` | validate existing dataset files; with `--fix`, drop invalid lines (a backup is kept) |
| `--stats` | print dataset and lifetime statistics |
| `--topic "..."` | force the first cycle to research a specific question |
| `--base-dir PATH` | output root (default `./autolab_data`) |
| `--url URL` | llama-server base URL (default `http://127.0.0.1:8080`) |
| `--model NAME` | model id sent to the server |
| `--memory PATH` | optional persona/context text file |
| `--ctx N` | context window override |
| `--provider auto\|serper\|ddgs` | search provider |
| `--max-hours H` / `--max-examples N` | run limits |
| `--no-lock` | skip the single-instance lock |
| `-v` / `-q` | verbose / quiet logging |

## Environment variables

All optional. Flags override environment variables.

| Variable | Default | Meaning |
|---|---|---|
| `AUTOLAB_LLAMA_URL` | `http://127.0.0.1:8080` | llama-server base URL |
| `AUTOLAB_MODEL_NAME` | `local-model` | model id sent to the server |
| `AUTOLAB_LLAMA_API_KEY` | – | bearer token if the server runs with `--api-key` |
| `AUTOLAB_BASE_DIR` | `./autolab_data` | output root |
| `AUTOLAB_MEMORY_PATH` | – | optional text file added to the system prompt of the answer, repair and conclusion stages |
| `AUTOLAB_MEMORY_MAX_CHARS` | `6000` | how much of the memory file is used |
| `AUTOLAB_MEMORY_ROLES` | `candidate,repair,conclusion` | which stages receive the memory text |
| `AUTOLAB_AREAS_FILE` | – | text file, one research area per line (`#` comments allowed); replaces the built-in list |
| `AUTOLAB_SEARCH_PROVIDER` | `auto` | `auto` (Serper if a key is set), `serper`, or `ddgs` |
| `SERPER_API_KEY` | – | Serper.dev key |
| `AUTOLAB_CTX` | `131072` | context target; the smaller of this and the server's value is used |
| `AUTOLAB_MODEL_TIMEOUT` | `600` | seconds allowed per model call (raise it for huge prompts or slow hardware) |
| `AUTOLAB_EXTRA_PARAMS` | – | JSON merged into every request, e.g. `{"chat_template_kwargs":{"enable_thinking":false}}` |
| `AUTOLAB_MIN_QUALITY` / `AUTOLAB_MIN_ACCURACY` | `8.0` / `8.0` | minimum critic scores (0-10) to accept an example |
| `AUTOLAB_STRICT_GROUNDING` | `1` | reject examples that fail the automated grounding audit |
| `AUTOLAB_CANDIDATES` | `1` | answers generated per question |
| `AUTOLAB_REPAIR_ROUNDS` | `1` | repair attempts per answer |
| `AUTOLAB_SFT_CITATIONS` | `keep` | how `[S1]` markers are written to `sft.jsonl`: `keep`, `named`, or `strip` |
| `AUTOLAB_SFT_SYSTEM` | generic research-assistant prompt | system prompt stored in `sft.jsonl` records |
| `AUTOLAB_SAVE_SOURCE_TEXT` | `0` | save full fetched page text to disk (off by default) |
| `AUTOLAB_MAX_HOURS` / `AUTOLAB_MAX_EXAMPLES` | `4` / unlimited | run limits |

## Output layout

Everything is written under the base directory:

```
autolab_data/
  dataset/
    sft.jsonl                    accepted training examples (chat format + metadata)
    preferences.jsonl            preference pairs
    research_conclusions.jsonl   short conclusions per accepted cycle
    attempted_questions.json     used to avoid repeating questions
    stats.json, seen_hashes.json
  research/                      trace of each accepted cycle
  rejected/                      trace and reasons for each rejected cycle
  sources/, evaluations/
  logs/autolab.log
```

Each `sft.jsonl` record contains `messages` (system, user, assistant) plus `meta` with the sources used and the grounding audit result.

## Exit codes

`0` ok / clean stop · `1` fatal error · `2` missing dependency · `3` preflight failed · `4` another instance holds the lock

## Customising research topics

By default the planner rotates through a broad list of science, technology and history areas. To focus it on your own domain, create a text file with one area per line (see `areas.example.txt`) and point to it:

```bash
export AUTOLAB_AREAS_FILE=./my_areas.txt
```

## Troubleshooting

- **Preflight fails on the model server:** check the URL/port and that `curl http://127.0.0.1:8080/v1/models` answers.
- **Model calls are very slow:** check that all layers are on the GPU, and disable thinking mode with `AUTOLAB_EXTRA_PARAMS`. Keeping the KV cache in system RAM (`--no-kv-offload`) is a large slowdown.
- **Timeouts on huge prompts:** raise `AUTOLAB_MODEL_TIMEOUT`.
- **Many empty searches:** use a Serper key and `--provider serper`.
- **Almost everything is rejected:** read the reasons in `rejected/*.json` before loosening anything. The default gates are strict, and small quantized models often fail them at first. `AUTOLAB_MIN_QUALITY` and `AUTOLAB_STRICT_GROUNDING` control the gates, but relaxing them lets weaker examples into your dataset.

## Limitations

- **Self-generated data can reinforce mistakes.** A model that writes, reviews and curates its own training data can amplify its own errors and blind spots. The grounding audit and critic reduce this but do not eliminate it. Sample and read your data, and hold out an evaluation set that the pipeline never touches.
- **The grounding audit is heuristic.** It checks cited ids, numbers and quotations against the fetched text. It cannot judge whether a source is right or whether a paraphrase is faithful.
- **Web sources vary in quality.** Source scoring helps, but community platforms are treated as testimony rather than fact, and low-quality pages can still slip through.
- **The planner chooses its own topics.** It may pick sensitive or contested subjects. Review the output and, if needed, restrict topics with `AUTOLAB_AREAS_FILE`.

## Responsible use

- AutoLab fetches public web pages and, without a Serper key, scrapes search-engine results through `ddgs`. That may conflict with the terms of service of some search engines. Using a search API such as Serper avoids this.
- **This version does not check `robots.txt`.** Keep request volume reasonable and respect the sites you collect from.
- Generated examples can contain paraphrased or quoted passages from copyrighted pages. Saving full page text is off by default. Before publishing or redistributing a dataset, check the licenses of your sources. This is not legal advice.
- Do not commit your datasets, logs, or persona/memory files to a public repository (the included `.gitignore` excludes the default output folder).

## Development

```bash
python autolab.py --selftest
```

runs 96 offline checks covering parsing, grounding audit, file handling and two end-to-end cycles against a stub server. Please run it before opening a pull request.

## License

MIT. See [LICENSE](LICENSE).
