# Word Guess AI

A Python package that plays a word guessing game with AI agents that combine **fastText embeddings** and **LLM reranking**.

## Play the demo

Access the AI agent at [this demo UI](https://guess.mind-it.rw/). You can assign any of the roles (spymaster or guesser from both teams) to either human or AI.

---

## How it works

### Vocabulary pipeline

Two vocabularies are built from [wordfreq](https://github.com/rspeer/wordfreq) Zipf-frequency data, filtered through spaCy for POS tagging and lemmatisation:

- **Game-word vocabulary** — Zipf 4–6.5, nouns only (~500–2000 common concrete nouns for board cards).
- **Clue-word vocabulary** — Zipf 3–7, nouns + adjectives + verbs (~30–60 K candidate clue words).

Both are filtered with a configurable profanity exclusion list (default for English at `./data/exclusions/en` from the [LDNOOBW bad-words submodule](https://github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words))

Both compiled vocabularies are cached as Parquet files under `~/.cache/codenames_ai/vocab/`.

### Embedding pipeline

Every word in the clue vocabulary is projected through a **fastText model** (`cc.en.300.bin`, ~7 GB on disk) and cached as an L2-normalised NumPy matrix (`.npz`). After the first build, the fastText binary is no longer needed at runtime.

### Spymaster agent

1. Compute cosine similarity of every clue candidate against every active board word (one NumPy matrix multiply).
1. For each clue, consider only *prefix subsets* of the descending-similarity-sorted friendlies (the only subsets that can improve margin).
1. Apply **hard vetoes**: margin below floor, or similarity to the assassin above ceiling.
1. Score survivors (based on a fairly complex rule)
1. Filter for legal clues (drop clues too close to a game word, or offensive words from a simple blacklist dictionary)
1. Build a lane-balanced shortlist for LLM rerank (target fractions across clue counts `N=1..7`, EV-first ranking within each lane, quality gate + graceful backfill).
1. Pass the shortlist to an **LLM** (any OpenAI-compatible endpoint) for scoring with a one-sentence reason, in order to capture subtle word connections that pure embeddings miss.
1. Blend with EV-aware LLM adjustment: `α × normalised_embedding_score + (1−α) × llm_adjusted`, where `llm_adjusted` is boosted or damped by expected reward relative to shortlist median EV.

### Guesser agent

Ranks unrevealed board cards by cosine similarity to the clue word, optionally reranked by the same LLM. Applies a stopping policy: always takes the top guess, takes picks 2..N only while confidence exceeds a floor, and may take a bonus N+1 guess if the gap to the next card is small.

### Risk knob

A single `risk` scalar in `[0, 1]` simultaneously tunes:

| Parameter | risk = 0 (cautious) | risk = 1 (aggressive) |
|---|---|---|
| Margin floor | 0.10 | 0.00 |
| Assassin ceiling | 0.25 | 0.45 |
| Ambition weight | 0.02 | 0.15 |
| Confidence floor (guesser) | 0.30 | −1.0 (always commit) |
| Bonus N+1 pick | disabled | enabled |

### Game model

`GameState` holds the board, turn history, current team, and phase. Three modes:

- **Single-shot analysis** — call `AISpymaster.give_clue()` directly from a notebook.
- **AI-vs-AI self-play** — `Game.play()` runs both teams to completion.
- **Mixed human/AI** — assign any combination of `HumanSpymaster`, `HumanGuesser`, `AISpymaster`, `AIGuesser` to the four roles.

---

## Requirements

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- The **spaCy English model** ships as a direct dependency and is installed automatically.
- The **fastText English vectors** (~7 GB) are **not bundled** — download them explicitly (see below).

---

## Installation

```bash
git clone --recurse-submodules <repo>
cd code-names-ai

# If you cloned without submodules:
# git submodule update --init --recursive

# Install the package and its dependencies
uv pip install -e ".[dev]"
```

> Without `uv`, use `pip install -e ".[dev]"` inside a venv.

---

## Credentials

Copy `.env.example` to `.env` (the file is gitignored) and fill in your LLM credentials:

```bash
# OpenAI
LLM_API=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
LLM_KEY=sk-...

# Mistral (alternative)
# LLM_API=https://api.mistral.ai/v1
# LLM_MODEL=mistral-small
# LLM_KEY=...

# Ollama (local, no key needed)
# LLM_API=http://localhost:11434/v1
# LLM_MODEL=llama3.2
# LLM_KEY=ollama
```

LLM responses are cached in `~/.cache/codenames_ai/llm.sqlite` by prompt hash, so re-running the same game never re-bills the API.

---

## CLI

### Download fastText vectors

```bash
codenames-ai download fasttext --lang en
```

Downloads ~7 GB into `~/.cache/codenames_ai/models/cc.en.300.bin`. Pass `--force` to re-download. This is the only command that requires internet access at runtime; all subsequent commands work offline.

Override the path with `CODENAMES_AI_FASTTEXT_PATH=/path/to/cc.en.300.bin`.

### Run golden-board regression tests

```bash
codenames-ai golden
```

Runs three curated synthetic boards through the spymaster and checks that it finds the correct clusters. No external model or API needed — uses isolated synthetic embeddings. Exits `0` if pass rate is 100%.

### Self-play evaluation tournament

Use `config/base.yaml` as the single source of truth for eval/serve YAML fields and defaults.

Run a 20-game tournament:

```bash
codenames-ai eval --runs 20 --config baseline.yaml
```

Compare two configs side by side:

```bash
codenames-ai eval --runs 20 --config conservative.yaml --config aggressive.yaml
```

Output is a Parquet file in `~/.cache/codenames_ai/evals/` with one row per game. Columns: `label`, `config_hash`, `seed`, `winner`, `first_team`, `num_clues`, `num_guesses`, `correct_guesses`, `assassin_hit`. Summary stats are printed to stdout.

Use `--embedding-only` to override `llm_rerank` in YAML and run without any LLM calls.

### Web UI

```bash
codenames-ai serve
# or: codenames-ai serve --host 0.0.0.0 --port 8080
```

Opens at `http://127.0.0.1:8000` and serves the React PWA shell plus JSON API endpoints.

> The web server requires the vocabulary and embedding matrix caches to exist. Run `codenames-ai download fasttext` once and the first `eval` or `serve` call will build the caches automatically.

---

## Docker

A single production-optimised image bundles everything: Python deps installed with `uv` against `uv.lock`, the React PWA built fresh by Vite in a node stage, and FastAPI/uvicorn serving the API and PWA shell. Runs as a non-root user, exposes port `8000`.

### Build and run with docker-compose

```bash
cp .env.example .env  # fill in LLM_KEY etc.
docker compose up --build
```

- App shell + JSON API: <http://localhost:8000>
- React PWA route: <http://localhost:8000/app/>

The `codenames-cache` named volume persists vocabulary, embedding-matrix, and LLM caches under `/cache` (`CODENAMES_AI_CACHE_DIR=/cache`). To use fastText, download `cc.en.300.bin` into that volume:

```bash
docker compose run --rm app codenames-ai download fasttext --lang en
```

### Build the image standalone

```bash
docker build -t codenames-ai:latest .
```

| Secret | Purpose |
|---|---|
| `LLM_API` | LLM endpoint (e.g. `https://api.openai.com/v1`) |
| `LLM_MODEL` | LLM model name (e.g. `gpt-4o-mini`) |
| `LLM_KEY` | API key for the LLM endpoint |

> The compose file references `ghcr.io/yann-j/code-names-ai:latest` by default; override per environment with `IMAGE` in `.env`. The GHCR package visibility can stay `private` — the deploy step authenticates with the workflow's `GITHUB_TOKEN`.

---

## Running tests

```bash
uv run pytest
```

Unit tests use synthetic embeddings and a mock LLM — no API key and no fastText binary required. Integration tests that need external resources are skipped automatically when the relevant env vars are absent.

---

## Cache layout

```
~/.cache/codenames_ai/
  vocab/
    en/<config_hash>.parquet    # vocabulary artifacts (game-word and clue)
  embed/
    en/<vocab_hash>__<provider_hash>.npz   # embedding matrix
  models/
    cc.en.300.bin               # fastText binary (downloaded on demand)
  evals/
    eval_20260507_213042.parquet   # tournament output
  llm.sqlite                    # LLM response cache
```

All artifacts are content-addressed by config hash. Changing a config produces a new file alongside old ones; no overwrites. The cache directory is overridable via `CODENAMES_AI_CACHE_DIR`.

---

## Architecture overview

```
codenames_ai/
  vocab/        — frequency source, spaCy filters, vocabulary builder
  embedding/    — EmbeddingProvider interface, FastTextProvider, matrix cache
  agent/        — AISpymaster, AIGuesser, SpymasterReranker, GuesserReranker,
                  scoring formula, DecisionTrace
  llm/          — LLMProvider interface, OpenAICompatibleProvider, SQLite cache
  game/         — Board, Card, Color, GameState, rules, orchestrator, HumanPlayer
  eval/         — tournament runner, metrics, persist, golden boards
  cli/          — CLI entry point, eval YAML schema, runtime wiring
  web/          — FastAPI app, JSON API routes, PWA static serving
```

The dependency graph flows one way: `game` → `agent` → `llm/embedding/vocab`. The CLI and web layers sit on top and import everything. The top-level `codenames_ai` package re-exports the full public surface.
