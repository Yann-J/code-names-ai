# Code Names AI

A Python package that plays the word game [Code Names](https://en.wikipedia.org/wiki/Codenames) with AI agents that combine **fastText embeddings** and **LLM reranking**. Use it as a research tool in a Jupyter notebook, run headless self-play tournaments to measure bot quality, or spin up a local web UI to play against (or alongside) the AI.

---

## How it works

### Vocabulary pipeline

Two vocabularies are built from [wordfreq](https://github.com/rspeer/wordfreq) Zipf-frequency data, filtered through spaCy for POS tagging and lemmatisation:

- **Game-word vocabulary** — Zipf 4–6.5, nouns only (~500–2000 common concrete nouns for board cards).
- **Clue-word vocabulary** — Zipf 3–7, nouns + adjectives + verbs (~30–60 K candidate clue words).

Both are cached as Parquet files under `~/.cache/codenames_ai/vocab/`.

### Embedding pipeline

Every word in the clue vocabulary is projected through a **fastText model** (`cc.en.300.bin`, ~7 GB on disk) and cached as an L2-normalised NumPy matrix (`.npz`). After the first build, the fastText binary is no longer needed at runtime.

### Spymaster agent

1. Compute cosine similarity of every clue candidate against every active board word (one NumPy matrix multiply).
1. For each clue, consider only *prefix subsets* of the descending-similarity-sorted friendlies (the only subsets that can improve margin).
1. Apply **hard vetoes**: margin below floor, or similarity to the assassin above ceiling.
1. Score survivors:

   ```
   score = friendly_min_sim
         + ambition_weight × (N−1)
         + margin_weight × margin
         + freq_bonus(zipf)
         − assassin_weight × sim(clue, assassin)
         − opponent_weight × max sim(clue, opponent)
   ```

1. Filter for legal clues (drop clues too close to a game word, or offensive words from a simple blacklist dictionary)
1. Pass the top-K candidates to an **LLM** (any OpenAI-compatible endpoint) for scoring with a one-sentence reason, in order to capture subtle word connections that wouldn't be surfaced by the pure embeddings.
1. Blend: `α × normalised_embedding_score + (1−α) × llm_score`.

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
git clone <repo>
cd code-names-ai

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

Create a YAML config file (see all defaults below):

```yaml
label: baseline
risk: 0.5
language: en
llm_rerank: true   # set false to skip LLM and use embedding-only agents
game_zipf:
  min: 4.0
  max: 6.5
clue_zipf:
  min: 3.0
  max: 7.0
embedding_top_k: 20    # top embedding candidates sent to the spymaster LLM
top_k_trace: 200       # how many ranked candidates to keep in the trace / API
blend_alpha: 0.5
```

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

#### All YAML fields

| Key | Default | Description |
|---|---|---|
| `label` | `"default"` | Name shown in comparison table and stored in parquet |
| `risk` | `0.5` | Risk knob 0–1 |
| `language` | `"en"` | Language code |
| `game_zipf` | `{min: 4.0, max: 6.5}` | Frequency window for board-card vocabulary (nested `min` / `max`) |
| `clue_zipf` | `{min: 3.0, max: 7.0}` | Frequency window for clue vocabulary (nested `min` / `max`) |
| `game_allowed_pos` | `["NOUN"]` | spaCy POS tags kept for game words |
| `clue_allowed_pos` | `["NOUN", "ADJ", "VERB"]` | spaCy POS tags kept for clues |
| `exclusions_path` | `null` | Path to a one-word-per-line exclusion file |
| `top_k_trace` | `200` | How many candidates to surface in the `SpymasterTrace` after rerank |
| `llm_rerank` | `true` | Enable LLM reranking step |
| `embedding_top_k` | `20` | Top embedding-scored candidates sent to the spymaster LLM |
| `blend_alpha` | `0.5` | α in `α·embedding + (1−α)·llm` blend |
| `guesser_extra_candidates` | `3` | Extra candidates passed to the guesser LLM beyond N |
| `prefer_min_targets` | `3` | Soft minimum friendly count each clue should aim for |

Deprecated YAML keys (still accepted): `game_zipf_min` / `game_zipf_max` → `game_zipf`, `clue_zipf_min` / `clue_zipf_max` → `clue_zipf`, and `rerank_top_k` → `embedding_top_k`.

### Web UI

```bash
codenames-ai serve
# or: codenames-ai serve --host 0.0.0.0 --port 8080
```

Opens at `http://127.0.0.1:8000`. Two modes:

- **Play** — pick a seed, set risk, assign each of the four roles (Red spymaster, Red guesser, Blue spymaster, Blue guesser) to Human or AI. Human spymasters enter a clue word and count; human guessers enter a comma-separated list of board words.
- **Analysis** — enter a seed and risk level; the page shows the spymaster's full ranked candidate list with embedding scores, LLM scores, margins, and reasons.

> The web server requires the vocabulary and embedding matrix caches to exist. Run `codenames-ai download fasttext` once and the first `eval` or `serve` call will build the caches automatically.

---

## Notebook / library usage

```python
from codenames_ai import (
    Config, StoragePaths,
    VocabConfig, load_or_build_vocabulary,
    FastTextProvider, load_or_build_embedding_matrix,
    AISpymaster, AIGuesser,
    generate_board, SpymasterView, GuesserView,
)

config = Config()
storage = StoragePaths.from_config(config)

# Build (or load cached) vocabularies
game_vocab_cfg = VocabConfig(language="en", zipf_min=4.0, zipf_max=6.5,
                              allowed_pos=frozenset({"NOUN"}))
game_vocab = load_or_build_vocabulary(game_vocab_cfg, storage)

clue_vocab_cfg = VocabConfig(language="en", zipf_min=3.0, zipf_max=7.0,
                              allowed_pos=frozenset({"NOUN", "ADJ"}))
clue_vocab = load_or_build_vocabulary(clue_vocab_cfg, storage)

# Load (or build cached) embedding matrix
provider = FastTextProvider(storage.models_dir / "cc.en.300.bin")
matrix = load_or_build_embedding_matrix(clue_vocab, provider, storage)

# Create agents
spymaster = AISpymaster(matrix, clue_vocab, risk=0.5)
guesser   = AIGuesser(matrix, risk=0.5)

# Generate a board and run one turn
board = generate_board(game_vocab, seed=42)
spy_trace = spymaster.give_clue(SpymasterView(board=board, team=board.first_team))

print(spy_trace.chosen.clue, spy_trace.chosen.n, spy_trace.chosen.targets)
# Inspect full trace
for c in spy_trace.top_candidates[:5]:
    print(f"  {c.clue:20s}  N={c.n}  score={c.score:.3f}  margin={c.margin:.3f}")
```

### LLM reranking (optional)

```python
from codenames_ai import LLMCache, OpenAICompatibleProvider
from codenames_ai.agent.rerank import SpymasterReranker, GuesserReranker

cache = LLMCache(storage.llm_cache_path)
llm = OpenAICompatibleProvider(
    model="gpt-4o-mini",
    base_url="https://api.openai.com/v1",
    api_key="sk-...",
    cache=cache,
)

spymaster = AISpymaster(
    matrix, clue_vocab, risk=0.5,
    reranker=SpymasterReranker(llm, top_k=200, blend_alpha=0.5),
)
```

### Human-in-the-loop play

```python
from codenames_ai import Game, HumanSpymaster, trivial_spymaster_trace

human_spy = HumanSpymaster()
game = Game(board,
            red_spymaster=human_spy,
            red_guesser=guesser,
            blue_spymaster=spymaster,
            blue_guesser=guesser)

# Before the game reaches the human spymaster phase, supply the clue:
human_spy.prepare(trivial_spymaster_trace("ocean", targets=(), n=2))
game.step()  # processes the human clue, then AI guesses
```

### Eval harness in a notebook

```python
from codenames_ai import run_tournament, aggregate, compare, save_records
from codenames_ai import iter_golden_cases, evaluate_golden, golden_pass_rate

# Self-play tournament
records = run_tournament(
    seeds=range(50),
    game_vocab=game_vocab,
    red_spymaster=spymaster, red_guesser=guesser,
    blue_spymaster=spymaster, blue_guesser=guesser,
    label="risk=0.5",
)
print(aggregate(records))

# Golden board regression
results = [evaluate_golden(spy, g) for g, spy in iter_golden_cases()]
print(golden_pass_rate(results))
```

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
  web/          — FastAPI app, Jinja2 templates (Play + Analysis modes)
```

The dependency graph flows one way: `game` → `agent` → `llm/embedding/vocab`. The CLI and web layers sit on top and import everything. The top-level `codenames_ai` package re-exports the full public surface.
