# PRD-0001: Code Names AI Agent

**Triage label:** `ready-for-agent`

## Problem Statement

The user wants to play and study the word game *Code Names* with an AI agent that can reason about word associations across multiple languages. Existing Code Names bots are typically English-only, hardcode their vocabulary, and either rely purely on word embeddings (which miss semantic nuance like homonyms, double meanings, and cultural context) or purely on LLM prompting (which misses the systematic exploration of the embedding space that makes spymaster decisions interesting and tunable). The user also wants to use the system as a research tool — varying embedding models, LLM models, vocabulary sources, scoring weights, and risk tolerance — without rewriting the agent each time. Finally, the user wants to prototype every layer in a Jupyter notebook before committing to a full program with a web UI, so the architecture must support both pure-Python notebook usage and an interactive web client through the same module APIs.

## Solution

A Python package (`codenames_ai`) that:

1. **Builds language-specific vocabularies** from public word-frequency sources, with configurable frequency windows and exclusion lists, separated into a small "game word" pool (board cards) and a larger "clue word" pool.
2. **Projects vocabularies through a swappable embedding model**, caching the resulting embedding matrices on disk so the heavy load only happens once.
3. **Provides spymaster and guesser agents** that combine an embedding-driven candidate-generation algorithm with an LLM-based reranking step. Both agents share a single configurable risk-tolerance knob that adjusts margin requirements, ambition (target count), and stopping policy.
4. **Exposes a clean game model** (board, state, turn history, role-scoped views) that supports three orchestration modes — single-shot analysis from a notebook, AI-vs-AI self-play, and mixed human/AI play with arbitrary role assignment.
5. **Ships a FastAPI + HTMX web UI** with a Play mode (human vs AI or watch AI vs AI) and an Analysis mode (paste/generate a board, see ranked clue candidates with full reasoning, tweak risk live).
6. **Includes an evaluation harness** that runs self-play tournaments and curated golden-board tests, so the user can empirically confirm that config or scoring changes actually improve the bot.

All model choices (embedding provider, LLM provider) and tuning knobs are surfaced through a typed Pydantic configuration. English is the only language in v1; the architecture is language-parameterized so additional languages are a configuration + artifact-build away.

## User Stories

### Notebook prototyping (researcher/developer)

1. As a developer, I want to build a vocabulary for a given language with one function call, so that I can start exploring word data without writing pipeline glue.
2. As a developer, I want the vocabulary to load from a cached artifact on subsequent calls, so that iteration in a notebook is fast.
3. As a developer, I want to inspect the vocabulary as a pandas DataFrame (surface form, lemma, Zipf score, POS), so that I can eyeball quality and tune filters.
4. As a developer, I want to configure the frequency window via a Zipf range (e.g. `[3.0, 7.0]`), so that the same parameters work consistently across languages.
5. As a developer, I want to add words to a per-language exclusion file, so that I can suppress proper nouns, slurs, or other unwanted entries without rebuilding from source.
6. As a developer, I want separate game-word and clue-word vocabularies derived from the same source with different filters, so that board cards stay concrete-noun-heavy while clue candidates can include adjectives.
7. As a developer, I want the embedding matrix to load from a cached artifact keyed by vocabulary version, so that I don't reproject 50K vectors on every notebook restart.
8. As a developer, I want to query nearest neighbors of a word (`matrix.nearest('apple', k=10)`) directly, so that I can sanity-check the embedding quality on the vocabulary I built.
9. As a developer, I want to swap the embedding provider (fastText → sentence-transformers → API embeddings) by changing a config value, so that I can A/B different models without code changes.
10. As a developer, I want to swap the LLM provider (Anthropic → OpenAI → local) by changing a config value, so that I can compare reranking behavior across models.
11. As a developer, I want LLM responses cached by prompt hash to disk, so that re-running a notebook cell doesn't re-bill the API.
12. As a developer, I want to generate a board deterministically from a seed, so that I can reproduce specific game scenarios when investigating bugs or odd behavior.
13. As a developer, I want to invoke the spymaster against a board view and receive both the chosen clue and a structured `DecisionTrace` (top candidates, per-component scores, LLM scores and reasons), so that I can understand and tune behavior.
14. As a developer, I want to invoke the guesser against a board view, clue, and count and receive guesses plus a `DecisionTrace`, so that I can debug guessing decisions independently of the spymaster.
15. As a developer, I want to play a single turn end-to-end in a notebook (spymaster → clue → guesser → guesses), so that I can see the full pipeline before integrating into a game loop.

### Configuration and tuning

16. As a developer, I want a single `risk` scalar (0–1) that simultaneously adjusts margin requirements, ambition for higher target counts, and stopping policy, so that I can tune behavior with one intuitive knob.
17. As a developer, I want to override individual scoring weights when the single `risk` knob isn't expressive enough, so that I can do precise experimentation.
18. As a developer, I want to configure the blend weight (`α`) between embedding score and LLM score in the final ranking, so that I can quantify how much each layer contributes.
19. As a developer, I want to set the rule strictness (lemma-only, lemma + substring, or extended morphology) per-game, so that I can match the rules my players actually play by.
20. As a developer, I want to configure the `top-K` number of candidates passed to the LLM rerank step, so that I can trade cost for quality.
21. As a developer, I want config validation to fail loudly with a clear error when I pass `risk=2.0` or any other invalid value, so that I don't silently get wrong behavior.
22. As a developer, I want to load my Anthropic API key from an environment variable or `.env` file (never from a config YAML), so that I cannot accidentally commit secrets.

### Game play (interactive)

23. As a player, I want to start a new game in the web UI with a fresh randomly generated board, so that I can play immediately.
24. As a player, I want to assign each of the four roles (Red spymaster, Red guesser, Blue spymaster, Blue guesser) to either myself or the AI before the game starts, so that I can play whichever side or position I want.
25. As a player playing a guesser role, I want to see the board with revealed cards' colors visible and unrevealed cards as plain words, so that I have the same information a human player would.
26. As a player playing a spymaster role, I want to see the full board with all colors visible to me only, so that I can plan clues.
27. As a player whose AI teammate is the spymaster, I want to see the AI's chosen clue and target count after it commits, so that I know what to guess for.
28. As a player whose AI teammate is the guesser, I want to see each AI guess revealed in turn with the resulting card color, so that I can follow the game.
29. As a player, I want past clues and guesses from both teams visible in a turn-history panel, so that I can reference what's already happened.
30. As a player, I want the game to detect win conditions (own team's last card revealed = win, assassin revealed = instant loss, opponent's last card revealed = opponent wins) and end the game with a clear result, so that the game flow is correct.
31. As a player, I want the AI's reasoning hidden during play (so I can't peek at the spymaster's intended targets when I'm the opposing guesser), with a post-game review panel that surfaces it, so that play is fair but the agent's behavior is still inspectable afterwards.
32. As a player, I want to watch a fully autonomous AI-vs-AI game, so that I can see how the agent behaves end-to-end without participating.

### Spymaster's helper (Analysis mode)

33. As a spymaster operator, I want to paste or generate a board in Analysis mode, so that I can use the tool as a thinking aid rather than as a game.
34. As a spymaster operator, I want to see the top-N candidate clues ranked, with each candidate's intended target subset, target count, and full score breakdown (margin, ambition bonus, frequency bonus, assassin proximity penalty), so that I can compare options.
35. As a spymaster operator, I want to see the LLM's per-candidate score and one-sentence reason alongside the embedding score, so that I can understand why a candidate ranks where it does.
36. As a spymaster operator, I want to adjust the `risk` slider live and see the ranking re-compute, so that I can explore safer vs more ambitious lines.
37. As a spymaster operator, I want to see clearly when no legal clue exists and what got vetoed, so that I understand fallback behavior.

### Evaluation and improvement

38. As a developer, I want to run a self-play tournament of N games via a CLI command, so that I can produce baseline performance numbers for the current configuration.
39. As a developer, I want tournament results recorded as a parquet file with one row per game (config, seed, winner, turns, assassin-hit count, etc.), so that I can analyze the data with the tools I already use.
40. As a developer, I want to run two configs side-by-side and see a comparison table (win rate, average turns, assassin-hit rate, average guesses per clue), so that I can answer "did this change help?".
41. As a developer, I want a curated set of golden boards with annotated good clues (target subsets), so that I can detect regressions in candidate generation specifically.
42. As a developer, I want games to be fully reproducible from `(seed, config_hash, llm_cache)`, so that an interesting game can be replayed and analyzed.

### Operability

43. As a developer, I want a CLI command to download the fastText model for a language, so that the heavy data dependency is opt-in and explicit about disk usage.
44. As a developer, I want a clear runtime error with the install command when the spaCy model for a language is missing, so that I'm not blocked by an obscure import error.
45. As a developer, I want INFO-level logging of major events (vocab built, game started, clue chosen, guess made) and DEBUG-level logging of full reasoning traces, so that I can pick the right verbosity for ops vs debugging.
46. As a developer, I want every spymaster and guesser decision to also return a structured `DecisionTrace`, so that the analysis UI and tests have the data they need without parsing log output.
47. As a developer, I want all on-disk artifacts content-addressed by config hash with no overwrites, so that I can keep multiple vocabularies/embedding matrices around for comparison.

### Multilingual (future, M10)

48. As a developer, I want to add a new language by building a vocabulary and embedding matrix for it without changing any agent or game code, so that the language abstraction is real rather than aspirational.

## Implementation Decisions

### Scope and language

- **English-only in v1.** All public APIs accept a `language` parameter so a second language is added by building artifacts, not by changing code.
- The Zipf-scale frequency knobs and the artifact pipeline are language-agnostic from day one.

### Embeddings

- **Default embedding model: fastText English (`cc.en.300.bin`)**, behind an `EmbeddingProvider` interface. Sentence-transformers and API-based embeddings can be swapped in via configuration.
- The fastText binary is **not bundled and not auto-downloaded**. A CLI helper fetches it on demand into `~/.cache/codenames_ai/models/`. The path is overridable via `CODENAMES_AI_FASTTEXT_PATH`.
- Vocabularies are projected through the embedding provider once and cached as an `EmbeddingMatrix` (numpy `.npz` with vectors and surface-form index). After build, the fastText binary is not needed for normal operation.

### Vocabulary

- **Frequency source: `wordfreq` library** (Zipf-scale aggregated frequencies), behind a `FrequencyProvider` interface for future swap-in.
- **Filters applied during build:** lowercase; allowed character set = letters + hyphens; minimum length 3; spaCy POS tag and lemma stored alongside surface form.
- **Game-word vocabulary:** Zipf range `[4.0, 6.5]`, POS = noun. Yields ~500–2000 concrete common nouns suitable for board cards.
- **Clue-word vocabulary:** Zipf range `[3.0, 7.0]`, POS = noun + adjective. Yields ~30–60K candidate clues.
- **Exclusion list:** plain-text file per language (one word per line), gitignored by default for user customization.
- **Vocabulary artifact:** parquet on disk under `~/.cache/codenames_ai/vocab/<language>/<config_hash>.parquet`. Columns: `surface, lemma, zipf, pos`.

### Spymaster algorithm

- **Candidate generation:** for each clue word in the vocabulary, compute similarity to all 25 board words. For each clue, the only target subsets evaluated are prefixes of the friendly-similarity-sorted-descending list (because adding a less-similar friendly never improves the margin). Collapses to roughly `|vocab| × 9` evaluations per turn — a small set of numpy matrix operations.
- For each `(clue, friendly_subset_S)` candidate, compute:
  - `friendly_min_sim = min over s in S of sim(clue, s)`
  - `best_non_friendly_sim = max over n in non-friendlies of sim(clue, n)`
  - `margin = friendly_min_sim − best_non_friendly_sim`
- **Hard vetoes:** reject candidate if `margin < margin_floor` or if `sim(clue, assassin) > assassin_ceiling`.
- **Score formula:**
  - `score = friendly_min_sim + ambition_weight · (N − 1) + margin_weight · margin + freq_bonus(zipf(clue)) − assassin_weight · sim(clue, assassin) − opponent_weight · max sim(clue, opponent)`
  - `freq_bonus(z) = freq_weight · tanh((z − 3) / 2)` — saturating, peaks near Zipf ~5+, prevents ultra-common words from winning purely on frequency.
- **Single user-facing `risk` knob (0–1)** modulates `ambition_weight`, `margin_weight`, `margin_floor`, and `assassin_ceiling` along a fixed mapping. Power users may override individual weights.

### LLM rerank

- **Top 10 candidates** from the embedding-stage scoring pass to the LLM.
- The LLM sees the full board (with team colors) and the 10 candidates as `(clue, intended_targets, N)` tuples. It does **not** see the embedding similarities or formula scores; the orchestrator combines opinions.
- The LLM returns a 0–1 score and a one-sentence reason per candidate. Structured output enforced via the provider's schema mechanism.
- **Final ranking:** `α · normalized_embedding_score + (1−α) · llm_score`, with `α` configurable, defaulting to 0.5.
- **The LLM cannot propose its own clues in v1.** Future extension: separate "candidate source" feeding the same scoring/rerank machinery.
- **Default LLM:** Claude Sonnet 4.6 via the Anthropic SDK behind an `LLMProvider` interface. Responses cached by `(prompt_hash, model)` in SQLite.

### Guesser algorithm

- **Candidate pool:** unrevealed cards on the board. Revealed cards are physically removed from candidacy.
- **Embedding score:** cosine similarity of clue to each unrevealed card.
- **LLM rerank:** top `N + 3` unrevealed cards passed to LLM along with the full board state (revealed colors visible) and the clue. LLM returns 0–1 scores with reasons.
- **Blended score:** same `α` formula as the spymaster.
- **Stopping policy** controlled by the same `risk` knob:
  - Always commit to the #1 guess.
  - Picks 2..N: commit only if `blended_score > confidence_floor(risk)`.
  - Bonus N+1 pick: commit only if `risk` is high enough AND the gap between Nth and (N+1)th score is below `gap_threshold`.
- **No theory-of-mind simulation in v1.** A future version may have the guesser invoke the spymaster pipeline internally to ask "for which subset is this clue plausibly a high-scoring spymaster choice?".

### Game model

- **Card** = `(word, true_color, revealed)` where `true_color ∈ {RED, BLUE, NEUTRAL, ASSASSIN}`.
- **Board** = 25 Cards + `first_team` (which side has 9 cards).
- **TurnEvent** = `{team, kind: CLUE | GUESS, clue?, guess?, outcome?}`.
- **GameState** = board + turn history + current team + current phase + winner + rng_seed.
- **SpymasterView** = full board including all colors + turn history.
- **GuesserView** = board with colors visible only on revealed cards + turn history.
- Turn history is shared between both teams' views.

### Orchestration

- **Three modes supported in v1:**
  - Single-shot analysis (stateless, the notebook bread-and-butter).
  - AI-vs-AI self-play (`Game.play()` runs both teams to completion).
  - Mixed human/AI with arbitrary role assignment (any of the four roles can be `HumanPlayer` or `AIPlayer`).
- **`HumanPlayer` and `AIPlayer` implement the same `Spymaster` and `Guesser` interfaces.** The game loop is role-agnostic.
- **Determinism:** boards are deterministic given `(vocabulary_hash, seed)`. Bot decisions are deterministic given seed + temperature-0 LLM calls + cached LLM responses. Replaying a game from `GameState` history produces identical decisions.

### Rules engine

- **Default legality rule (configurable):** lemma match + substring check (both directions). Catches `cats`/`cat`, `running`/`run`, compounds like `catnap` when `cat` is on the board.
- **Hyphenated tokens** treated as single surface forms; substring check still fires across hyphen boundaries.
- **Strictness is configurable** (lemma-only, lemma+substring, extended morphology). Lemma+substring is the default.
- **No-legal-clue fallback:** progressively relax constraints — first lower `margin_floor` by 10% and retry, then again, then drop `ambition_weight` and try only N=1 clues with smaller subsets, then emit a pass (`clue=None`, `N=0`) with a `DecisionTrace` explaining what got vetoed. Logged as a warning; counted as a metric in the eval harness.

### Configuration

- **Pydantic v2** for the `Config` object. Validation, env-var loading, JSON serialization.
- **Secrets** (e.g. `ANTHROPIC_API_KEY`) loaded from environment variables, with optional `.env` file via `python-dotenv` if present. Never read from YAML configs.
- **Per-language overrides** supported in the config schema (different Zipf ranges, different exclusion lists per language).

### Storage and caching

- **Three artifact layers**, each content-addressed by config hash:
  - Vocabularies → parquet under `~/.cache/codenames_ai/vocab/<language>/<config_hash>.parquet`.
  - Embedding matrices → npz under `~/.cache/codenames_ai/embed/<language>/<vocab_hash>__<embed_provider_hash>.npz`.
  - LLM responses → SQLite at `~/.cache/codenames_ai/llm.sqlite`, keyed by `(prompt_hash, model)`.
- **No overwrites** — changing config produces a new artifact alongside old ones. Manual pruning only.
- **Cache directory** overridable via `CODENAMES_AI_CACHE_DIR`.

### Web UI

- **Stack:** FastAPI + HTMX + Jinja2. No JS framework, no build pipeline.
- **Modes in v1:** Play mode (A) and Analysis mode (B). Game replay (C) deferred.
- **Single-user, single-session.** Games held in an in-memory registry by game ID. Optionally serialized to the artifact cache for resume.
- **No DB, no auth, no multi-tenancy** in v1.
- **Reasoning visibility:** in Analysis mode, full reasoning surfaced (it's the point). In Play mode, reasoning hidden during play when humans hold opposing roles; surfaced in a post-game review panel.
- **Server-side state model** is the same `GameState` used by the rest of the system; the browser is a thin renderer.

### Observability

- **Standard library `logging`**, configured once. INFO for high-level events, DEBUG for full reasoning traces.
- **`DecisionTrace` is a first-class return value** from spymaster and guesser methods. Every decision returns `(result, trace)`. Drives the analysis UI and structured test assertions.

### Tooling

- **Python 3.12.**
- **`uv`** as the package manager. `uv pip install -e .` for editable install.
- **`src/` layout** to force editable install and catch packaging issues early.
- **`pytest`** for tests.
- **External data dependencies:**
  - fastText vectors: opt-in download via CLI. Documented disk-space cost.
  - spaCy model (`en_core_web_sm` for v1): documented `python -m spacy download` install. Vocab builder prints a friendly error with the install command if missing.

## Testing Decisions

### What makes a good test

- **Test external behavior, not implementation details.** Assert on the public interface of the module (return values, side effects) rather than on private helpers or intermediate structures.
- **Fast and deterministic.** No live API calls in unit tests. The LLM cache and a mocked `LLMProvider` make this trivial.
- **Synthetic inputs preferred for unit tests.** A 4-word board with hand-set similarities is more useful than a real 25-word board for testing the scoring formula, because the expected outcome is computable by hand.
- **One assertion's worth of behavior per test.** A test that checks "does the risk knob change the chosen clue?" should set up two configs and assert the choices differ — not also check the score values, the trace shape, etc. (those are separate tests).

### Modules to be tested

- **`vocab` filters** — POS, length, character-set, exclusions, Zipf range. Pure functions; trivial.
- **`agent.scoring`** — the score formula and the single-`risk`-knob mapping. Highest test-leverage module in the project. Hand-construct synthetic candidates with known scores; assert the formula picks the right one and the `risk` knob changes the choice.
- **`game.rules`** — legality check (lemma + substring), win-condition check. Pure functions.
- **`game.board`** — `Board.generate(seed)` is deterministic; correct color counts (9/8/7/1) given `first_team`; assassin count is always 1.
- **`agent.spymaster`** — full algorithm with a mocked `EmbeddingMatrix` and a mocked `LLMProvider`. Tests assert on the `DecisionTrace` (top candidates, scores, vetoes fired) rather than on logs. Includes the no-legal-clue fallback chain.
- **`agent.guesser`** — same approach: mocked deps, assert on `DecisionTrace` plus the chosen guess sequence and stopping behavior.
- **Integration test:** AI-vs-AI self-play on a fixed seed runs to completion and produces a winner without exceptions.
- **Integration test:** replaying a game from `GameState` history produces identical decisions (validates the full determinism story).

### Modules NOT directly unit-tested

- **`embedding` loader** — the npz cache and fastText projection are I/O-heavy and exercised end-to-end by the eval harness.
- **`llm` provider** — the API client and SQLite cache are integration territory; the agents test against a mocked `LLMProvider`.

### Eval harness as part of v1

- **Self-play tournament CLI** (`codenames-ai eval --runs N --config a.yaml --config b.yaml`) producing a comparison table of win rate, average turns, assassin-hit rate, average guesses per clue.
- **Curated golden boards** (10–20 boards with annotated good target subsets) asserting the spymaster's intended subset matches one of the human-labeled good options.
- **Tournament outputs** stored under `~/.cache/codenames_ai/evals/<run_id>.parquet` for analysis with the user's existing tools.
- The eval harness ships in milestone M7, **before the web UI**, so all subsequent tuning has empirical signal behind it.

### Prior art

- No prior art in this codebase (greenfield project).

## Out of Scope

The following are deliberately **not in scope for v1**:

- **Languages beyond English.** The architecture is parameterized; building a second language is M10, after the rest of the system is validated end-to-end in English.
- **Game replay UI** (mode C). The data is preserved in `GameState`; replay rendering is deferred.
- **Persistence layer / database.** Games live in the in-memory registry; the artifact cache provides optional disk serialization. Resumeable shareable game URLs require a real DB and are deferred.
- **Multi-user, authentication, multi-tenant deployment.** Single-user local app for v1.
- **Public-dataset benchmarks** (e.g. academic Code Names datasets). Self-play and golden boards provide enough signal for prototype-stage tuning.
- **LLM-proposed candidate clues.** v1 LLM is restricted to reranking embedding-derived candidates; LLM-as-source is a future extension.
- **Theory-of-mind guesser** that internally simulates the spymaster pipeline.
- **Special "zero" and "infinity" clues** allowed by official rules. v1 supports clues with `N ≥ 1`; passing emits `(clue=None, N=0)`.
- **Multi-word clues and acronyms.** Single-word clues only.
- **Phonetic or semantic-graph legality matching** (the "extended morphology" / WordNet rule). Plumbed in the strictness enum but disabled by default; not implemented in v1.
- **Stricter game-vocabulary POS filtering than noun.** Adjectives are allowed in the clue vocabulary but not the game vocabulary.
- **Auto-installing fastText or spaCy models.** Both remain explicit user actions.
- **Evaluating tuning across multiple LLM providers as part of CI.** The provider abstraction supports it; running such evals is a manual user activity.

## Further Notes

### Build order (implementation roadmap)

The project ships in eleven milestones, each leaving a notebook-usable deliverable:

- **M0 — Skeleton.** `pyproject.toml`, `src/codenames_ai/`, `Config`, `storage.py`, smoke import.
- **M1 — Vocab pipeline.** `FrequencyProvider`, `WordfreqProvider`, spaCy filters, `Vocabulary` builder, parquet artifact.
- **M2 — Embedding pipeline.** `EmbeddingProvider`, `FastTextProvider`, `EmbeddingMatrix`, npz artifact.
- **M3 — Spymaster v1 (embedding only, no LLM).** Algorithm B + scoring formula + `risk` knob.
- **M4 — Guesser v1 (embedding only).** Similarity ranking + stopping policy.
- **M5 — LLM layer.** `LLMProvider`, `AnthropicProvider`, SQLite cache, LLM rerank in both agents, blended scores.
- **M6 — Game orchestrator.** `Game.play()` runs AI-vs-AI to completion.
- **M7 — Eval harness.** Self-play tournaments + golden boards. First baseline numbers.
- **M8 — `HumanPlayer`.** Notebook-based human play; same interface the UI consumes.
- **M9 — Web UI.** FastAPI + HTMX, Play + Analysis modes.
- **M10 — Second language.** Validates that the language abstraction held.

### Two opinionated sequencing calls

- **Embedding-only spymaster (M3) ships before the LLM (M5).** This produces a working baseline whose performance is measurable before the LLM layer is added. The improvement attributable to the LLM can then be quantified by the eval harness (M7) rather than asserted by vibes.
- **Eval harness (M7) ships before the web UI (M9).** The harness is what makes the system self-improving — without it, the UI is the only way to evaluate, which is slow and subjective.
