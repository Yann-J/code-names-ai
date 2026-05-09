from __future__ import annotations

from dataclasses import dataclass

from codenames_ai.agent.scoring import ScoringWeights, StopPolicy
from codenames_ai.game.models import Clue


@dataclass(frozen=True)
class RiskSnapshot:
    """Optional telemetry for one AI decision — omitted when unavailable (older traces / humans)."""

    base_risk: float
    effective_risk: float
    delta_objectives: float
    ours_unrevealed: int
    theirs_unrevealed: int
    dynamic_enabled: bool


@dataclass(frozen=True)
class ScoreComponents:
    """Breakdown for inspection; embedding score equals ``expected_reward_raw``."""

    expected_reward_raw: float
    """Monte Carlo estimate of expected turn reward for this clue and ``N``."""

    friendly_min_sim: float = 0.0
    """Lowest cosine similarity among the ``N`` included friendly targets (diagnostic)."""

    @property
    def total(self) -> float:
        return float(self.expected_reward_raw)


@dataclass(frozen=True)
class Candidate:
    """A scored `(clue, target_subset, N)` proposal.

    `score` is the **final** ranking score (used for sort order). When no LLM
    reranking is applied, `score == embedding_score == components.total`. When
    reranked, `score` blends EV with an LLM value proxy; `embedding_score` stays
    the pre-rerank EV.
    """

    clue: str
    targets: tuple[str, ...]
    n: int
    score: float
    embedding_score: float
    components: ScoreComponents
    margin: float
    zipf: float
    llm_score: float | None = None
    llm_reason: str | None = None


@dataclass(frozen=True)
class SpymasterTrace:
    """Full record of a spymaster decision: chosen clue plus runner-ups and metadata.

    `chosen` is `None` when no candidate survived the hard vetoes — i.e. the
    spymaster passes (M3 raises this case as an explicit `NoLegalClueError`;
    relaxation chain lands later).
    """

    chosen: Candidate | None
    top_candidates: tuple[Candidate, ...]
    weights: ScoringWeights
    veto_count: int
    illegal_count: int
    relaxation_steps: int = 0
    risk_snapshot: RiskSnapshot | None = None

    @property
    def clue(self) -> Clue:
        if self.chosen is None:
            return Clue(word="", count=0)
        return Clue(word=self.chosen.clue, count=self.chosen.n)


@dataclass(frozen=True)
class CandidateGuess:
    """An unrevealed card scored against a clue, with its rank in the final ordering.

    `similarity` is the raw cosine similarity to the clue word and is preserved
    through any LLM rerank. `score` is the final ranking score (used for sort
    order and stopping decisions); it equals `similarity` for embedding-only
    guessers and the blended value when a reranker is applied.
    """

    word: str
    similarity: float
    score: float
    rank: int  # 0-indexed in the descending final-score ranking
    committed: bool  # whether the guesser actually picked this card
    is_bonus: bool  # True only for the optional N+1 pick
    llm_score: float | None = None
    llm_reason: str | None = None


@dataclass(frozen=True)
class GuesserTrace:
    """Full record of a guesser decision for one clue."""

    candidates: tuple[CandidateGuess, ...]  # all unrevealed cards, sorted desc by sim
    guesses: tuple[str, ...]  # the actual ordered picks the guesser commits to
    stop_policy: StopPolicy
    bonus_attempted: bool
    stop_reason: str  # 'reached_n' | 'reached_n_plus_bonus' | 'confidence_floor' | 'no_more_candidates' | 'pass_clue'
    risk_snapshot: RiskSnapshot | None = None
