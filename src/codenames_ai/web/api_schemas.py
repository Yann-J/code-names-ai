from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from codenames_ai.agent.trace import Candidate, SpymasterTrace
from codenames_ai.game.models import Card, Color
from codenames_ai.game.state import TurnPhase
from codenames_ai.web.play_session import PlaySession

GuessFlashKind = Literal["team", "other", "assassin"]
WinReason = Literal["assassin", "all_words"]


class GuessFlash(BaseModel):
    kind: GuessFlashKind
    word: str


class TeamRoles(BaseModel):
    spymaster: Literal["human", "ai"]
    guesser: Literal["human", "ai"]


class RolesPayload(BaseModel):
    red: TeamRoles
    blue: TeamRoles


class CluePayload(BaseModel):
    word: str
    count: int


class TurnEventPayload(BaseModel):
    team: str
    kind: Literal["CLUE", "GUESS"]
    clue: CluePayload | None = None
    guess: str | None = None
    outcome_color: str | None = None


class BoardCardPayload(BaseModel):
    """Guesser-safe by default: unrevealed cards omit true colors unless ``secret_color`` is set."""

    word: str
    revealed: bool
    revealed_as: str | None = None
    secret_color: str | None = Field(
        default=None,
        description="True card color for unrevealed cards when client requested spymaster overlay.",
    )


class GameUiPayload(BaseModel):
    show_end_turn: bool
    can_click_guess: bool
    waiting_on_ai: bool
    show_spymaster_form: bool
    show_guesser_form: bool


class GameSnapshot(BaseModel):
    id: str
    seed: int
    risk: float
    roles: RolesPayload
    current_team: str
    current_phase: str
    winner: str | None
    win_reason: WinReason | None = None
    is_over: bool
    guesser_attempts_remaining: int | None
    latest_clue: CluePayload | None
    guess_count_after_latest_clue: int
    cards: list[BoardCardPayload]
    turn_history: list[TurnEventPayload]
    ui: GameUiPayload
    guess_flash: GuessFlash | None = None


class CreateGameBody(BaseModel):
    seed: int = 0
    risk: float = 0.5
    red_spy: Literal["human", "ai"] = "ai"
    red_guess: Literal["human", "ai"] = "ai"
    blue_spy: Literal["human", "ai"] = "ai"
    blue_guess: Literal["human", "ai"] = "ai"


class CreateGameResponse(BaseModel):
    id: str
    state: GameSnapshot


class SpymasterGuessBody(BaseModel):
    word: str
    count: int = Field(ge=0, le=9)


class GuessesBody(BaseModel):
    words: list[str]


class ErrorResponse(BaseModel):
    detail: str


def _card_to_payload(c: Card, *, include_secret_colors: bool) -> BoardCardPayload:
    if c.revealed:
        return BoardCardPayload(
            word=c.word,
            revealed=True,
            revealed_as=c.color.value,
            secret_color=None,
        )
    sec = c.color.value if include_secret_colors else None
    return BoardCardPayload(
        word=c.word,
        revealed=False,
        revealed_as=None,
        secret_color=sec,
    )


def _clue_payload(clue) -> CluePayload | None:
    if clue is None:
        return None
    return CluePayload(word=clue.word, count=clue.count)


def _turn_events(sess: PlaySession) -> list[TurnEventPayload]:
    out: list[TurnEventPayload] = []
    for ev in sess.game.state.turn_history:
        clue_p = _clue_payload(ev.clue) if ev.clue else None
        out.append(
            TurnEventPayload(
                team=ev.team.value,
                kind=ev.kind,
                clue=clue_p,
                guess=ev.guess,
                outcome_color=ev.outcome_color.value if ev.outcome_color else None,
            )
        )
    return out


def _roles_payload(sess: PlaySession) -> RolesPayload:
    r = sess.roles
    return RolesPayload(
        red=TeamRoles(spymaster=r[Color.RED]["spymaster"], guesser=r[Color.RED]["guesser"]),
        blue=TeamRoles(spymaster=r[Color.BLUE]["spymaster"], guesser=r[Color.BLUE]["guesser"]),
    )


def build_game_snapshot(
    sess: PlaySession,
    *,
    include_secret_colors: bool,
    guess_flash: dict[str, str] | GuessFlash | None = None,
) -> GameSnapshot:
    g = sess.game
    st = g.state
    phase = st.current_phase
    team_now = st.current_team
    clue = st.latest_clue()

    show_end_turn = (
        phase == TurnPhase.GUESSER
        and clue is not None
        and not clue.is_pass()
        and sess.roles[team_now]["guesser"] == "human"
        and not st.is_over
        and st.guess_count_after_latest_clue() >= 1
    )

    can_click = (
        phase == TurnPhase.GUESSER
        and sess.roles[team_now]["guesser"] == "human"
        and not st.is_over
    )

    waiting_on_ai = not st.is_over and (
        (
            phase == TurnPhase.SPYMASTER
            and sess.roles[team_now]["spymaster"] == "ai"
        )
        or (
            phase == TurnPhase.GUESSER
            and sess.roles[team_now]["guesser"] == "ai"
        )
    )

    show_spy_form = (
        not st.is_over
        and phase == TurnPhase.SPYMASTER
        and sess.roles[team_now]["spymaster"] == "human"
    )
    show_guess_form = (
        not st.is_over
        and phase == TurnPhase.GUESSER
        and sess.roles[team_now]["guesser"] == "human"
    )

    flash_model: GuessFlash | None = None
    if guess_flash and guess_flash.get("kind") in ("team", "other", "assassin"):
        flash_model = GuessFlash(kind=guess_flash["kind"], word=guess_flash["word"])  # type: ignore[arg-type]

    winner_s = st.winner.value if st.winner else None
    win_reason: WinReason | None = None
    if st.is_over and st.winner is not None:
        last_guess = next((ev for ev in reversed(st.turn_history) if ev.kind == "GUESS"), None)
        if last_guess and last_guess.outcome_color == Color.ASSASSIN:
            win_reason = "assassin"
        else:
            win_reason = "all_words"

    return GameSnapshot(
        id=sess.id,
        seed=st.rng_seed,
        risk=sess.risk,
        roles=_roles_payload(sess),
        current_team=team_now.value,
        current_phase=phase.value,
        winner=winner_s,
        win_reason=win_reason,
        is_over=st.is_over,
        guesser_attempts_remaining=st.guesser_attempts_remaining,
        latest_clue=_clue_payload(clue) if clue and not clue.is_pass() else None,
        guess_count_after_latest_clue=st.guess_count_after_latest_clue(),
        cards=[_card_to_payload(c, include_secret_colors=include_secret_colors) for c in st.board.cards],
        turn_history=_turn_events(sess),
        ui=GameUiPayload(
            show_end_turn=show_end_turn,
            can_click_guess=can_click,
            waiting_on_ai=waiting_on_ai,
            show_spymaster_form=show_spy_form,
            show_guesser_form=show_guess_form,
        ),
        guess_flash=flash_model,
    )


# --- Analysis ---


class ScoreComponentsPayload(BaseModel):
    """Embedding score diagnostics; ``total`` equals MC expected reward (EV)."""

    expected_reward_raw: float
    friendly_min_sim: float = 0.0
    total: float


class CandidatePayload(BaseModel):
    clue: str
    targets: tuple[str, ...]
    n: int
    score: float
    embedding_score: float
    components: ScoreComponentsPayload
    margin: float
    zipf: float
    llm_score: float | None = None
    llm_reason: str | None = None


class ChosenPayload(BaseModel):
    clue: str
    n: int
    targets: tuple[str, ...]


class RiskSnapshotPayload(BaseModel):
    base_risk: float
    effective_risk: float
    delta_objectives: float
    ours_unrevealed: int
    theirs_unrevealed: int
    dynamic_enabled: bool


class SpymasterTracePayload(BaseModel):
    chosen: ChosenPayload | None
    top_candidates: list[CandidatePayload]
    veto_count: int
    illegal_count: int
    risk_snapshot: RiskSnapshotPayload | None = None


class AnalysisBoardCard(BaseModel):
    word: str
    color: str


class AnalysisResponse(BaseModel):
    seed: int
    risk: float
    traces: dict[str, SpymasterTracePayload]
    board: list[AnalysisBoardCard]
    first_team: str


class AnalysisRequestBody(BaseModel):
    seed: int = 0
    risk: float = 0.5


def spymaster_trace_to_payload(trace: SpymasterTrace) -> SpymasterTracePayload:
    rs_payload: RiskSnapshotPayload | None = None
    if trace.risk_snapshot is not None:
        rs = trace.risk_snapshot
        rs_payload = RiskSnapshotPayload(
            base_risk=rs.base_risk,
            effective_risk=rs.effective_risk,
            delta_objectives=rs.delta_objectives,
            ours_unrevealed=rs.ours_unrevealed,
            theirs_unrevealed=rs.theirs_unrevealed,
            dynamic_enabled=rs.dynamic_enabled,
        )

    def cand(c: Candidate) -> CandidatePayload:
        comp = c.components
        return CandidatePayload(
            clue=c.clue,
            targets=c.targets,
            n=c.n,
            score=c.score,
            embedding_score=c.embedding_score,
            components=ScoreComponentsPayload(
                expected_reward_raw=comp.expected_reward_raw,
                friendly_min_sim=comp.friendly_min_sim,
                total=comp.total,
            ),
            margin=c.margin,
            zipf=c.zipf,
            llm_score=c.llm_score,
            llm_reason=c.llm_reason,
        )

    chosen: ChosenPayload | None = None
    if trace.chosen:
        ch = trace.chosen
        chosen = ChosenPayload(clue=ch.clue, n=ch.n, targets=ch.targets)
    return SpymasterTracePayload(
        chosen=chosen,
        top_candidates=[cand(x) for x in trace.top_candidates],
        veto_count=trace.veto_count,
        illegal_count=trace.illegal_count,
        risk_snapshot=rs_payload,
    )
