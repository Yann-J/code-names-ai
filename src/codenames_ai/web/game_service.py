from __future__ import annotations

from codenames_ai.cli.runtime import EvalRuntime
from codenames_ai.game.board import generate_board
from codenames_ai.game.human import HumanGuesser, HumanSpymaster
from codenames_ai.game.models import Card, Clue, Color
from codenames_ai.game.orchestrator import Game
from codenames_ai.game.rules import is_legal_clue
from codenames_ai.game.state import TurnEvent, TurnPhase, prior_clue_surfaces_lower
from codenames_ai.vocab.models import Vocabulary
from codenames_ai.web.play_session import PlaySession, Role

GuessFlashDict = dict[str, str]

ALLOWED_GUESS_FLASH_KINDS = frozenset({"team", "other", "assassin"})


def human_clue_lemma_for_surface(game_vocab: Vocabulary, surface_lower: str) -> str:
    """Lemma for legality checks; falls back to the surface when absent from game vocab."""
    df = game_vocab.df
    hit = df.loc[df["surface"] == surface_lower, "lemma"]
    if len(hit) > 0:
        return str(hit.iloc[0]).lower()
    return surface_lower


def human_clue_validation_error(
    *,
    turn_history: tuple[TurnEvent, ...],
    active_cards: tuple[Card, ...],
    game_vocab: Vocabulary,
    word_lower: str,
    count: int,
) -> str | None:
    """Return a user-facing error string, or ``None`` if the clue is allowed."""
    if word_lower == "" and count == 0:
        return None
    if word_lower == "":
        return "Clue word cannot be empty unless you pass (count 0)."
    prior = prior_clue_surfaces_lower(turn_history)
    if word_lower in prior:
        return "That clue was already used this game."
    lemma = human_clue_lemma_for_surface(game_vocab, word_lower)
    if not is_legal_clue(
        clue_surface=word_lower,
        clue_lemma=lemma,
        active_cards=active_cards,
        strictness="lemma_substring",
        forbidden_surfaces=None,
    ):
        return "Illegal clue: too close to a word still on the board."
    return None


def ensure_human_clue_legal(sess: PlaySession, game_vocab: Vocabulary, word_lower: str, count: int) -> None:
    """Raise ``ValueError`` with a short message if the clue violates board / reuse rules."""
    msg = human_clue_validation_error(
        turn_history=sess.game.state.turn_history,
        active_cards=sess.game.state.board.active(),
        game_vocab=game_vocab,
        word_lower=word_lower,
        count=count,
    )
    if msg is not None:
        raise ValueError(msg)


def role_key(team: Color, spymaster: bool) -> str:
    return f"{team.value.lower()}_{'spymaster' if spymaster else 'guesser'}"


def roles_have_human_guesser(roles: dict[Color, dict[str, Role]]) -> bool:
    return any(roles[team]["guesser"] == "human" for team in (Color.RED, Color.BLUE))


def roles_have_human_spymaster(roles: dict[Color, dict[str, Role]]) -> bool:
    return any(roles[team]["spymaster"] == "human" for team in (Color.RED, Color.BLUE))


def new_play_session(
    *,
    session_id: str,
    rt: EvalRuntime,
    seed: int,
    risk: float,
    roles: dict[Color, dict[str, Role]],
) -> PlaySession:
    humans: dict[str, HumanSpymaster | HumanGuesser] = {}
    rs, rg, bs, bg = make_players(rt, roles, humans)
    board = generate_board(rt.game_vocab, seed=seed)
    game = Game(
        board,
        red_spymaster=rs,
        red_guesser=rg,
        blue_spymaster=bs,
        blue_guesser=bg,
        seed=seed,
    )
    sess = PlaySession(id=session_id, game=game, roles=roles, humans=humans, risk=risk)
    advance_ai(sess)
    return sess


def rematch_play_session(sess: PlaySession, rt: EvalRuntime, new_seed: int) -> None:
    sess.humans.clear()
    sess.ui_guess_flash = None
    sess.last_ai_spymaster = None
    sess.last_ai_guesser = None
    rs, rg, bs, bg = make_players(rt, sess.roles, sess.humans)
    board = generate_board(rt.game_vocab, seed=new_seed)
    sess.game = Game(
        board,
        red_spymaster=rs,
        red_guesser=rg,
        blue_spymaster=bs,
        blue_guesser=bg,
        seed=new_seed,
    )
    advance_ai(sess)


def make_players(
    rt: EvalRuntime,
    roles: dict[Color, dict[str, Role]],
    humans: dict[str, HumanSpymaster | HumanGuesser],
) -> tuple:
    def wrap_spy(team: Color):
        if roles[team]["spymaster"] == "human":
            h = HumanSpymaster()
            humans[role_key(team, True)] = h
            return h
        return rt.spymaster

    def wrap_guess(team: Color):
        if roles[team]["guesser"] == "human":
            h = HumanGuesser()
            humans[role_key(team, False)] = h
            return h
        return rt.guesser

    return (
        wrap_spy(Color.RED),
        wrap_guess(Color.RED),
        wrap_spy(Color.BLUE),
        wrap_guess(Color.BLUE),
    )


def advance_ai(sess: PlaySession) -> None:
    g = sess.game
    while not g.state.is_over:
        team = g.state.current_team
        phase = g.state.current_phase
        if phase == TurnPhase.DONE:
            break
        kind = "spymaster" if phase == TurnPhase.SPYMASTER else "guesser"
        if sess.roles[team][kind] == "human":
            return
        clue_before: Clue | None = None
        if kind == "guesser":
            clue_before = g.state.latest_clue()
        n_spy_traces = len(g.spymaster_traces)
        n_guess_traces = len(g.guesser_traces)
        g.step()
        if kind == "spymaster" and len(g.spymaster_traces) > n_spy_traces:
            sess.last_ai_spymaster = (team, g.spymaster_traces[-1])
            sess.last_ai_guesser = None
        elif (
            kind == "guesser"
            and len(g.guesser_traces) > n_guess_traces
            and clue_before is not None
            and not clue_before.is_pass()
        ):
            sess.last_ai_guesser = (team, clue_before, g.guesser_traces[-1])
            sess.last_ai_spymaster = None


def apply_human_guess_words(sess: PlaySession, guesses: list[str]) -> GuessFlashDict | None:
    """Apply one or more human guesses in one request; mirrors the web form loop."""
    flash: GuessFlashDict | None = None
    for w in guesses:
        guess_team = sess.game.state.current_team
        sess.game.apply_human_guess(w)
        after_ev = sess.game.state.turn_history[-1] if sess.game.state.turn_history else None
        if after_ev and after_ev.kind == "GUESS" and after_ev.guess == w:
            oc = after_ev.outcome_color
            if oc is not None:
                if oc == guess_team:
                    kind = "team"
                elif oc == Color.ASSASSIN:
                    kind = "assassin"
                else:
                    kind = "other"
                flash = {"word": w, "kind": kind}
                sess.ui_guess_flash = flash
        if sess.game.state.is_over:
            break
        if sess.game.state.current_phase != TurnPhase.GUESSER:
            break
        if sess.roles[sess.game.state.current_team]["guesser"] != "human":
            break
    return flash
