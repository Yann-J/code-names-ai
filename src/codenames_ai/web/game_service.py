from __future__ import annotations

from codenames_ai.cli.runtime import EvalRuntime
from codenames_ai.game.human import HumanGuesser, HumanSpymaster
from codenames_ai.game.models import Color
from codenames_ai.game.state import TurnPhase
from codenames_ai.web.play_session import PlaySession, Role

GuessFlashDict = dict[str, str]

ALLOWED_GUESS_FLASH_KINDS = frozenset({"team", "other", "assassin"})


def role_key(team: Color, spymaster: bool) -> str:
    return f"{team.value.lower()}_{'spymaster' if spymaster else 'guesser'}"


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
        g.step()


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
