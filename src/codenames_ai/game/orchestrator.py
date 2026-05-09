from __future__ import annotations

import logging
from dataclasses import replace

from codenames_ai.agent.interfaces import Guesser, NoLegalClueError, Spymaster
from codenames_ai.agent.trace import GuesserTrace, SpymasterTrace
from codenames_ai.game.models import Board, Clue, Color, GuesserView, SpymasterView
from codenames_ai.game.state import (
    GameState,
    TurnEvent,
    TurnPhase,
    check_win,
    prior_clue_surfaces_lower,
    reveal_card,
)

logger = logging.getLogger(__name__)


class Game:
    """Orchestrates a Code Names match through phase-by-phase steps.

    The state machine alternates `TurnPhase.SPYMASTER` and `TurnPhase.GUESSER`
    for the current team; `step()` advances exactly one phase, `play()` runs
    to completion. `Game` is mode-agnostic: any combination of `AISpymaster` /
    `AIGuesser` / `HumanPlayer` (M8) implementations slots in.

    Per-phase traces are captured on the instance so a UI or eval harness can
    surface them; the `GameState.turn_history` stays lean (no traces) so it
    remains pickleable / JSON-serializable.
    """

    def __init__(
        self,
        board: Board,
        *,
        red_spymaster: Spymaster,
        red_guesser: Guesser,
        blue_spymaster: Spymaster,
        blue_guesser: Guesser,
        seed: int = 0,
        max_clues: int = 50,
    ) -> None:
        self._players: dict[tuple[Color, str], Spymaster | Guesser] = {
            (Color.RED, "spymaster"): red_spymaster,
            (Color.RED, "guesser"): red_guesser,
            (Color.BLUE, "spymaster"): blue_spymaster,
            (Color.BLUE, "guesser"): blue_guesser,
        }
        self.max_clues = max_clues
        self.state = GameState(
            board=board,
            turn_history=(),
            current_team=board.first_team,
            current_phase=TurnPhase.SPYMASTER,
            winner=None,
            rng_seed=seed,
            guesser_attempts_remaining=None,
        )
        self.spymaster_traces: list[SpymasterTrace] = []
        self.guesser_traces: list[GuesserTrace] = []
        self._clue_count = 0

    @classmethod
    def from_state(
        cls,
        state: GameState,
        *,
        red_spymaster: Spymaster,
        red_guesser: Guesser,
        blue_spymaster: Spymaster,
        blue_guesser: Guesser,
        max_clues: int = 50,
    ) -> Game:
        """Build a game anchored on an existing ``GameState`` (fork / resume)."""
        self = object.__new__(cls)
        self._players = {
            (Color.RED, "spymaster"): red_spymaster,
            (Color.RED, "guesser"): red_guesser,
            (Color.BLUE, "spymaster"): blue_spymaster,
            (Color.BLUE, "guesser"): blue_guesser,
        }
        self.max_clues = max_clues
        self.state = state
        self.spymaster_traces = []
        self.guesser_traces = []
        self._clue_count = sum(1 for ev in state.turn_history if ev.kind == "CLUE")
        return self

    def step(self) -> GameState:
        """Advance one phase. No-op if the game is already over."""
        if self.state.is_over:
            return self.state
        if self.state.current_phase == TurnPhase.SPYMASTER:
            self._step_spymaster()
        else:
            self._step_guesser()
        return self.state

    def play(self) -> GameState:
        while not self.state.is_over:
            self.step()
        return self.state

    def _step_spymaster(self) -> None:
        team = self.state.current_team
        spymaster: Spymaster = self._players[(team, "spymaster")]  # type: ignore[assignment]
        prior_clues = prior_clue_surfaces_lower(self.state.turn_history)
        view = SpymasterView(
            board=self.state.board, team=team, prior_clue_words=prior_clues
        )

        try:
            trace = spymaster.give_clue(view)
            self.spymaster_traces.append(trace)
            clue = trace.clue
        except NoLegalClueError:
            logger.info("spymaster %s passed (no legal clue)", team.value)
            clue = Clue(word="", count=0)

        self._record_event(
            TurnEvent(team=team, kind="CLUE", clue=clue),
            advance_phase=TurnPhase.GUESSER,
        )
        self._clue_count += 1
        if self._clue_count >= self.max_clues:
            logger.info("max_clues reached; ending game without winner")
            self._end()

    def _step_guesser(self) -> None:
        team = self.state.current_team
        clue = self.state.latest_clue()
        if clue is None or clue.is_pass():
            # No clue this turn (spymaster passed). Switch to opponent.
            self._switch_team()
            return

        guesser: Guesser = self._players[(team, "guesser")]  # type: ignore[assignment]
        view = GuesserView(board=self.state.board, team=team)
        trace = guesser.guess(view, clue)
        self.guesser_traces.append(trace)

        ar = self.state.guesser_attempts_remaining
        for guess_word in trace.guesses:
            try:
                new_board = reveal_card(self.state.board, guess_word)
            except ValueError:
                # Guesser proposed an already-revealed or unknown card.
                # Treat as turn-ending error.
                logger.warning(
                    "guesser %s proposed invalid card %r; ending turn",
                    team.value,
                    guess_word,
                )
                break
            revealed_card = next(c for c in new_board.cards if c.word == guess_word)
            outcome_color = revealed_card.color
            event = TurnEvent(
                team=team,
                kind="GUESS",
                guess=guess_word,
                outcome_color=outcome_color,
            )
            next_ar = ar - 1 if ar is not None else None
            self.state = replace(
                self.state,
                board=new_board,
                turn_history=self.state.turn_history + (event,),
                guesser_attempts_remaining=next_ar,
            )
            ar = next_ar

            winner = check_win(self.state)
            if winner is not None:
                self._end(winner=winner)
                return
            if outcome_color != team:
                # Wrong color — turn ends regardless of remaining picks.
                break
            if ar is not None and ar <= 0:
                # Used all guesses allowed for this clue.
                break

        if not self.state.is_over:
            self._switch_team()

    def apply_human_guess(self, word: str) -> None:
        """Reveal one card during a human guesser's turn (one HTTP action per guess)."""
        if self.state.is_over or self.state.current_phase != TurnPhase.GUESSER:
            raise ValueError("not in guesser phase")
        team = self.state.current_team
        clue = self.state.latest_clue()
        if clue is None or clue.is_pass():
            return

        guess_word = word.strip().lower()
        ar = self.state.guesser_attempts_remaining
        if ar is not None and ar <= 0:
            raise ValueError("no guesses remaining this turn")

        matching = [c for c in self.state.board.cards if c.word == guess_word]
        if not matching:
            logger.warning(
                "human guesser %s chose unknown card %r; ending turn",
                team.value,
                guess_word,
            )
            if not self.state.is_over:
                self._switch_team()
            return
        if matching[0].revealed:
            # Duplicate POST / double-click on a flipped card: do not end the turn.
            return

        new_board = reveal_card(self.state.board, guess_word)

        revealed_card = next(c for c in new_board.cards if c.word == guess_word)
        outcome_color = revealed_card.color
        event = TurnEvent(
            team=team,
            kind="GUESS",
            guess=guess_word,
            outcome_color=outcome_color,
        )
        next_ar = ar - 1 if ar is not None else None
        self.state = replace(
            self.state,
            board=new_board,
            turn_history=self.state.turn_history + (event,),
            guesser_attempts_remaining=next_ar,
        )

        winner = check_win(self.state)
        if winner is not None:
            self._end(winner=winner)
            return

        if outcome_color != team:
            self._switch_team()
            return

        if next_ar is not None and next_ar <= 0:
            self._switch_team()
            return
        # Else stay in GUESSER phase for another pick.

    def end_guessing_turn(self) -> None:
        """Voluntarily stop guessing for this clue (Codenames: after ≥1 guess)."""
        if self.state.is_over or self.state.current_phase != TurnPhase.GUESSER:
            raise ValueError("not in guesser phase")
        clue = self.state.latest_clue()
        if clue is None or clue.is_pass():
            return
        if self.state.guess_count_after_latest_clue() < 1:
            raise ValueError("make at least one guess before ending the turn")
        if not self.state.is_over:
            self._switch_team()

    def _record_event(self, event: TurnEvent, *, advance_phase: TurnPhase) -> None:
        guess_rem: int | None
        if advance_phase == TurnPhase.GUESSER and event.kind == "CLUE":
            c = event.clue
            if c and not c.is_pass():
                guess_rem = 25 if c.count == 0 else c.count + 1
            else:
                guess_rem = None
        else:
            guess_rem = self.state.guesser_attempts_remaining
        self.state = replace(
            self.state,
            turn_history=self.state.turn_history + (event,),
            current_phase=advance_phase,
            guesser_attempts_remaining=guess_rem,
        )

    def _switch_team(self) -> None:
        if self.state.is_over:
            return
        self.state = replace(
            self.state,
            current_team=self.state.current_team.opponent(),
            current_phase=TurnPhase.SPYMASTER,
            guesser_attempts_remaining=None,
        )

    def _end(self, *, winner: Color | None = None) -> None:
        self.state = replace(
            self.state,
            current_phase=TurnPhase.DONE,
            winner=winner,
            guesser_attempts_remaining=None,
        )
