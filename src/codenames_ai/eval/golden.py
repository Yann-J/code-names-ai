from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from codenames_ai.agent.interfaces import Spymaster
from codenames_ai.game.models import Board, Color, SpymasterView

GoldenMatchMode = Literal["exact", "subset", "overlap"]


@dataclass(frozen=True)
class GoodClueOption:
    """A labeled set of friendly cards that should plausibly be targeted by some good clue."""

    target_subset: frozenset[str]
    label: str


@dataclass(frozen=True)
class GoldenBoard:
    """A hand-curated regression test for the spymaster's candidate-generation algorithm.

    The spymaster's chosen targets are checked against `options` via the
    configured match mode; whether the *clue word itself* is "good" is left to
    human judgment — automated regression checks the cluster recovery, not the
    LLM-style reasonableness of the surface form.
    """

    name: str
    board: Board
    team: Color
    options: tuple[GoodClueOption, ...]


@dataclass(frozen=True)
class GoldenResult:
    name: str
    matched: bool
    matched_label: str | None
    chosen_clue: str
    chosen_targets: tuple[str, ...]
    chosen_n: int


def evaluate_golden(
    spymaster: Spymaster,
    golden: GoldenBoard,
    *,
    mode: GoldenMatchMode = "subset",
) -> GoldenResult:
    """Run `spymaster` on `golden.board`; check chosen targets against `golden.options`.

    Match modes:
      - `exact`: chosen target set equals one option's target_subset.
      - `subset` (default): chosen targets are a subset of some option (all picked
        targets are in the same labeled cluster).
      - `overlap`: at least one chosen target is in some option.
    """
    view = SpymasterView(board=golden.board, team=golden.team)
    trace = spymaster.give_clue(view)
    if trace.chosen is None:
        return GoldenResult(
            name=golden.name,
            matched=False,
            matched_label=None,
            chosen_clue="",
            chosen_targets=(),
            chosen_n=0,
        )

    chosen_targets = frozenset(trace.chosen.targets)
    matched_label: str | None = None
    for opt in golden.options:
        if mode == "exact" and chosen_targets == opt.target_subset:
            matched_label = opt.label
            break
        if mode == "subset" and chosen_targets and chosen_targets <= opt.target_subset:
            matched_label = opt.label
            break
        if mode == "overlap" and chosen_targets & opt.target_subset:
            matched_label = opt.label
            break
    return GoldenResult(
        name=golden.name,
        matched=matched_label is not None,
        matched_label=matched_label,
        chosen_clue=trace.chosen.clue,
        chosen_targets=trace.chosen.targets,
        chosen_n=trace.chosen.n,
    )


def evaluate_goldens(
    spymaster: Spymaster,
    goldens: Sequence[GoldenBoard],
    *,
    mode: GoldenMatchMode = "subset",
) -> list[GoldenResult]:
    return [evaluate_golden(spymaster, g, mode=mode) for g in goldens]


def golden_pass_rate(results: Sequence[GoldenResult]) -> float:
    if not results:
        return 0.0
    return sum(1 for r in results if r.matched) / len(results)
