from codenames_ai.game.board import generate_board
from codenames_ai.game.models import (
    Board,
    Card,
    Clue,
    Color,
    GuesserView,
    SpymasterView,
)
from codenames_ai.game.rules import RuleStrictness, is_legal_clue
from codenames_ai.game.state import (
    GameState,
    TurnEvent,
    TurnPhase,
    check_win,
    reveal_card,
)

# Note: `Game` (the orchestrator) is intentionally NOT re-exported here. The
# orchestrator depends on agent abstractions and re-exporting it from this
# package creates a cycle (agent → game → orchestrator → agent.interfaces).
# Import it from `codenames_ai.game.orchestrator` or from the top-level
# `codenames_ai` package, both of which sit above the agent layer.

__all__ = [
    "Board",
    "Card",
    "Clue",
    "Color",
    "GameState",
    "GuesserView",
    "RuleStrictness",
    "SpymasterView",
    "TurnEvent",
    "TurnPhase",
    "check_win",
    "generate_board",
    "is_legal_clue",
    "reveal_card",
]
