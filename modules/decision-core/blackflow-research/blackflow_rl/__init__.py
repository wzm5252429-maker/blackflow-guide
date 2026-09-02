"""Blackflow route-planning research toolkit.

The core simulator and MCTS deliberately do not import PyTorch.  Neural-network
support lives in :mod:`blackflow_rl.network` and is loaded only by training
commands, so map generation and rule validation remain usable in lightweight
environments.
"""

from .domain import Action, ActionKind, GameState, NodeType, Transition
from .mapgen import MapGenerator, UnverifiedRuleError
from .rules import Ruleset, load_ruleset
from .simulator import BlackflowSimulator

__all__ = [
    "Action",
    "ActionKind",
    "BlackflowSimulator",
    "GameState",
    "MapGenerator",
    "NodeType",
    "Ruleset",
    "Transition",
    "UnverifiedRuleError",
    "load_ruleset",
]

__version__ = "0.2.0"
