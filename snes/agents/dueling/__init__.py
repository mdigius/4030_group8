"""Dueling DQN agent package."""

from .network import DuelingQNetwork
from .trainer import main

__all__ = ["DuelingQNetwork", "main"]
