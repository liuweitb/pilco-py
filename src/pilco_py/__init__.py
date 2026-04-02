"""Readable PILCO-style pendulum recreation built with NumPy, PyTorch, GPyTorch, and MuJoCo."""

from .config import ExperimentConfig
from .trainer import PILCOPendulumTrainer

__all__ = ["ExperimentConfig", "PILCOPendulumTrainer"]
