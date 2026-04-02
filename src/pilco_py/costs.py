from __future__ import annotations

import numpy as np
import torch


def emg_effort_cost_numpy(
    states: np.ndarray,
    biceps_weight: float,
    triceps_weight: float,
    width: float,
) -> np.ndarray:
    """Hamaya-style saturating cost over EMG effort only."""

    states = np.asarray(states, dtype=np.float64)
    effort = biceps_weight * states[..., 2] ** 2 + triceps_weight * states[..., 3] ** 2
    return 1.0 - np.exp(-0.5 * effort / (width**2))


def emg_effort_cost_torch(
    states: torch.Tensor,
    biceps_weight: float,
    triceps_weight: float,
    width: float,
) -> torch.Tensor:
    """Torch version of the EMG-only cost used during policy optimization."""

    effort = biceps_weight * states[..., 2] ** 2 + triceps_weight * states[..., 3] ** 2
    return 1.0 - torch.exp(-0.5 * effort / (width**2))
