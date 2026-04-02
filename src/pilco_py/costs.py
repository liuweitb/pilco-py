from __future__ import annotations

import numpy as np
import torch


def wrap_angle_numpy(theta: np.ndarray | float) -> np.ndarray:
    theta_array = np.asarray(theta, dtype=np.float64)
    return np.arctan2(np.sin(theta_array), np.cos(theta_array))


def wrap_angle_torch(theta: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(theta), torch.cos(theta))


def pendulum_tip_position_numpy(theta: np.ndarray, length: float) -> np.ndarray:
    theta = np.asarray(theta, dtype=np.float64)
    return np.stack((length * np.sin(theta), -length * np.cos(theta)), axis=-1)


def pendulum_tip_position_torch(theta: torch.Tensor, length: float) -> torch.Tensor:
    return torch.stack((length * torch.sin(theta), -length * torch.cos(theta)), dim=-1)


def pendulum_cost_numpy(states: np.ndarray, length: float, width: float) -> np.ndarray:
    theta = np.asarray(states, dtype=np.float64)[..., 1]
    tip = pendulum_tip_position_numpy(theta, length)
    target_tip = np.array([0.0, length], dtype=np.float64)
    distance_sq = np.sum((tip - target_tip) ** 2, axis=-1)
    return 1.0 - np.exp(-0.5 * distance_sq / (width**2))


def pendulum_cost_torch(states: torch.Tensor, length: float, width: float) -> torch.Tensor:
    theta = states[..., 1]
    tip = pendulum_tip_position_torch(theta, length)
    target_tip = torch.tensor([0.0, length], dtype=states.dtype, device=states.device)
    distance_sq = torch.sum((tip - target_tip) ** 2, dim=-1)
    return 1.0 - torch.exp(-0.5 * distance_sq / (width**2))


def emg_effort_cost_numpy(
    states: np.ndarray,
    biceps_weight: float,
    triceps_weight: float,
    width: float,
) -> np.ndarray:
    states = np.asarray(states, dtype=np.float64)
    effort = biceps_weight * states[..., 2] ** 2 + triceps_weight * states[..., 3] ** 2
    return 1.0 - np.exp(-0.5 * effort / (width**2))


def emg_effort_cost_torch(
    states: torch.Tensor,
    biceps_weight: float,
    triceps_weight: float,
    width: float,
) -> torch.Tensor:
    effort = biceps_weight * states[..., 2] ** 2 + triceps_weight * states[..., 3] ** 2
    return 1.0 - torch.exp(-0.5 * effort / (width**2))
