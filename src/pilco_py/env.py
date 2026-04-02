from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .config import PendulumConfig
from .costs import pendulum_cost_numpy, wrap_angle_numpy


@dataclass(slots=True)
class Trajectory:
    observed_states: np.ndarray
    latent_states: np.ndarray
    actions: np.ndarray
    costs: np.ndarray

    @property
    def total_cost(self) -> float:
        return float(np.sum(self.costs))


class PendulumPlant:
    """Single-link pendulum with the same state convention as the original MATLAB code.

    State layout: `[angular_velocity, angle]`, where angle `0` is hanging down and
    `pi` is the upright target.
    """

    def __init__(self, config: PendulumConfig) -> None:
        self.config = config

    def sample_initial_state(self, rng: np.random.Generator) -> np.ndarray:
        state = rng.multivariate_normal(
            mean=self.config.initial_mean_array,
            cov=self.config.initial_covariance_array,
        )
        state[1] = wrap_angle_numpy(state[1])
        return state

    def observe(self, state: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        noise = rng.normal(scale=self.config.measurement_noise_array, size=2)
        observed = np.asarray(state, dtype=np.float64) + noise
        observed[1] = wrap_angle_numpy(observed[1])
        return observed

    def state_to_features(self, state: np.ndarray) -> np.ndarray:
        omega, theta = np.asarray(state, dtype=np.float64)
        return np.array([omega, np.sin(theta), np.cos(theta)], dtype=np.float64)

    def feature_batch(self, states: np.ndarray) -> np.ndarray:
        omega = states[:, 0]
        theta = states[:, 1]
        return np.column_stack((omega, np.sin(theta), np.cos(theta)))

    def dynamics_rhs(self, state: np.ndarray, torque: float) -> np.ndarray:
        omega, theta = state
        cfg = self.config
        omega_dot = (
            torque
            - cfg.damping * omega
            - cfg.mass * cfg.gravity * cfg.length * np.sin(theta) / 2.0
        ) / (cfg.mass * cfg.length**2 / 3.0)
        return np.array([omega_dot, omega], dtype=np.float64)

    def step(self, state: np.ndarray, torque: float) -> np.ndarray:
        dt = self.config.dt
        action = float(np.clip(torque, -self.config.max_torque, self.config.max_torque))
        k1 = self.dynamics_rhs(state, action)
        k2 = self.dynamics_rhs(state + 0.5 * dt * k1, action)
        k3 = self.dynamics_rhs(state + 0.5 * dt * k2, action)
        k4 = self.dynamics_rhs(state + dt * k3, action)
        next_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        next_state[1] = wrap_angle_numpy(next_state[1])
        return next_state

    def rollout(
        self,
        rng: np.random.Generator,
        policy: Callable[[np.ndarray], float] | None = None,
        start_state: np.ndarray | None = None,
        horizon_steps: int | None = None,
    ) -> Trajectory:
        horizon = horizon_steps or self.config.horizon_steps
        latent_states = np.zeros((horizon + 1, 2), dtype=np.float64)
        observed_states = np.zeros((horizon + 1, 2), dtype=np.float64)
        actions = np.zeros(horizon, dtype=np.float64)
        costs = np.zeros(horizon, dtype=np.float64)

        state = self.sample_initial_state(rng) if start_state is None else np.asarray(start_state, dtype=np.float64).copy()
        state[1] = wrap_angle_numpy(state[1])
        observation = self.observe(state, rng)
        latent_states[0] = state
        observed_states[0] = observation

        for step in range(horizon):
            if policy is None:
                action = rng.uniform(-self.config.max_torque, self.config.max_torque)
            else:
                action = float(policy(self.state_to_features(observation)))
            action = float(np.clip(action, -self.config.max_torque, self.config.max_torque))

            next_state = self.step(state, action)
            next_observation = self.observe(next_state, rng)

            latent_states[step + 1] = next_state
            observed_states[step + 1] = next_observation
            actions[step] = action
            costs[step] = pendulum_cost_numpy(
                next_state[None, :],
                length=self.config.length,
                width=self.config.cost_width,
            )[0]

            state = next_state
            observation = next_observation

        return Trajectory(
            observed_states=observed_states,
            latent_states=latent_states,
            actions=actions,
            costs=costs,
        )
