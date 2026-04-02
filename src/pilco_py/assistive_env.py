'''
Simulator
'''


from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .config import AssistiveArmConfig
from .costs import emg_effort_cost_numpy


@dataclass(slots=True)
class AssistiveTrajectory:
    observed_states: np.ndarray
    latent_states: np.ndarray
    actions: np.ndarray
    costs: np.ndarray
    reference_states: np.ndarray

    @property
    def total_cost(self) -> float:
        return float(np.sum(self.costs))


class AssistiveArmPlant:
    """Simple 1-DoF shared-control arm that mimics the Hamaya-style augmented state."""

    def __init__(self, config: AssistiveArmConfig) -> None:
        self.config = config

    def sample_initial_state(self, rng: np.random.Generator) -> np.ndarray:
        state = rng.multivariate_normal(
            mean=self.config.initial_mean_array,
            cov=self.config.initial_covariance_array,
        )
        state[2:] = np.clip(state[2:], 0.0, 1.0)
        return state

    def observe(self, state: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        observed = np.asarray(state, dtype=np.float64) + rng.normal(
            scale=self.config.measurement_noise_array,
            size=4,
        )
        observed[2:] = np.clip(observed[2:], 0.0, 1.0)
        return observed

    def reference_state(self, step: int) -> np.ndarray:
        t = step * self.config.dt
        T = self.config.horizon_seconds
        delta = self.config.reference_goal - self.config.reference_start
        phase = np.clip(t / T, 0.0, 1.0)
        q = self.config.reference_start + delta * 0.5 * (1.0 - np.cos(np.pi * phase))
        qdot = delta * 0.5 * np.pi / T * np.sin(np.pi * phase)
        qddot = delta * 0.5 * (np.pi / T) ** 2 * np.cos(np.pi * phase)
        return np.array([q, qdot, qddot], dtype=np.float64)

    def nominal_velocity_for_angle(self, angles: np.ndarray) -> np.ndarray:
        references = np.array([self.reference_state(step) for step in range(self.config.horizon_steps + 1)])
        reference_angles = references[:, 0]
        reference_velocities = references[:, 1]
        nominal = np.interp(angles, reference_angles, reference_velocities)
        return nominal

    def random_pressure_mean(self, step: int) -> float:
        t = step * self.config.dt
        return self.config.max_pressure * abs(np.sin(0.5 * np.pi * t))

    def _desired_user_torque(self, q: float, qdot: float, step: int) -> float:
        '''
        Simulated user intent. 
        Given the current robot state, return torque the user want to be applied.
        '''
        
        q_ref, qdot_ref, qddot_ref = self.reference_state(step)
        gravity = self.config.gravity_gain * np.sin(q)
        feedforward = self.config.inertia * qddot_ref + gravity + self.config.load_torque
        correction = self.config.intent_kp * (q_ref - q) + self.config.intent_kd * (qdot_ref - qdot)
        return feedforward + correction

    def step(self, state: np.ndarray, pressure: float, step_index: int) -> np.ndarray:
        q, qdot, eb, et = state
        pressure = float(np.clip(pressure, 0.0, self.config.max_pressure))
        assist_torque = self.config.pressure_to_torque_gain * pressure
        desired_user_torque = self._desired_user_torque(q, qdot, step_index)
        residual_torque = desired_user_torque - assist_torque

        biceps_target = np.clip(np.maximum(residual_torque, 0.0) / self.config.max_user_torque, 0.0, 1.0)
        triceps_target = np.clip(np.maximum(-residual_torque, 0.0) / self.config.max_user_torque, 0.0, 1.0)

        alpha = self.config.dt / self.config.activation_time_constant
        eb_next = np.clip(eb + alpha * (biceps_target - eb), 0.0, 1.0)
        et_next = np.clip(et + alpha * (triceps_target - et), 0.0, 1.0)

        user_torque = self.config.user_torque_gain * (eb_next - et_next)
        gravity = self.config.gravity_gain * np.sin(q)
        qddot = (
            user_torque
            + assist_torque
            - gravity
            - self.config.load_torque
            - self.config.damping * qdot
        ) / self.config.inertia

        qdot_next = qdot + self.config.dt * qddot
        q_next = q + self.config.dt * qdot_next
        return np.array([q_next, qdot_next, eb_next, et_next], dtype=np.float64)

    def rollout(
        self,
        rng: np.random.Generator,
        policy: Callable[[np.ndarray], float] | None = None,
        start_state: np.ndarray | None = None,
        horizon_steps: int | None = None,
    ) -> AssistiveTrajectory:
        horizon = horizon_steps or self.config.horizon_steps
        latent_states = np.zeros((horizon + 1, 4), dtype=np.float64)
        observed_states = np.zeros((horizon + 1, 4), dtype=np.float64)
        reference_states = np.zeros((horizon + 1, 3), dtype=np.float64)
        actions = np.zeros(horizon, dtype=np.float64)
        costs = np.zeros(horizon, dtype=np.float64)

        state = self.sample_initial_state(rng) if start_state is None else np.asarray(start_state, dtype=np.float64).copy()
        observation = self.observe(state, rng)
        latent_states[0] = state
        observed_states[0] = observation
        reference_states[0] = self.reference_state(0)

        for step in range(horizon):
            if policy is None:
                action = rng.normal(self.random_pressure_mean(step), self.config.pressure_noise_std)
            else:
                action = float(policy(observation))
            action = float(np.clip(action, 0.0, self.config.max_pressure))

            next_state = self.step(state, action, step)
            next_observation = self.observe(next_state, rng)

            latent_states[step + 1] = next_state
            observed_states[step + 1] = next_observation
            reference_states[step + 1] = self.reference_state(step + 1)
            actions[step] = action
            costs[step] = emg_effort_cost_numpy(
                next_state[None, :],
                biceps_weight=self.config.biceps_cost_weight,
                triceps_weight=self.config.triceps_cost_weight,
                width=self.config.cost_width,
            )[0]

            state = next_state
            observation = next_observation

        return AssistiveTrajectory(
            observed_states=observed_states,
            latent_states=latent_states,
            actions=actions,
            costs=costs,
            reference_states=reference_states,
        )
