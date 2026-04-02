from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class PendulumConfig:
    dt: float = 0.1
    horizon_seconds: float = 4.0
    initial_state_mean: tuple[float, float] = (0.0, 0.0)
    initial_state_covariance: tuple[tuple[float, float], tuple[float, float]] = (
        (0.01, 0.0),
        (0.0, 0.01),
    )
    length: float = 1.0
    mass: float = 1.0
    gravity: float = 9.82
    damping: float = 0.01
    max_torque: float = 2.5
    measurement_noise_std: tuple[float, float] = (0.1, 0.01)
    target_theta: float = float(np.pi)
    cost_width: float = 0.5
    rollout_seed: int = 5

    @property
    def horizon_steps(self) -> int:
        return int(np.ceil(self.horizon_seconds / self.dt))

    @property
    def initial_mean_array(self) -> np.ndarray:
        return np.asarray(self.initial_state_mean, dtype=np.float64)

    @property
    def initial_covariance_array(self) -> np.ndarray:
        return np.asarray(self.initial_state_covariance, dtype=np.float64)

    @property
    def measurement_noise_array(self) -> np.ndarray:
        return np.asarray(self.measurement_noise_std, dtype=np.float64)


@dataclass(slots=True)
class DynamicsConfig:
    gp_train_steps: int = 150
    gp_jitter: float = 1e-6


@dataclass(slots=True)
class PolicyOptimizationConfig:
    hidden_sizes: tuple[int, int] = (64, 64)
    learning_rate: float = 3e-2
    adam_steps: int = 200
    num_particles: int = 128
    evaluation_particles: int = 512
    common_random_seed: int = 23
    use_model_uncertainty: bool = True
    gradient_clip_norm: float = 10.0


@dataclass(slots=True)
class TrainingConfig:
    initial_random_rollouts: int = 1
    policy_episodes: int = 8
    output_dir: Path = Path("artifacts/pendulum")
    render_every_episode: bool = True
    save_video: bool = True


@dataclass(slots=True)
class ExperimentConfig:
    pendulum: PendulumConfig = field(default_factory=PendulumConfig)
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    policy_optim: PolicyOptimizationConfig = field(default_factory=PolicyOptimizationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class AssistiveArmConfig:
    dt: float = 0.2
    horizon_seconds: float = 4.0
    initial_state_mean: tuple[float, float, float, float] = (1.3, 0.0, 0.18, 0.03)
    initial_state_covariance: tuple[
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        tuple[float, float, float, float],
    ] = (
        (0.02, 0.0, 0.0, 0.0),
        (0.0, 0.05, 0.0, 0.0),
        (0.0, 0.0, 0.01, 0.0),
        (0.0, 0.0, 0.0, 0.005),
    )
    measurement_noise_std: tuple[float, float, float, float] = (0.02, 0.04, 0.01, 0.01)
    reference_start: float = 1.3
    reference_goal: float = 2.5
    max_pressure: float = 0.65
    pressure_noise_std: float = 0.05
    inertia: float = 0.28
    damping: float = 0.55
    gravity_gain: float = 4.0
    load_torque: float = 1.8
    user_torque_gain: float = 11.0
    pressure_to_torque_gain: float = 10.0
    activation_time_constant: float = 0.25
    max_user_torque: float = 12.0
    intent_kp: float = 7.0
    intent_kd: float = 1.5
    biceps_cost_weight: float = 0.2
    triceps_cost_weight: float = 0.4
    cost_width: float = 0.5
    rollout_seed: int = 11

    @property
    def horizon_steps(self) -> int:
        return int(np.ceil(self.horizon_seconds / self.dt))

    @property
    def initial_mean_array(self) -> np.ndarray:
        return np.asarray(self.initial_state_mean, dtype=np.float64)

    @property
    def initial_covariance_array(self) -> np.ndarray:
        return np.asarray(self.initial_state_covariance, dtype=np.float64)

    @property
    def measurement_noise_array(self) -> np.ndarray:
        return np.asarray(self.measurement_noise_std, dtype=np.float64)


@dataclass(slots=True)
class AssistiveTrainingConfig:
    initial_random_rollouts: int = 5
    policy_episodes: int = 10
    output_dir: Path = Path("artifacts/assistive")
    save_video: bool = True


@dataclass(slots=True)
class AssistiveExperimentConfig:
    assistive_arm: AssistiveArmConfig = field(default_factory=AssistiveArmConfig)
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    policy_optim: PolicyOptimizationConfig = field(default_factory=PolicyOptimizationConfig)
    training: AssistiveTrainingConfig = field(default_factory=AssistiveTrainingConfig)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)
