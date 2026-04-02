from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .assistive_env import AssistiveArmPlant, AssistiveTrajectory
from .assistive_visualization import (
    save_assistive_animation,
    save_assistive_rollout_diagnostics,
    save_assistive_training_curve,
    save_policy_and_interaction_maps,
)
from .config import AssistiveExperimentConfig
from .costs import emg_effort_cost_torch
from .gpytorch_dynamics import IndependentGPDynamicsModel
from .policy import PositiveMLPPolicy


@dataclass(slots=True)
class AssistiveRolloutPrediction:
    mean: np.ndarray
    std: np.ndarray
    expected_total_cost: float


class AssistiveStrategyTrainer:
    """Hamaya-style shared-control experiment built on top of the PILCO scaffold."""

    def __init__(self, config: AssistiveExperimentConfig) -> None:
        self.config = config
        self.output_dir = Path(config.training.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.environment = AssistiveArmPlant(config.assistive_arm)
        self.rng = np.random.default_rng(config.assistive_arm.rollout_seed)
        torch.set_default_dtype(torch.float64)

        self.policy = PositiveMLPPolicy(
            input_dim=4,
            hidden_sizes=config.policy_optim.hidden_sizes,
            max_action=config.assistive_arm.max_pressure,
        )

        self.history: list[dict[str, float]] = []
        self.trajectories: list[AssistiveTrajectory] = []
        self._train_inputs = np.zeros((0, 5), dtype=np.float64)
        self._train_targets = np.zeros((0, 4), dtype=np.float64)

    def run(self) -> None:
        for rollout_index in range(self.config.training.initial_random_rollouts):
            trajectory = self.environment.rollout(self.rng, policy=None)
            self._append_trajectory(trajectory)

        for episode in range(1, self.config.training.policy_episodes + 1):
            model = self._fit_dynamics_model()
            self._optimize_policy(model)
            prediction = self.predict_rollout(model)
            trajectory = self._run_policy_rollout()
            self._append_trajectory(trajectory)
            self._save_episode_artifacts(episode, trajectory, prediction, model)
            self.history.append(
                {
                    "episode": float(episode),
                    "rollout_cost": trajectory.total_cost,
                    "predicted_cost": prediction.expected_total_cost,
                    "train_points": float(self._train_inputs.shape[0]),
                }
            )

        save_assistive_training_curve(self.history, self.output_dir / "training_curve.png")
        self._write_history()

    def _fit_dynamics_model(self) -> IndependentGPDynamicsModel:
        model = IndependentGPDynamicsModel(jitter=self.config.dynamics.gp_jitter)
        model.fit(
            self._train_inputs,
            self._train_targets,
            training_steps=self.config.dynamics.gp_train_steps,
        )
        return model

    def _optimize_policy(self, model: IndependentGPDynamicsModel) -> None:
        optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.config.policy_optim.learning_rate)
        for step in range(self.config.policy_optim.adam_steps):
            optimizer.zero_grad(set_to_none=True)
            loss = self._policy_loss(
                model=model,
                num_particles=self.config.policy_optim.num_particles,
                sample_model_uncertainty=self.config.policy_optim.use_model_uncertainty,
                seed=self.config.policy_optim.common_random_seed,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.policy_optim.gradient_clip_norm)
            optimizer.step()
            if (step + 1) % 50 == 0:
                print(f"assistive policy step {step + 1:03d}/{self.config.policy_optim.adam_steps}: loss={loss.item():.4f}")

    def _policy_loss(
        self,
        model: IndependentGPDynamicsModel,
        num_particles: int,
        sample_model_uncertainty: bool,
        seed: int,
    ) -> torch.Tensor:
        _, _, total_cost = self._simulate_particles(
            model=model,
            num_particles=num_particles,
            sample_model_uncertainty=sample_model_uncertainty,
            seed=seed,
            require_grad=True,
        )
        return total_cost

    def predict_rollout(self, model: IndependentGPDynamicsModel) -> AssistiveRolloutPrediction:
        mean, std, total_cost = self._simulate_particles(
            model=model,
            num_particles=self.config.policy_optim.evaluation_particles,
            sample_model_uncertainty=True,
            seed=self.config.policy_optim.common_random_seed + 101,
        )
        return AssistiveRolloutPrediction(mean=mean, std=std, expected_total_cost=total_cost)

    def _run_policy_rollout(self) -> AssistiveTrajectory:
        def policy_callback(state: np.ndarray) -> float:
            with torch.no_grad():
                state_tensor = torch.as_tensor(state[None, :], dtype=torch.get_default_dtype())
                action = self.policy(state_tensor).cpu().numpy()[0, 0]
            return float(action)

        return self.environment.rollout(self.rng, policy=policy_callback)

    def _simulate_particles(
        self,
        model: IndependentGPDynamicsModel,
        num_particles: int,
        sample_model_uncertainty: bool,
        seed: int,
        require_grad: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, torch.Tensor] | tuple[np.ndarray, np.ndarray, float]:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)

        mean = torch.tensor(self.config.assistive_arm.initial_mean_array, dtype=torch.get_default_dtype())
        covariance = torch.tensor(self.config.assistive_arm.initial_covariance_array, dtype=torch.get_default_dtype())
        chol = torch.linalg.cholesky(covariance)
        particles = mean + torch.randn((num_particles, mean.shape[0]), generator=generator, dtype=mean.dtype) @ chol.T
        particles = torch.cat((particles[:, :2], torch.clamp(particles[:, 2:], 0.0, 1.0)), dim=-1)
        if not require_grad:
            particles = particles.detach()

        predicted_means = [particles.mean(dim=0)]
        predicted_stds = [particles.std(dim=0, unbiased=False)]
        total_cost = torch.tensor(0.0, dtype=torch.get_default_dtype())

        for _ in range(self.config.assistive_arm.horizon_steps):
            actions = self.policy(particles).squeeze(-1)
            model_inputs = torch.cat((particles, actions[:, None]), dim=-1)
            delta_mean, delta_var = model.predict_torch(model_inputs)

            if sample_model_uncertainty:
                delta = delta_mean + torch.randn(delta_mean.shape, generator=generator, dtype=delta_mean.dtype) * torch.sqrt(delta_var + 1e-6)
            else:
                delta = delta_mean

            next_particles = particles + delta
            particles = torch.cat((next_particles[:, :2], torch.clamp(next_particles[:, 2:], 0.0, 1.0)), dim=-1)
            total_cost = total_cost + emg_effort_cost_torch(
                particles,
                biceps_weight=self.config.assistive_arm.biceps_cost_weight,
                triceps_weight=self.config.assistive_arm.triceps_cost_weight,
                width=self.config.assistive_arm.cost_width,
            ).mean()

            predicted_means.append(particles.mean(dim=0))
            predicted_stds.append(particles.std(dim=0, unbiased=False))

        mean_array = torch.stack(predicted_means).detach().cpu().numpy()
        std_array = torch.stack(predicted_stds).detach().cpu().numpy()
        if require_grad:
            return mean_array, std_array, total_cost
        return mean_array, std_array, float(total_cost.item())

    def _append_trajectory(self, trajectory: AssistiveTrajectory) -> None:
        current = trajectory.observed_states[:-1]
        nxt = trajectory.observed_states[1:]
        actions = trajectory.actions[:, None]
        deltas = nxt - current
        self._train_inputs = np.vstack((self._train_inputs, np.hstack((current, actions))))
        self._train_targets = np.vstack((self._train_targets, deltas))
        self.trajectories.append(trajectory)

    def _save_episode_artifacts(
        self,
        episode: int,
        trajectory: AssistiveTrajectory,
        prediction: AssistiveRolloutPrediction,
        dynamics_model: IndependentGPDynamicsModel,
    ) -> None:
        save_assistive_rollout_diagnostics(
            trajectory=trajectory,
            predicted_mean=prediction.mean,
            predicted_std=prediction.std,
            config=self.config.assistive_arm,
            output_path=self.output_dir / f"episode_{episode:02d}_diagnostics.png",
        )
        save_policy_and_interaction_maps(
            plant=self.environment,
            policy=self.policy,
            dynamics_model=dynamics_model,
            trajectory=trajectory,
            output_path=self.output_dir / f"episode_{episode:02d}_policy_maps.png",
        )
        if self.config.training.save_video:
            save_assistive_animation(
                trajectory=trajectory,
                config=self.config.assistive_arm,
                output_path=self.output_dir / f"episode_{episode:02d}.gif",
            )

    def _write_history(self) -> None:
        serializable = [
            {
                "episode": int(entry["episode"]),
                "rollout_cost": float(entry["rollout_cost"]),
                "predicted_cost": float(entry["predicted_cost"]),
                "train_points": int(entry["train_points"]),
            }
            for entry in self.history
        ]
        with (self.output_dir / "history.json").open("w", encoding="utf-8") as handle:
            json.dump(serializable, handle, indent=2)
