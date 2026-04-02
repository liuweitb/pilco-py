from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .config import ExperimentConfig
from .costs import pendulum_cost_torch, wrap_angle_numpy, wrap_angle_torch
from .env import PendulumPlant, Trajectory
from .gpytorch_dynamics import IndependentGPDynamicsModel
from .policy import SquashedMLPPolicy
from .visualization import (
    render_mujoco_rollout,
    save_rollout_diagnostics,
    save_tip_trajectory_plot,
    save_training_curve,
)


@dataclass(slots=True)
class RolloutPrediction:
    mean: np.ndarray
    std: np.ndarray
    expected_total_cost: float


class PILCOPendulumTrainer:
    """Readable end-to-end training loop inspired by the original PILCO pendulum example."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.output_dir = Path(config.training.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.environment = PendulumPlant(config.pendulum)
        self.rng = np.random.default_rng(config.pendulum.rollout_seed)
        torch.set_default_dtype(torch.float64)

        self.policy = SquashedMLPPolicy(
            input_dim=3,
            hidden_sizes=config.policy_optim.hidden_sizes,
            max_action=config.pendulum.max_torque,
        )

        self.history: list[dict[str, float]] = []
        self.trajectories: list[Trajectory] = []
        self._train_inputs = np.zeros((0, 4), dtype=np.float64)
        self._train_targets = np.zeros((0, 2), dtype=np.float64)

    def run(self) -> None:
        for rollout_index in range(self.config.training.initial_random_rollouts):
            trajectory = self.environment.rollout(self.rng, policy=None)
            self._append_trajectory(trajectory)
            self._save_random_rollout_plot(trajectory, rollout_index)

        for episode in range(1, self.config.training.policy_episodes + 1):
            model = self._fit_dynamics_model()
            self._optimize_policy(model)
            prediction = self.predict_rollout(model)
            trajectory = self._run_policy_rollout()
            self._append_trajectory(trajectory)
            self._save_episode_artifacts(episode, trajectory, prediction)

            history_row = {
                "episode": float(episode),
                "rollout_cost": trajectory.total_cost,
                "predicted_cost": prediction.expected_total_cost,
                "train_points": float(self._train_inputs.shape[0]),
            }
            self.history.append(history_row)

        save_training_curve(self.history, self.output_dir / "training_curve.png")
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
        optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=self.config.policy_optim.learning_rate,
        )

        for step in range(self.config.policy_optim.adam_steps):
            optimizer.zero_grad(set_to_none=True)
            loss = self._policy_loss(
                model=model,
                num_particles=self.config.policy_optim.num_particles,
                sample_model_uncertainty=self.config.policy_optim.use_model_uncertainty,
                seed=self.config.policy_optim.common_random_seed,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(),
                max_norm=self.config.policy_optim.gradient_clip_norm,
            )
            optimizer.step()

            if (step + 1) % 50 == 0:
                print(f"policy step {step + 1:03d}/{self.config.policy_optim.adam_steps}: loss={loss.item():.4f}")

    def predict_rollout(self, model: IndependentGPDynamicsModel) -> RolloutPrediction:
        mean, std, expected_total_cost = self._simulate_particles(
            model=model,
            num_particles=self.config.policy_optim.evaluation_particles,
            sample_model_uncertainty=True,
            seed=self.config.policy_optim.common_random_seed + 101,
        )
        return RolloutPrediction(mean=mean, std=std, expected_total_cost=expected_total_cost)

    def _run_policy_rollout(self) -> Trajectory:
        def policy_callback(features: np.ndarray) -> float:
            with torch.no_grad():
                feature_tensor = torch.as_tensor(features[None, :], dtype=torch.get_default_dtype())
                action = self.policy(feature_tensor).cpu().numpy()[0, 0]
            return float(action)

        return self.environment.rollout(self.rng, policy=policy_callback)

    def _policy_loss(
        self,
        model: IndependentGPDynamicsModel,
        num_particles: int,
        sample_model_uncertainty: bool,
        seed: int,
    ) -> torch.Tensor:
        _, _, expected_total_cost = self._simulate_particles(
            model=model,
            num_particles=num_particles,
            sample_model_uncertainty=sample_model_uncertainty,
            seed=seed,
            require_grad=True,
        )
        return expected_total_cost

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

        mean = torch.tensor(self.config.pendulum.initial_mean_array, dtype=torch.get_default_dtype())
        covariance = torch.tensor(self.config.pendulum.initial_covariance_array, dtype=torch.get_default_dtype())
        covariance_cholesky = torch.linalg.cholesky(covariance)
        base_samples = torch.randn(
            (num_particles, mean.shape[0]),
            generator=generator,
            dtype=mean.dtype,
        )
        particles = mean + base_samples @ covariance_cholesky.T
        particles = torch.stack((particles[:, 0], wrap_angle_torch(particles[:, 1])), dim=-1)
        if not require_grad:
            particles = particles.detach()

        predicted_means = [particles.mean(dim=0)]
        predicted_stds = [particles.std(dim=0, unbiased=False)]
        total_cost = torch.tensor(0.0, dtype=torch.get_default_dtype())

        for _ in range(self.config.pendulum.horizon_steps):
            features = torch.stack(
                (
                    particles[:, 0],
                    torch.sin(particles[:, 1]),
                    torch.cos(particles[:, 1]),
                ),
                dim=-1,
            )
            actions = self.policy(features).squeeze(-1)
            gp_inputs = torch.cat((features, actions[:, None]), dim=-1)
            delta_mean, delta_var = model.predict_torch(gp_inputs)

            if sample_model_uncertainty:
                noise = torch.randn(
                    delta_mean.shape,
                    generator=generator,
                    dtype=delta_mean.dtype,
                )
                delta = delta_mean + noise * torch.sqrt(delta_var + 1e-6)
            else:
                delta = delta_mean

            next_particles = particles + delta
            particles = torch.stack(
                (
                    next_particles[:, 0],
                    wrap_angle_torch(next_particles[:, 1]),
                ),
                dim=-1,
            )
            total_cost = total_cost + pendulum_cost_torch(
                particles,
                length=self.config.pendulum.length,
                width=self.config.pendulum.cost_width,
            ).mean()

            predicted_means.append(particles.mean(dim=0))
            predicted_stds.append(particles.std(dim=0, unbiased=False))

        mean_array = torch.stack(predicted_means).detach().cpu().numpy()
        std_array = torch.stack(predicted_stds).detach().cpu().numpy()

        if require_grad:
            return mean_array, std_array, total_cost
        return mean_array, std_array, float(total_cost.item())

    def _append_trajectory(self, trajectory: Trajectory) -> None:
        current = trajectory.observed_states[:-1]
        nxt = trajectory.observed_states[1:]
        features = self.environment.feature_batch(current)
        actions = trajectory.actions[:, None]
        deltas = nxt - current
        deltas[:, 1] = wrap_angle_numpy(deltas[:, 1])

        self._train_inputs = np.vstack((self._train_inputs, np.hstack((features, actions))))
        self._train_targets = np.vstack((self._train_targets, deltas))
        self.trajectories.append(trajectory)

    def _save_random_rollout_plot(self, trajectory: Trajectory, rollout_index: int) -> None:
        predicted_mean = trajectory.latent_states.copy()
        predicted_std = np.zeros_like(predicted_mean)
        save_rollout_diagnostics(
            trajectory=trajectory,
            predicted_mean=predicted_mean,
            predicted_std=predicted_std,
            config=self.config.pendulum,
            output_path=self.output_dir / f"random_rollout_{rollout_index:02d}.png",
        )

    def _save_episode_artifacts(
        self,
        episode: int,
        trajectory: Trajectory,
        prediction: RolloutPrediction,
    ) -> None:
        save_rollout_diagnostics(
            trajectory=trajectory,
            predicted_mean=prediction.mean,
            predicted_std=prediction.std,
            config=self.config.pendulum,
            output_path=self.output_dir / f"episode_{episode:02d}_diagnostics.png",
        )
        save_tip_trajectory_plot(
            trajectory=trajectory,
            predicted_mean=prediction.mean,
            predicted_std=prediction.std,
            config=self.config.pendulum,
            output_path=self.output_dir / f"episode_{episode:02d}_tip_path.png",
        )

        if self.config.training.save_video and self.config.training.render_every_episode:
            render_mujoco_rollout(
                trajectory=trajectory,
                config=self.config.pendulum,
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
