from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".mpl-cache").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(".cache").resolve()))

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch

from .assistive_env import AssistiveArmPlant, AssistiveTrajectory
from .config import AssistiveArmConfig
from .gpytorch_dynamics import IndependentGPDynamicsModel
from .policy import PositiveMLPPolicy


def save_assistive_training_curve(history: list[dict[str, float]], output_path: Path) -> None:
    if not history:
        return

    trials = [entry["episode"] for entry in history]
    observed = [entry["rollout_cost"] for entry in history]
    predicted = [entry["predicted_cost"] for entry in history]

    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    axis.plot(trials, observed, marker="o", label="Observed long-term EMG cost")
    axis.plot(trials, predicted, marker="s", label="Predicted long-term EMG cost")
    axis.set_xlabel("Learning trial")
    axis.set_ylabel("Accumulated cost")
    axis.set_title("Hamaya-style assistive strategy learning curve")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_assistive_rollout_diagnostics(
    trajectory: AssistiveTrajectory,
    predicted_mean: np.ndarray,
    predicted_std: np.ndarray,
    config: AssistiveArmConfig,
    output_path: Path,
) -> None:
    time = np.arange(trajectory.actions.shape[0] + 1) * config.dt

    # This figure mirrors the paper's rollout plots: tracking, assistance, EMG,
    # and reward all on one page.
    figure, axes = plt.subplots(4, 1, figsize=(10, 12), constrained_layout=True)
    axes[0].plot(time, trajectory.reference_states[:, 0], linestyle="--", color="black", label="Reference angle")
    axes[0].plot(time, trajectory.latent_states[:, 0], color="tab:red", label="Observed angle")
    axes[0].plot(time, predicted_mean[:, 0], color="tab:blue", label="Predicted mean angle")
    axes[0].fill_between(
        time,
        predicted_mean[:, 0] - 2.0 * predicted_std[:, 0],
        predicted_mean[:, 0] + 2.0 * predicted_std[:, 0],
        alpha=0.2,
        color="tab:blue",
        label="Predicted 2σ",
    )
    axes[0].set_ylabel("Angle [rad]")
    axes[0].set_title("Assistive rollout: reference tracking, pressure, and EMGs")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].step(time[:-1], trajectory.actions, where="post", color="tab:green")
    axes[1].set_ylabel("Pressure [MPa]")
    axes[1].set_title("Learned assistive input")
    axes[1].grid(alpha=0.3)

    axes[2].plot(time, trajectory.latent_states[:, 2], color="tab:orange", label="Biceps EMG")
    axes[2].plot(time, trajectory.latent_states[:, 3], color="tab:purple", label="Triceps EMG")
    axes[2].set_ylabel("Activation")
    axes[2].set_title("Observed muscle activity")
    axes[2].grid(alpha=0.3)
    axes[2].legend()

    axes[3].plot(time[:-1], 1.0 - trajectory.costs, color="tab:brown")
    axes[3].set_xlabel("Time [s]")
    axes[3].set_ylabel("Reward")
    axes[3].set_title("Immediate reward derived from EMG-only cost")
    axes[3].grid(alpha=0.3)

    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_policy_and_interaction_maps(
    plant: AssistiveArmPlant,
    policy: PositiveMLPPolicy,
    dynamics_model: IndependentGPDynamicsModel,
    trajectory: AssistiveTrajectory,
    output_path: Path,
) -> None:
    config = plant.config
    angle_grid = np.linspace(config.reference_start, config.reference_goal, 80)
    biceps_grid = np.linspace(0.0, 0.45, 80)
    pressure_grid = np.linspace(0.0, config.max_pressure, 80)

    qdot_nominal = plant.nominal_velocity_for_angle(angle_grid)
    mean_eb = float(np.mean(trajectory.latent_states[:, 2]))
    mean_et = float(np.mean(trajectory.latent_states[:, 3]))

    # Policy map: how much assistance is produced as angle and biceps effort change.
    angle_mesh, biceps_mesh = np.meshgrid(angle_grid, biceps_grid)
    policy_inputs = np.column_stack(
        (
            angle_mesh.ravel(),
            np.interp(angle_mesh.ravel(), angle_grid, qdot_nominal),
            biceps_mesh.ravel(),
            np.full(angle_mesh.size, mean_et),
        )
    )
    with torch.no_grad():
        pressure_map = (
            policy(torch.as_tensor(policy_inputs, dtype=torch.float64))
            .cpu()
            .numpy()
            .reshape(angle_mesh.shape)
        )

    # Interaction map: what the learned GP predicts for one-step EMG changes
    # under different arm angles and robot pressures.
    angle_pressure_mesh, pressure_mesh = np.meshgrid(angle_grid, pressure_grid)
    interaction_inputs = np.column_stack(
        (
            angle_pressure_mesh.ravel(),
            np.interp(angle_pressure_mesh.ravel(), angle_grid, qdot_nominal),
            np.full(angle_pressure_mesh.size, mean_eb),
            np.full(angle_pressure_mesh.size, mean_et),
            pressure_mesh.ravel(),
        )
    )
    delta_mean, _ = dynamics_model.predict_numpy(interaction_inputs)
    delta_eb = delta_mean[:, 2].reshape(angle_pressure_mesh.shape)
    delta_et = delta_mean[:, 3].reshape(angle_pressure_mesh.shape)

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)

    map0 = axes[0].contourf(angle_mesh, biceps_mesh, pressure_map, levels=25, cmap="viridis")
    axes[0].set_title("Learned assistive strategy")
    axes[0].set_xlabel("Angle q [rad]")
    axes[0].set_ylabel("Biceps EMG")
    figure.colorbar(map0, ax=axes[0], label="Pressure [MPa]")

    map1 = axes[1].contourf(angle_pressure_mesh, pressure_mesh, delta_eb, levels=25, cmap="coolwarm")
    axes[1].set_title("Predicted Δ biceps EMG")
    axes[1].set_xlabel("Angle q [rad]")
    axes[1].set_ylabel("Pressure [MPa]")
    figure.colorbar(map1, ax=axes[1], label="ΔEb")

    map2 = axes[2].contourf(angle_pressure_mesh, pressure_mesh, delta_et, levels=25, cmap="coolwarm")
    axes[2].set_title("Predicted Δ triceps EMG")
    axes[2].set_xlabel("Angle q [rad]")
    axes[2].set_ylabel("Pressure [MPa]")
    figure.colorbar(map2, ax=axes[2], label="ΔEt")

    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_assistive_animation(
    trajectory: AssistiveTrajectory,
    config: AssistiveArmConfig,
    output_path: Path,
) -> None:
    frames: list[np.ndarray] = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arm_length = 0.4

    for step, state in enumerate(trajectory.latent_states):
        q = state[0]
        q_ref = trajectory.reference_states[step, 0]
        tip = np.array([arm_length * np.cos(q), arm_length * np.sin(q)])
        ref_tip = np.array([arm_length * np.cos(q_ref), arm_length * np.sin(q_ref)])

        figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

        axes[0].plot([0.0, tip[0]], [0.0, tip[1]], color="tab:red", linewidth=4, label="Measured arm")
        axes[0].plot([0.0, ref_tip[0]], [0.0, ref_tip[1]], color="black", linestyle="--", linewidth=2, label="Reference")
        axes[0].scatter([0.0], [0.0], color="black", s=100)
        axes[0].scatter([tip[0]], [tip[1]], color="#f4c542", s=140)
        axes[0].set_xlim(-0.1, 0.45)
        axes[0].set_ylim(0.0, 0.45)
        axes[0].set_aspect("equal", adjustable="box")
        axes[0].set_title(f"Assistive rollout frame {step:02d}")
        axes[0].grid(alpha=0.3)
        axes[0].legend(loc="lower right")

        labels = ["Pressure", "Biceps", "Triceps"]
        values = [
            # Normalize pressure so all three shared-control signals fit on the
            # same 0..1 horizontal bar chart.
            trajectory.actions[min(step, len(trajectory.actions) - 1)] / config.max_pressure if step < len(trajectory.actions) else 0.0,
            state[2],
            state[3],
        ]
        colors = ["tab:green", "tab:orange", "tab:purple"]
        axes[1].barh(labels, values, color=colors)
        axes[1].set_xlim(0.0, 1.0)
        axes[1].set_title("Shared-control signals")
        axes[1].grid(alpha=0.3, axis="x")

        figure.canvas.draw()
        frame = np.asarray(figure.canvas.buffer_rgba())[..., :3].copy()
        frames.append(frame)
        plt.close(figure)

    imageio.mimsave(output_path, frames, fps=max(1, int(round(1.0 / config.dt))))
