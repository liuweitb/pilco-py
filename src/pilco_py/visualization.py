from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".mpl-cache").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(".cache").resolve()))

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np

from .config import PendulumConfig
from .costs import pendulum_cost_numpy, pendulum_tip_position_numpy
from .env import Trajectory


def save_training_curve(history: list[dict[str, float]], output_path: Path) -> None:
    if not history:
        return

    episodes = [entry["episode"] for entry in history]
    rollout_costs = [entry["rollout_cost"] for entry in history]
    predicted_costs = [entry["predicted_cost"] for entry in history]
    train_points = [entry["train_points"] for entry in history]

    figure, axes = plt.subplots(2, 1, figsize=(9, 8), constrained_layout=True)
    axes[0].plot(episodes, rollout_costs, marker="o", label="Observed rollout cost")
    axes[0].plot(episodes, predicted_costs, marker="s", label="Predicted cost")
    axes[0].set_xlabel("Policy episode")
    axes[0].set_ylabel("Total cost")
    axes[0].set_title("PILCO-style learning progress")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(episodes, train_points, marker="o", color="tab:purple")
    axes[1].set_xlabel("Policy episode")
    axes[1].set_ylabel("Transitions in GP dataset")
    axes[1].set_title("Dynamics model dataset growth")
    axes[1].grid(alpha=0.3)

    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_rollout_diagnostics(
    trajectory: Trajectory,
    predicted_mean: np.ndarray,
    predicted_std: np.ndarray,
    config: PendulumConfig,
    output_path: Path,
) -> None:
    timesteps = np.arange(trajectory.actions.shape[0] + 1) * config.dt
    rewards = 1.0 - trajectory.costs
    observed_theta = trajectory.latent_states[:, 1]
    predicted_theta = predicted_mean[:, 1]
    theta_std = predicted_std[:, 1]

    figure, axes = plt.subplots(3, 1, figsize=(10, 10), constrained_layout=True)

    axes[0].plot(timesteps, observed_theta, label="Observed angle")
    axes[0].plot(timesteps, predicted_theta, label="Predicted mean angle")
    axes[0].fill_between(
        timesteps,
        predicted_theta - 2.0 * theta_std,
        predicted_theta + 2.0 * theta_std,
        alpha=0.25,
        label="Predicted 2σ",
    )
    axes[0].axhline(np.pi, linestyle="--", color="black", linewidth=1.0, label="Upright target")
    axes[0].set_ylabel("Angle [rad]")
    axes[0].set_title("Predicted rollout versus realized pendulum trajectory")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].step(timesteps[:-1], trajectory.actions, where="post", color="tab:green")
    axes[1].set_ylabel("Torque [Nm]")
    axes[1].set_title("Applied control")
    axes[1].grid(alpha=0.3)

    axes[2].plot(timesteps[:-1], rewards, color="tab:orange")
    axes[2].set_xlabel("Time [s]")
    axes[2].set_ylabel("Immediate reward")
    axes[2].set_title("Reward profile along the rollout")
    axes[2].grid(alpha=0.3)

    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_tip_trajectory_plot(
    trajectory: Trajectory,
    predicted_mean: np.ndarray,
    predicted_std: np.ndarray,
    config: PendulumConfig,
    output_path: Path,
) -> None:
    observed_tip = pendulum_tip_position_numpy(trajectory.latent_states[:, 1], config.length)
    predicted_tip = pendulum_tip_position_numpy(predicted_mean[:, 1], config.length)
    target = np.array([0.0, config.length], dtype=np.float64)

    figure, axis = plt.subplots(figsize=(7, 7), constrained_layout=True)
    axis.plot(observed_tip[:, 0], observed_tip[:, 1], color="tab:red", linewidth=2, label="Observed tip path")
    axis.plot(predicted_tip[:, 0], predicted_tip[:, 1], color="tab:blue", linewidth=2, label="Predicted tip path")

    uncertainty_radius = 2.0 * config.length * np.clip(predicted_std[:, 1], 0.0, None)
    axis.scatter(
        predicted_tip[:, 0],
        predicted_tip[:, 1],
        s=np.maximum(20.0, 200.0 * uncertainty_radius),
        alpha=0.15,
        color="tab:blue",
        label="Predicted angle uncertainty",
    )
    axis.scatter([target[0]], [target[1]], marker="+", s=220, color="black", label="Target tip")
    axis.set_title("Pendulum tip path, matching the paper's geometric view")
    axis.set_xlabel("x [m]")
    axis.set_ylabel("z [m]")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.3)
    axis.legend()

    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _render_matplotlib_rollout(
    trajectory: Trajectory,
    config: PendulumConfig,
    output_path: Path,
) -> None:
    frames: list[np.ndarray] = []
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for step, state in enumerate(trajectory.latent_states):
        reward = 1.0
        if step > 0:
            reward = 1.0 - float(
                pendulum_cost_numpy(state[None, :], length=config.length, width=config.cost_width)[0]
            )

        figure, axis = plt.subplots(figsize=(6, 6))
        theta = float(state[1])
        tip = pendulum_tip_position_numpy(np.array([theta]), config.length)[0]
        axis.plot([0.0, tip[0]], [0.0, tip[1]], color="tab:red", linewidth=4)
        axis.scatter([0.0], [0.0], color="black", s=120)
        axis.scatter([tip[0]], [tip[1]], color="#f4c542", s=160)
        axis.scatter([0.0], [config.length], marker="+", color="black", s=220)
        axis.set_title(f"Fallback rollout render: step={step:02d}, reward={reward:.3f}")
        axis.set_xlim(-1.3 * config.length, 1.3 * config.length)
        axis.set_ylim(-1.3 * config.length, 1.3 * config.length)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.3)
        axis.set_xlabel("x [m]")
        axis.set_ylabel("z [m]")
        figure.canvas.draw()
        frame = np.asarray(figure.canvas.buffer_rgba())[..., :3].copy()
        frames.append(frame)
        plt.close(figure)

    imageio.mimsave(output_path, frames, fps=max(1, int(round(1.0 / config.dt))))


def render_mujoco_rollout(
    trajectory: Trajectory,
    config: PendulumConfig,
    output_path: Path,
) -> None:
    try:
        import mujoco
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "MuJoCo rendering requires `mujoco` and `imageio`. Install dependencies with `uv sync`."
        ) from exc

    xml = f"""
    <mujoco model="pilco_pendulum">
      <option timestep="{config.dt}" gravity="0 0 {-config.gravity}" integrator="RK4"/>
      <visual>
        <global offwidth="960" offheight="720"/>
        <rgba haze="1 1 1 1"/>
      </visual>
      <worldbody>
        <light pos="0 0 3"/>
        <geom type="plane" pos="0 0 -1.1" size="2 2 0.1" rgba="0.96 0.96 0.96 1"/>
        <body name="pivot" pos="0 0 0">
          <joint name="hinge" type="hinge" axis="0 1 0"/>
          <geom type="sphere" size="0.05" rgba="0.1 0.1 0.1 1"/>
          <geom type="capsule" fromto="0 0 0 0 0 {-config.length}" size="0.03" rgba="0.86 0.19 0.19 1"/>
          <site name="tip" pos="0 0 {-config.length}" size="0.02" rgba="0.95 0.8 0.2 1"/>
        </body>
        <site name="target" pos="0 0 {config.length}" size="0.04" rgba="0.1 0.1 0.9 1"/>
      </worldbody>
    </mujoco>
    """
    try:
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=720, width=960)

        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)
        camera.lookat[:] = np.array([0.0, 0.0, 0.0])
        camera.distance = 1.8
        camera.azimuth = 45.0
        camera.elevation = -20.0

        option = mujoco.MjvOption()
        mujoco.mjv_defaultOption(option)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        frames: list[np.ndarray] = []
        for state in trajectory.latent_states:
            data.qpos[0] = float(state[1])
            data.qvel[0] = float(state[0])
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera, scene_option=option)
            frames.append(renderer.render())

        imageio.mimsave(output_path, frames, fps=max(1, int(round(1.0 / config.dt))))
        renderer.close()
    except Exception:
        _render_matplotlib_rollout(trajectory=trajectory, config=config, output_path=output_path)
