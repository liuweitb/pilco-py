from __future__ import annotations

import argparse
from pathlib import Path

from .assistive_trainer import AssistiveStrategyTrainer
from .config import AssistiveExperimentConfig, ExperimentConfig
from .trainer import PILCOPendulumTrainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train readable PILCO-style experiments.")
    parser.add_argument(
        "--experiment",
        choices=("pendulum", "assistive"),
        default="pendulum",
        help="Which PILCO-style experiment to run.",
    )
    parser.add_argument("--episodes", type=int, default=8, help="Number of policy-improvement episodes.")
    parser.add_argument("--initial-rollouts", type=int, default=1, help="Random rollouts before policy learning.")
    parser.add_argument("--policy-steps", type=int, default=200, help="Adam steps per policy update.")
    parser.add_argument("--particles", type=int, default=128, help="Particles for policy optimization.")
    parser.add_argument("--eval-particles", type=int, default=512, help="Particles for prediction plots.")
    parser.add_argument("--gp-train-steps", type=int, default=150, help="GPyTorch optimizer steps per output GP.")
    parser.add_argument("--seed", type=int, default=5, help="Base random seed for rollouts.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Artifact directory.")
    parser.add_argument("--skip-video", action="store_true", help="Disable MuJoCo GIF rendering.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.experiment == "pendulum":
        config = ExperimentConfig()
        config.training.policy_episodes = args.episodes
        config.training.initial_random_rollouts = args.initial_rollouts
        if args.output_dir is not None:
            config.training.output_dir = args.output_dir
        config.training.save_video = not args.skip_video
        config.pendulum.rollout_seed = args.seed
        config.policy_optim.adam_steps = args.policy_steps
        config.policy_optim.num_particles = args.particles
        config.policy_optim.evaluation_particles = args.eval_particles
        config.dynamics.gp_train_steps = args.gp_train_steps

        trainer = PILCOPendulumTrainer(config)
        trainer.run()
        print(f"artifacts written to {config.training.output_dir}")
        return

    config = AssistiveExperimentConfig()
    config.training.policy_episodes = args.episodes
    config.training.initial_random_rollouts = args.initial_rollouts
    if args.output_dir is not None:
        config.training.output_dir = args.output_dir
    config.training.save_video = not args.skip_video
    config.assistive_arm.rollout_seed = args.seed
    config.policy_optim.adam_steps = args.policy_steps
    config.policy_optim.num_particles = args.particles
    config.policy_optim.evaluation_particles = args.eval_particles
    config.dynamics.gp_train_steps = args.gp_train_steps

    trainer = AssistiveStrategyTrainer(config)
    trainer.run()
    print(f"artifacts written to {config.training.output_dir}")
