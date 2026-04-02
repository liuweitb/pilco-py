from __future__ import annotations

import argparse
from pathlib import Path

from .assistive_trainer import AssistiveStrategyTrainer
from .config import AssistiveExperimentConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the Hamaya-style assistive arm controller.")
    parser.add_argument("--episodes", type=int, default=8, help="Number of policy-improvement episodes.")
    parser.add_argument("--initial-rollouts", type=int, default=1, help="Random rollouts before policy learning.")
    parser.add_argument("--policy-steps", type=int, default=200, help="Adam steps per policy update.")
    parser.add_argument("--particles", type=int, default=128, help="Particles for policy optimization.")
    parser.add_argument("--eval-particles", type=int, default=512, help="Particles for prediction plots.")
    parser.add_argument("--gp-train-steps", type=int, default=150, help="GPyTorch optimizer steps per output GP.")
    parser.add_argument("--seed", type=int, default=5, help="Base random seed for rollouts.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Artifact directory.")
    parser.add_argument("--skip-video", action="store_true", help="Disable rollout GIF rendering.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

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
