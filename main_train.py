"""Main entry point for firefighting RL training. Agent: reference_agent

Usage
-----
    python main_train.py
    python main_train.py --config-dir config/
    python main_train.py --timesteps 100000

All W&B runs are tagged with AGENT_NAME = "reference_agent".
"""
AGENT_NAME = "reference_agent"

import argparse
import os
import sys

# Ensure the project root is on sys.path when this script is run directly.
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from training.train import train


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Train a firefighting RL agent ({AGENT_NAME})."
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="config/",
        help="Directory containing env_config.yaml, agent_config.yaml, "
             "and train_config.yaml (default: config/).",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default=None,
        choices=["DQN", "PPO"],
        help="Override the algorithm specified in train_config.yaml.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Override total_timesteps from train_config.yaml.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a checkpoint .zip to resume training from.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    print(f"[main_train] AGENT_NAME = {AGENT_NAME}")
    print(f"[main_train] config_dir = {args.config_dir}")
    print(f"[main_train] algorithm  = {args.algorithm or '(from config)'}")
    print(f"[main_train] timesteps  = {args.timesteps or '(from config)'}")

    train(
        config_dir=args.config_dir,
        algorithm=args.algorithm,
        total_timesteps=args.timesteps,
        resume_from=args.resume,
    )


if __name__ == "__main__":
    main()
