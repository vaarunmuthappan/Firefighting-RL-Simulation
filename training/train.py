"""Main training orchestration for the firefighting RL agent.

Loads YAML configs, builds environment and agent, wires up MLflow logging,
runs training, evaluates the final model, and saves artefacts — all in one
``train()`` function that main_train.py calls.
"""
import os
from typing import Optional

import yaml

from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent
from environment.fire_env import FireEnv
from training.callbacks import CheckpointCallback, EvalCallback
from training.evaluate import evaluate_agent
from utils.logger import MLflowLogger, SB3MLflowCallback


def _load_yaml(path: str) -> dict:
    """Load a YAML file and return its contents as a dict."""
    with open(path, "r") as fh:
        return yaml.safe_load(fh) or {}


def train(
    config_dir: str = "config/",
    algorithm: Optional[str] = None,
    total_timesteps: Optional[int] = None,
) -> None:
    """End-to-end training pipeline for the firefighting RL agent.

    Steps
    -----
    1. Load env_config.yaml, agent_config.yaml, train_config.yaml.
    2. Setup MLflowLogger and tag the run with AGENT_NAME.
    3. Log all hyperparameters to MLflow.
    4. Build FireEnv from merged config.
    5. Build DQNAgent or PPOAgent based on train_config.
    6. Build SB3MLflowCallback and CheckpointCallback.
    7. Train for ``total_timesteps`` steps.
    8. Save the final model checkpoint.
    9. Run post-training evaluation.
    10. Log evaluation metrics and final model artifact to MLflow.
    11. End the MLflow run.

    Args:
        config_dir: Path to the directory containing the three YAML config
            files (default "config/").
        algorithm: Override the algorithm specified in train_config.yaml.
            Accepts "DQN" or "PPO".
        total_timesteps: Override total training timesteps from train_config.
    """
    # ------------------------------------------------------------------
    # 1. Load configs
    # ------------------------------------------------------------------
    env_cfg = _load_yaml(os.path.join(config_dir, "env_config.yaml"))
    agent_cfg = _load_yaml(os.path.join(config_dir, "agent_config.yaml"))
    train_cfg = _load_yaml(os.path.join(config_dir, "train_config.yaml"))

    # Apply CLI overrides
    if algorithm is not None:
        train_cfg["algorithm"] = algorithm
    if total_timesteps is not None:
        train_cfg["total_timesteps"] = total_timesteps

    # Merge env + agent configs into a single flat dict for FireEnv
    merged_env_cfg = {**env_cfg, **agent_cfg}

    algo_name: str = train_cfg.get("algorithm", "DQN").upper()
    run_name: str = train_cfg.get("run_name", "reference_agent")
    experiment_name: str = train_cfg.get("experiment_name", "Firefighting-RL")
    timesteps: int = int(train_cfg.get("total_timesteps", 50000))
    checkpoint_dir: str = train_cfg.get("checkpoint_dir", "checkpoints/")
    checkpoint_freq: int = int(train_cfg.get("checkpoint_freq", 10000))
    eval_episodes: int = int(train_cfg.get("eval_episodes", 5))

    # ------------------------------------------------------------------
    # 2. Setup MLflowLogger
    # ------------------------------------------------------------------
    tags = {
        "agent": "reference_agent",
        "algorithm": algo_name,
        "AGENT_NAME": "reference_agent",
    }
    logger = MLflowLogger(experiment_name, run_name, tags=tags)

    try:
        # ------------------------------------------------------------------
        # 3. Log hyperparameters
        # ------------------------------------------------------------------
        all_params = {
            **{f"env_{k}": v for k, v in env_cfg.items()},
            **{f"agent_{k}": v for k, v in agent_cfg.items()},
            **{f"train_{k}": v for k, v in train_cfg.items()},
        }
        # MLflow requires scalar param values; convert lists to strings
        serialisable = {
            k: str(v) if isinstance(v, (list, dict)) else v
            for k, v in all_params.items()
        }
        logger.log_params(serialisable)

        # ------------------------------------------------------------------
        # 4. Build FireEnv
        # ------------------------------------------------------------------
        env = FireEnv(merged_env_cfg)

        # ------------------------------------------------------------------
        # 5. Build agent
        # ------------------------------------------------------------------
        agent_model_cfg = {
            "policy": train_cfg.get("policy", "CnnPolicy"),
            "learning_rate": float(train_cfg.get("learning_rate", 1e-4)),
            "buffer_size": int(train_cfg.get("buffer_size", 50000)),
            "batch_size": int(train_cfg.get("batch_size", 32)),
            "gamma": float(train_cfg.get("gamma", 0.99)),
            "train_freq": int(train_cfg.get("train_freq", 4)),
            "target_update_interval": int(train_cfg.get("target_update_interval", 1000)),
            "exploration_fraction": float(train_cfg.get("exploration_fraction", 0.3)),
            "exploration_final_eps": float(train_cfg.get("exploration_final_eps", 0.05)),
            "verbose": int(train_cfg.get("verbose", 1)),
        }

        if algo_name == "DQN":
            agent = DQNAgent(env, agent_model_cfg)
        elif algo_name == "PPO":
            agent = PPOAgent(env, agent_model_cfg)
        else:
            raise ValueError(
                f"Unknown algorithm '{algo_name}'. Expected 'DQN' or 'PPO'."
            )

        # ------------------------------------------------------------------
        # 6. Build callbacks
        # ------------------------------------------------------------------
        mlflow_cb = SB3MLflowCallback(logger, log_freq=1000)
        checkpoint_cb = CheckpointCallback(
            checkpoint_freq=checkpoint_freq,
            checkpoint_dir=checkpoint_dir,
            run_name=run_name,
            mlflow_logger=logger,
        )

        # ------------------------------------------------------------------
        # 7. Train
        # ------------------------------------------------------------------
        print(f"[train] Starting {algo_name} training for {timesteps} timesteps …")
        agent.train(
            total_timesteps=timesteps,
            callbacks=[mlflow_cb, checkpoint_cb],
        )

        # ------------------------------------------------------------------
        # 8. Save final model
        # ------------------------------------------------------------------
        os.makedirs(checkpoint_dir, exist_ok=True)
        final_path = os.path.join(checkpoint_dir, f"{run_name}_{algo_name.lower()}")
        agent.save(final_path)
        print(f"[train] Final model saved to {final_path}.zip")

        # ------------------------------------------------------------------
        # 9. Evaluate
        # ------------------------------------------------------------------
        print(f"[train] Running evaluation over {eval_episodes} episodes …")
        eval_env = FireEnv(merged_env_cfg)
        eval_results = evaluate_agent(agent, eval_env, num_episodes=eval_episodes)
        print(
            f"[train] Eval results: "
            f"mean_reward={eval_results['mean_reward']:.3f} "
            f"std={eval_results['std_reward']:.3f} "
            f"mean_length={eval_results['mean_length']:.1f}"
        )

        # ------------------------------------------------------------------
        # 10. Log eval metrics + artefact
        # ------------------------------------------------------------------
        logger.log_metrics(
            {
                "final_mean_reward": eval_results["mean_reward"],
                "final_std_reward": eval_results["std_reward"],
                "final_mean_length": eval_results["mean_length"],
            },
            step=timesteps,
        )
        if os.path.isfile(final_path + ".zip"):
            logger.log_artifact(final_path + ".zip")

    finally:
        # ------------------------------------------------------------------
        # 11. End MLflow run (always, even on exception)
        # ------------------------------------------------------------------
        logger.end_run()
        print("[train] MLflow run closed.")
