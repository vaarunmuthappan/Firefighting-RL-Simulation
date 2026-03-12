"""Training callbacks for SB3-based agents.

Provides:
- CheckpointCallback: saves model checkpoints and logs artifact paths to MLflow.
- EvalCallback: runs evaluation episodes and logs mean reward to MLflow.
"""
import os
from typing import Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class CheckpointCallback(BaseCallback):
    """SB3 callback that saves model checkpoints every N steps.

    Checkpoints are stored in ``checkpoint_dir`` with filenames based on
    the current step count. If an ``MLflowLogger`` is provided the artifact
    path is also logged to the active MLflow run.

    Attributes:
        checkpoint_freq: Save a checkpoint every this many env steps.
        checkpoint_dir: Directory where checkpoint files are written.
        run_name: Prefix used in checkpoint filenames.
        mlflow_logger: Optional MLflowLogger to log artifact paths.
    """

    def __init__(
        self,
        checkpoint_freq: int,
        checkpoint_dir: str,
        run_name: str = "agent",
        mlflow_logger=None,
        verbose: int = 1,
    ) -> None:
        """Initialise CheckpointCallback.

        Args:
            checkpoint_freq: Number of env steps between checkpoints.
            checkpoint_dir: Directory path for saving checkpoint files.
            run_name: Prefix for checkpoint filenames (e.g. "reference_agent").
            mlflow_logger: Optional MLflowLogger instance. If provided, the
                checkpoint path is logged as an MLflow artifact.
            verbose: SB3 verbosity level (0=quiet, 1=info).
        """
        super().__init__(verbose=verbose)
        self.checkpoint_freq = checkpoint_freq
        self.checkpoint_dir = checkpoint_dir
        self.run_name = run_name
        self.mlflow_logger = mlflow_logger

    def _on_step(self) -> bool:
        """Save a checkpoint when the step counter is a multiple of checkpoint_freq.

        Returns:
            True (continue training).
        """
        if self.num_timesteps % self.checkpoint_freq == 0:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(
                self.checkpoint_dir,
                f"{self.run_name}_{self.num_timesteps}_steps",
            )
            self.model.save(checkpoint_path)

            if self.verbose >= 1:
                print(f"[CheckpointCallback] Saved checkpoint: {checkpoint_path}.zip")

            if self.mlflow_logger is not None:
                try:
                    self.mlflow_logger.log_artifact(checkpoint_path + ".zip")
                except Exception:
                    # Don't crash training if MLflow upload fails
                    pass

        return True


class EvalCallback:
    """Simple evaluation runner that logs mean episode reward to MLflow.

    This is intentionally not a SB3 BaseCallback subclass so it can be
    called explicitly from the training loop at any desired frequency
    without relying on SB3's callback machinery.

    Attributes:
        env: The gymnasium environment used for evaluation.
        num_episodes: Number of episodes to average over.
        mlflow_logger: Optional MLflowLogger for metric reporting.
        verbose: Verbosity level.
    """

    def __init__(
        self,
        env,
        num_episodes: int = 5,
        mlflow_logger=None,
        verbose: int = 1,
    ) -> None:
        """Initialise EvalCallback.

        Args:
            env: A gymnasium-compatible environment (not wrapped in Monitor).
            num_episodes: Episodes to run per evaluation call.
            mlflow_logger: Optional MLflowLogger instance.
            verbose: 0 = quiet, 1 = print mean reward.
        """
        self.env = env
        self.num_episodes = num_episodes
        self.mlflow_logger = mlflow_logger
        self.verbose = verbose

    def evaluate(self, model, step: int) -> float:
        """Run evaluation episodes and return mean episode reward.

        Args:
            model: Any agent with a ``predict(obs, deterministic=True)``
                method (i.e. an SB3 model or DQNAgent/PPOAgent).
            step: Current training step (used as the MLflow metric x-axis).

        Returns:
            Mean episode reward across ``num_episodes`` episodes.
        """
        episode_rewards = []

        for _ in range(self.num_episodes):
            obs, _ = self.env.reset()
            done = False
            ep_reward = 0.0

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = self.env.step(int(action))
                ep_reward += reward
                done = terminated or truncated

            episode_rewards.append(ep_reward)

        mean_reward = float(np.mean(episode_rewards))

        if self.verbose >= 1:
            print(
                f"[EvalCallback] step={step} "
                f"mean_reward={mean_reward:.3f} "
                f"over {self.num_episodes} episodes"
            )

        if self.mlflow_logger is not None:
            try:
                self.mlflow_logger.log_metrics(
                    {"eval_mean_reward": mean_reward}, step=step
                )
            except Exception:
                pass

        return mean_reward
