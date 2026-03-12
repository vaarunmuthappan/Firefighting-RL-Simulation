"""Training callbacks for SB3-based agents.

Provides:
- CheckpointCallback: saves model checkpoints and optionally logs to W&B.
- EvalCallback: runs evaluation episodes and logs mean reward to W&B.
"""
import os
from typing import Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class CheckpointCallback(BaseCallback):
    """SB3 callback that saves model checkpoints every N steps.

    Checkpoints are stored in ``checkpoint_dir`` with filenames based on
    the current step count. If a ``wandb_logger`` is provided the artifact
    is also uploaded to the active W&B run.
    """

    def __init__(
        self,
        checkpoint_freq: int,
        checkpoint_dir: str,
        run_name: str = "agent",
        wandb_logger=None,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose=verbose)
        self.checkpoint_freq = checkpoint_freq
        self.checkpoint_dir = checkpoint_dir
        self.run_name = run_name
        self.wandb_logger = wandb_logger

    def _on_step(self) -> bool:
        if self.num_timesteps % self.checkpoint_freq == 0:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(
                self.checkpoint_dir,
                f"{self.run_name}_{self.num_timesteps}_steps",
            )
            self.model.save(checkpoint_path)

            if self.verbose >= 1:
                print(f"[CheckpointCallback] Saved checkpoint: {checkpoint_path}.zip")

            if self.wandb_logger is not None:
                try:
                    self.wandb_logger.log_artifact(
                        checkpoint_path + ".zip",
                        name=f"checkpoint-{self.num_timesteps}",
                        artifact_type="model",
                    )
                except Exception:
                    pass

        return True


class EvalCallback:
    """Simple evaluation runner that logs mean episode reward to W&B.

    Not an SB3 BaseCallback subclass — called explicitly from the training
    loop at any desired frequency.
    """

    def __init__(
        self,
        env,
        num_episodes: int = 5,
        wandb_logger=None,
        verbose: int = 1,
    ) -> None:
        self.env = env
        self.num_episodes = num_episodes
        self.wandb_logger = wandb_logger
        self.verbose = verbose

    def evaluate(self, model, step: int) -> float:
        """Run evaluation episodes and return mean episode reward."""
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

        if self.wandb_logger is not None:
            try:
                self.wandb_logger.log(
                    {"eval_mean_reward": mean_reward}, step=step
                )
            except Exception:
                pass

        return mean_reward
