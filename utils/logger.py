"""MLflow/DagsHub logging utilities for the firefighting RL project.

Provides MLflowLogger (a thin wrapper around the mlflow Python API) and
SB3MLflowCallback (a stable-baselines3 callback that ships episode metrics
to MLflow automatically during training).

Credentials are read exclusively via os.getenv() after load_dotenv() —
never hard-coded here.
"""
import os
from typing import Any, Dict, Optional

import mlflow
from dotenv import load_dotenv
from stable_baselines3.common.callbacks import BaseCallback


class MLflowLogger:
    """Context-manager-friendly wrapper around the MLflow tracking API.

    Reads DagsHub credentials from environment variables (loaded from .env
    via python-dotenv). All credential access is done through ``os.getenv``
    so no secrets are ever stored in source code.

    Usage::

        with MLflowLogger("Firefighting-RL", "reference_agent",
                          tags={"agent": "reference_agent"}) as logger:
            logger.log_params({"lr": 1e-4})
            for step in range(1000):
                logger.log_metrics({"reward": r}, step=step)
            logger.log_artifact("checkpoints/model.zip")

    Attributes:
        experiment_name: MLflow experiment name.
        run_name: Display name for this run.
        run: The active ``mlflow.ActiveRun`` object.
    """

    def __init__(
        self,
        experiment_name: str,
        run_name: str,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """Initialise and start an MLflow run.

        Args:
            experiment_name: Name of the MLflow experiment to log into.
            run_name: Human-readable name for this individual run.
            tags: Optional dict of string tags attached to the run.
        """
        load_dotenv()

        dagshub_username = os.getenv("DAGSHUB_USERNAME")
        dagshub_token = os.getenv("DAGSHUB_TOKEN")
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

        if dagshub_username:
            os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_username
        if dagshub_token:
            os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        mlflow.set_experiment(experiment_name)
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.run = mlflow.start_run(run_name=run_name, tags=tags or {})

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log a dict of hyperparameters to the active run.

        Args:
            params: Key-value pairs of parameter names and values.
        """
        mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """Log scalar metrics at a specific training step.

        Args:
            metrics: Dict mapping metric name to scalar value.
            step: Global training step used as the x-axis in MLflow UI.
        """
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, path: str) -> None:
        """Upload a local file or directory as a run artifact.

        Args:
            path: Local filesystem path to the file or directory.
        """
        mlflow.log_artifact(path)

    def end_run(self) -> None:
        """Close the active MLflow run."""
        mlflow.end_run()

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "MLflowLogger":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.end_run()


class SB3MLflowCallback(BaseCallback):
    """SB3 training callback that logs episode metrics to MLflow.

    At every completed episode (detected via the ``done`` flag in the info
    dict provided by the ``Monitor`` wrapper or SB3's internal episode
    buffer), the callback ships ``episode_reward`` and ``episode_length`` to
    the active MLflow run.

    Attributes:
        mlflow_logger: The ``MLflowLogger`` instance to write metrics to.
        log_freq: Minimum number of steps between consecutive MLflow writes
            (avoids excessive API calls for short episodes).
    """

    def __init__(self, mlflow_logger: MLflowLogger, log_freq: int = 1000) -> None:
        """Initialise the callback.

        Args:
            mlflow_logger: An already-started MLflowLogger instance.
            log_freq: How often (in steps) to emit metrics (default 1000).
        """
        super().__init__(verbose=0)
        self.mlflow_logger = mlflow_logger
        self.log_freq = log_freq
        self._episode_rewards: list = []
        self._episode_lengths: list = []

    def _on_step(self) -> bool:
        """Called after every environment step by SB3.

        Checks whether any episode ended this step and, if so, logs the
        episode reward and length to MLflow.

        Returns:
            True (always — returning False would stop training early).
        """
        # SB3 stores per-env episode info in self.locals["infos"]
        infos = self.locals.get("infos", [])
        for info in infos:
            # SB3 Monitor wrapper stores episode stats under "episode" key
            if "episode" in info:
                ep_reward = info["episode"]["r"]
                ep_length = info["episode"]["l"]
                self._episode_rewards.append(ep_reward)
                self._episode_lengths.append(ep_length)

        # Emit to MLflow at the requested frequency
        if self.num_timesteps % self.log_freq == 0 and self._episode_rewards:
            mean_reward = sum(self._episode_rewards) / len(self._episode_rewards)
            mean_length = sum(self._episode_lengths) / len(self._episode_lengths)
            self.mlflow_logger.log_metrics(
                {
                    "episode_reward": mean_reward,
                    "episode_length": mean_length,
                },
                step=self.num_timesteps,
            )
            self._episode_rewards.clear()
            self._episode_lengths.clear()

        return True
