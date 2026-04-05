"""Weights & Biases logging utilities for the firefighting RL project.

Provides WandbLogger (a thin wrapper around the wandb Python API) and
SB3WandbCallback (a stable-baselines3 callback that ships episode metrics
and periodic GIF recordings to W&B during training).

Credentials are read exclusively via os.getenv() after load_dotenv() —
never hard-coded here.
"""
import glob
import os
import shutil
import tempfile
from typing import Any, Dict, List, Optional

import numpy as np
import wandb
from dotenv import load_dotenv
from stable_baselines3.common.callbacks import BaseCallback


class WandbLogger:
    """Context-manager-friendly wrapper around the W&B tracking API.

    Reads W&B credentials from environment variables (loaded from .env
    via python-dotenv). All credential access is done through ``os.getenv``
    so no secrets are ever stored in source code.

    Usage::

        with WandbLogger("Firefighting RL", "reference_agent",
                         config={"lr": 1e-4}) as logger:
            logger.log({"reward": r}, step=step)
            logger.log_gif("/path/to/episode.gif", step=step)
    """

    def __init__(
        self,
        project: str,
        run_name: str,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> None:
        load_dotenv()

        api_key = os.getenv("WANDB_API_KEY")
        entity = os.getenv("WANDB_ENTITY")

        if api_key:
            wandb.login(key=api_key)

        self.run = wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            config=config or {},
            tags=tags or [],
            reinit=True,
        )

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def log(self, data: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log scalar metrics or media to the active run."""
        wandb.log(data, step=step)

    def log_config(self, config: Dict[str, Any]) -> None:
        """Update the run config with additional key-value pairs."""
        wandb.config.update(config, allow_val_change=True)

    def log_gif(self, gif_path: str, key: str = "episode_replay",
                step: Optional[int] = None, caption: str = "") -> None:
        """Upload a GIF file as a wandb.Video to the run.

        Args:
            gif_path: Local path to the .gif file.
            key: Metric key under which the video appears in the dashboard.
            step: Training step for the x-axis.
            caption: Optional caption shown below the video.
        """
        if os.path.isfile(gif_path):
            wandb.log(
                {key: wandb.Video(gif_path, fps=10, format="gif",
                                  caption=caption)},
                step=step,
            )

    def log_artifact(self, path: str, name: str = "model",
                     artifact_type: str = "model") -> None:
        """Upload a file or directory as a versioned W&B artifact."""
        artifact = wandb.Artifact(name, type=artifact_type)
        if os.path.isdir(path):
            artifact.add_dir(path)
        else:
            artifact.add_file(path)
        self.run.log_artifact(artifact)

    def finish(self) -> None:
        """Close the active W&B run."""
        wandb.finish()

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "WandbLogger":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.finish()


class SB3WandbCallback(BaseCallback):
    """SB3 callback that logs per-episode metrics to W&B.

    At every completed episode (detected via SB3's Monitor wrapper),
    logs ``episode_reward`` and ``episode_length`` to the active W&B run.
    These produce the two main dashboard charts:
        - Rewards over episodes
        - Episode length over episodes
    """

    def __init__(self, wandb_logger: WandbLogger, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        self.wandb_logger = wandb_logger
        self._episode_count = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self._episode_count += 1
                ep_reward = info["episode"]["r"]
                ep_length = info["episode"]["l"]
                self.wandb_logger.log(
                    {
                        "episode_reward": ep_reward,
                        "episode_length": ep_length,
                        "episode": self._episode_count,
                    },
                    step=self.num_timesteps,
                )
        return True


class GifRecorderCallback(BaseCallback):
    """SB3 callback that records a GIF of agent behaviour at fixed intervals.

    At every ``gif_freq`` timesteps, resets the *existing* training environment,
    runs one episode with the current policy (deterministic), saves the frames
    as a GIF, and uploads it to W&B.

    Storage-conscious: only records at the specified interval, and cleans
    up local temp files immediately after upload.
    """

    def __init__(
        self,
        train_env,
        wandb_logger: WandbLogger,
        gif_freq: int = 10000,
        verbose: int = 1,
    ) -> None:
        """
        Args:
            train_env: The *already-initialised* FireEnv used for training.
                       Reused for GIF recording to avoid re-triggering
                       LANDFIRE downloads.
            wandb_logger: Active WandbLogger for uploading.
            gif_freq: Record a GIF every this many timesteps.
            verbose: 0 = quiet, 1 = print status.
        """
        super().__init__(verbose=verbose)
        self.train_env = train_env
        self.wandb_logger = wandb_logger
        self.gif_freq = gif_freq

    def _on_step(self) -> bool:
        # Skip first trigger (step 0 / very early) and only record at milestones
        if self.num_timesteps < self.gif_freq:
            return True
        if self.num_timesteps % self.gif_freq != 0:
            return True

        try:
            self._record_and_upload()
        except Exception as e:
            if self.verbose >= 1:
                print(f"[GifRecorderCallback] Failed to record GIF: {e}")
        return True

    def _record_and_upload(self) -> None:
        """Record episode frames with full terrain layers and upload to W&B.

        Rendering layers (bottom to top):
          1. Terrain base: elevation hillshade + population heat map + roads
          2. Fire overlay: burning (red-orange), burned (dark charcoal)
          3. Mitigation: fireline (blue), scratchline (orange), wetline (cyan)
          4. Fire trucks: yellow 3×3 markers
          5. Fire stations: cyan triangles (static)
        """
        from PIL import Image as PILImage, ImageDraw

        gif_env = self.train_env

        # ---- Build terrain base frame once --------------------------------
        has_terrain = hasattr(gif_env, "build_terrain_base_rgb")
        if has_terrain:
            terrain_base = gif_env.build_terrain_base_rgb()  # (H, W, 3) uint8
        else:
            h_fb = getattr(gif_env, "grid_rows", 256)
            w_fb = getattr(gif_env, "grid_cols", 256)
            terrain_base = np.full((h_fb, w_fb, 3), [34, 139, 34], dtype=np.uint8)

        obs, _ = gif_env.reset()
        done = False
        total_reward = 0.0
        frames = []
        frame_step = 0

        num_agents = getattr(gif_env, "num_agents", 1)
        frame_interval = max(1, num_agents)

        # Station markers (static)
        stations = getattr(gif_env, "stations", [])

        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = gif_env.step(int(action))
            total_reward += reward
            done = terminated or truncated
            frame_step += 1

            if frame_step % frame_interval == 0 or done:
                # ---- Start with terrain base --------------------------------
                rgb = terrain_base.copy()
                h, w = rgb.shape[:2]

                # Support both DataDrivenFireEnv and FireEnv
                if hasattr(gif_env, "fire_map"):
                    fire_map = np.copy(gif_env.fire_map)
                else:
                    fire_map = np.copy(gif_env.sim.fire_map)

                # ---- Fire / burned overlay ----------------------------------
                burning_mask = fire_map == 1
                burned_mask  = fire_map == 2
                # Blend burning cells: bright orange-red (alpha 0.85)
                rgb[burning_mask] = (
                    rgb[burning_mask] * 0.15 + np.array([255, 65, 0]) * 0.85
                ).astype(np.uint8)
                # Blend burned cells: dark charcoal (alpha 0.80)
                rgb[burned_mask] = (
                    rgb[burned_mask] * 0.20 + np.array([45, 30, 25]) * 0.80
                ).astype(np.uint8)

                # ---- Mitigation overlay -------------------------------------
                fl_mask = fire_map == 3   # fireline   → blue
                sl_mask = fire_map == 4   # scratchline → orange
                wl_mask = fire_map == 5   # wetline    → cyan
                rgb[fl_mask] = (rgb[fl_mask] * 0.2 + np.array([0, 100, 255]) * 0.8).astype(np.uint8)
                rgb[sl_mask] = (rgb[sl_mask] * 0.2 + np.array([255, 140, 0]) * 0.8).astype(np.uint8)
                rgb[wl_mask] = (rgb[wl_mask] * 0.2 + np.array([0, 210, 210]) * 0.8).astype(np.uint8)

                # ---- Fire truck agents: yellow 3×3 markers ------------------
                if hasattr(gif_env, "agents"):
                    agent_positions = [ag["pos"] for ag in gif_env.agents]
                else:
                    agent_positions = [gif_env.agent_pos]

                for ap in agent_positions:
                    for di in (-1, 0, 1):
                        for dj in (-1, 0, 1):
                            rr, cc = ap[0] + di, ap[1] + dj
                            if 0 <= rr < h and 0 <= cc < w:
                                rgb[rr, cc] = [255, 220, 0]  # bright yellow

                # ---- Scale up 4× (nearest-neighbour for pixel crispness) ----
                scale = 4
                img = PILImage.fromarray(rgb).resize(
                    (w * scale, h * scale), PILImage.NEAREST
                )
                draw = ImageDraw.Draw(img)

                # Fire stations: cyan triangle markers
                for s in stations:
                    sr, sc = s.get("grid_row", 0), s.get("grid_col", 0)
                    cx = sc * scale + scale // 2
                    cy = sr * scale + scale // 2
                    sz = scale * 2
                    draw.polygon(
                        [(cx, cy - sz), (cx - sz, cy + sz), (cx + sz, cy + sz)],
                        fill=(0, 200, 200), outline=(255, 255, 255)
                    )

                frames.append(img)

        if frames:
            tmp_dir = tempfile.mkdtemp(prefix="wandb_gif_")
            gif_path = os.path.join(tmp_dir, "episode.gif")
            frames[0].save(
                gif_path,
                save_all=True,
                append_images=frames[1:],
                duration=100,  # 100ms per frame = 10 fps
                loop=0,
            )
            caption = f"Step {self.num_timesteps} | Reward: {total_reward:.1f}"
            self.wandb_logger.log_gif(
                gif_path,
                key="episode_replay",
                step=self.num_timesteps,
                caption=caption,
            )
            if self.verbose >= 1:
                print(f"[GifRecorderCallback] Logged GIF at step {self.num_timesteps} "
                      f"(reward={total_reward:.1f}, {len(frames)} frames)")
            shutil.rmtree(tmp_dir, ignore_errors=True)
        else:
            if self.verbose >= 1:
                print(f"[GifRecorderCallback] No frames captured at step {self.num_timesteps}")
