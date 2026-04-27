"""Main training orchestration for the firefighting RL agent.

Loads YAML configs, builds environment and agent, wires up W&B logging
(including periodic GIF recording), runs training, evaluates the final
model, and saves artefacts.
"""
import os
from typing import Optional

import numpy as np
import yaml

from agents.dqn_agent import DQNAgent
from environment.data_fire_env import DataDrivenFireEnv
from training.callbacks import CheckpointCallback, EvalCallback
from training.evaluate import evaluate_agent
from utils.logger import WandbLogger, SB3WandbCallback, GifRecorderCallback


def _load_yaml(path: str) -> dict:
    """Load a YAML file and return its contents as a dict."""
    with open(path, "r") as fh:
        return yaml.safe_load(fh) or {}


def train(
    config_dir: str = "config/",
    algorithm: Optional[str] = None,
    total_timesteps: Optional[int] = None,
    resume_from: Optional[str] = None,
) -> None:
    """End-to-end training pipeline with W&B tracking.

    Logs to W&B:
    - Per-episode: reward, episode length  (→ line charts)
    - Periodic: GIF recordings of agent behaviour (→ media panel)
    - Final: evaluation metrics, model checkpoint artifact

    Args:
        config_dir: Path to YAML config directory.
        algorithm: Override algorithm from config ("DQN" or "PPO").
        total_timesteps: Override total training timesteps.
    """
    # ------------------------------------------------------------------
    # 1. Load configs
    # ------------------------------------------------------------------
    env_cfg = _load_yaml(os.path.join(config_dir, "env_config.yaml"))
    agent_cfg = _load_yaml(os.path.join(config_dir, "agent_config.yaml"))
    train_cfg = _load_yaml(os.path.join(config_dir, "train_config.yaml"))

    if algorithm is not None:
        train_cfg["algorithm"] = algorithm
    if total_timesteps is not None:
        train_cfg["total_timesteps"] = total_timesteps

    merged_env_cfg = {**env_cfg, **agent_cfg}

    algo_name: str = train_cfg.get("algorithm", "DQN").upper()
    run_name: str = train_cfg.get("run_name", "reference_agent")
    timesteps: int = int(train_cfg.get("total_timesteps", 50000))
    checkpoint_dir: str = train_cfg.get("checkpoint_dir", "checkpoints/")
    checkpoint_freq: int = int(train_cfg.get("checkpoint_freq", 10000))
    eval_episodes: int = int(train_cfg.get("eval_episodes", 5))
    gif_freq: int = int(train_cfg.get("gif_freq", 10000))

    # ------------------------------------------------------------------
    # 2. Setup W&B logger
    # ------------------------------------------------------------------
    all_params = {
        **{f"env/{k}": v for k, v in env_cfg.items()},
        **{f"agent/{k}": v for k, v in agent_cfg.items()},
        **{f"train/{k}": v for k, v in train_cfg.items()},
    }
    # Convert non-scalar values to strings for W&B config
    config_for_wandb = {
        k: str(v) if isinstance(v, (list, dict)) else v
        for k, v in all_params.items()
    }

    project = os.getenv("WANDB_PROJECT", train_cfg.get("wandb_project", "Firefighting RL"))
    logger = WandbLogger(
        project=project,
        run_name=run_name,
        config=config_for_wandb,
        tags=["reference_agent", algo_name],
    )

    try:
        # ------------------------------------------------------------------
        # 3. Build environment
        # ------------------------------------------------------------------
        env = DataDrivenFireEnv(merged_env_cfg)

        # ------------------------------------------------------------------
        # 4. Build agent
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

        if algo_name != "DQN":
            raise ValueError(f"Unknown algorithm '{algo_name}'. Only 'DQN' is supported.")
        agent = DQNAgent(env, agent_model_cfg)

        # Resume from checkpoint if specified
        resume_step = 0
        if resume_from is not None:
            print(f"[train] Resuming from checkpoint: {resume_from}")
            agent.load(resume_from)
            # Parse step count from filename like "reference_agent_20000_steps.zip"
            import re
            m = re.search(r"_(\d+)_steps", resume_from)
            if m:
                resume_step = int(m.group(1))
            remaining = timesteps - resume_step
            if remaining <= 0:
                print(f"[train] Already completed {resume_step} steps (target {timesteps}). Nothing to do.")
                return
            timesteps = remaining
            print(f"[train] Resuming from step {resume_step}, {timesteps} steps remaining")

        # ------------------------------------------------------------------
        # 5. Build callbacks
        # ------------------------------------------------------------------
        wandb_cb = SB3WandbCallback(logger)
        checkpoint_cb = CheckpointCallback(
            checkpoint_freq=checkpoint_freq,
            checkpoint_dir=checkpoint_dir,
            run_name=run_name,
            wandb_logger=logger,
        )
        gif_cb = GifRecorderCallback(
            train_env=env,
            wandb_logger=logger,
            gif_freq=gif_freq,
        )

        # ------------------------------------------------------------------
        # 6. Train
        # ------------------------------------------------------------------
        print(f"[train] Starting {algo_name} training for {timesteps} timesteps …")
        print(f"[train] GIFs will be recorded every {gif_freq} steps")
        agent.train(
            total_timesteps=timesteps,
            callbacks=[wandb_cb, checkpoint_cb, gif_cb],
        )

        # ------------------------------------------------------------------
        # 7. Save final model
        # ------------------------------------------------------------------
        os.makedirs(checkpoint_dir, exist_ok=True)
        final_path = os.path.join(checkpoint_dir, f"{run_name}_{algo_name.lower()}")
        agent.save(final_path)
        print(f"[train] Final model saved to {final_path}.zip")

        # ------------------------------------------------------------------
        # 8. Evaluate
        # ------------------------------------------------------------------
        print(f"[train] Running evaluation over {eval_episodes} episodes …")
        eval_env = DataDrivenFireEnv(merged_env_cfg)
        eval_results = evaluate_agent(agent, eval_env, num_episodes=eval_episodes)
        print(
            f"[train] Eval results: "
            f"mean_reward={eval_results['mean_reward']:.3f} "
            f"std={eval_results['std_reward']:.3f} "
            f"mean_length={eval_results['mean_length']:.1f}"
        )

        # ------------------------------------------------------------------
        # 9. Log final metrics + model artifact to W&B
        # ------------------------------------------------------------------
        logger.log(
            {
                "final_mean_reward": eval_results["mean_reward"],
                "final_std_reward": eval_results["std_reward"],
                "final_mean_length": eval_results["mean_length"],
            },
            step=timesteps,
        )
        if os.path.isfile(final_path + ".zip"):
            logger.log_artifact(
                final_path + ".zip",
                name="final-model",
                artifact_type="model",
            )

        # ------------------------------------------------------------------
        # 10. Record one final GIF using matplotlib (no display needed)
        # ------------------------------------------------------------------
        print("[train] Recording final evaluation GIF …")
        try:
            import shutil
            import tempfile
            from PIL import Image as PILImage

            from PIL import ImageDraw
            gif_env = DataDrivenFireEnv(merged_env_cfg)
            terrain_base = gif_env.build_terrain_base_rgb() if hasattr(gif_env, "build_terrain_base_rgb") else None
            obs, _ = gif_env.reset()
            done = False
            total_reward = 0.0
            frames = []
            step_count = 0
            stations = getattr(gif_env, "stations", [])
            scale = 4

            while not done:
                action = agent.get_action(obs)
                obs, reward, terminated, truncated, _ = gif_env.step(action)
                total_reward += reward
                done = terminated or truncated
                step_count += 1

                if step_count % max(1, gif_env.num_agents) == 0 or done:
                    fire_map = np.copy(gif_env.fire_map)
                    h, w = fire_map.shape
                    rgb = terrain_base.copy() if terrain_base is not None else np.zeros((h, w, 3), dtype=np.uint8)
                    # Fire overlay
                    burning_mask = fire_map == 1
                    burned_mask  = fire_map == 2
                    rgb[burning_mask] = (rgb[burning_mask] * 0.15 + np.array([255, 65, 0]) * 0.85).astype(np.uint8)
                    rgb[burned_mask]  = (rgb[burned_mask]  * 0.20 + np.array([45, 30, 25]) * 0.80).astype(np.uint8)
                    # Mitigation
                    rgb[fire_map == 3] = (rgb[fire_map == 3] * 0.2 + np.array([0, 100, 255]) * 0.8).astype(np.uint8)
                    rgb[fire_map == 4] = (rgb[fire_map == 4] * 0.2 + np.array([255, 140, 0]) * 0.8).astype(np.uint8)
                    rgb[fire_map == 5] = (rgb[fire_map == 5] * 0.2 + np.array([0, 210, 210]) * 0.8).astype(np.uint8)
                    img = PILImage.fromarray(rgb).resize((w * scale, h * scale), PILImage.NEAREST)
                    draw = ImageDraw.Draw(img)
                    # Yellow agents: PIL circles drawn after upscale so they always
                    # appear above any mitigation colour (orange scratchline trail, etc.)
                    agent_radius = scale + 1
                    for ag in gif_env.agents:
                        ap = ag["pos"]
                        cx = ap[1] * scale + scale // 2
                        cy = ap[0] * scale + scale // 2
                        draw.ellipse(
                            [(cx - agent_radius - 1, cy - agent_radius - 1),
                             (cx + agent_radius + 1, cy + agent_radius + 1)],
                            fill=(40, 40, 40),
                        )
                        draw.ellipse(
                            [(cx - agent_radius, cy - agent_radius),
                             (cx + agent_radius, cy + agent_radius)],
                            fill=(255, 220, 0),
                        )
                    for s in stations:
                        cx = s.get("grid_col", 0) * scale + scale // 2
                        cy = s.get("grid_row", 0) * scale + scale // 2
                        sz = scale * 2
                        draw.polygon([(cx, cy - sz), (cx - sz, cy + sz), (cx + sz, cy + sz)], fill=(0, 200, 200), outline=(255, 255, 255))
                    frames.append(img)

            if frames:
                tmp_dir = tempfile.mkdtemp(prefix="wandb_final_gif_")
                gif_path = os.path.join(tmp_dir, "final_episode.gif")
                frames[0].save(
                    gif_path, save_all=True, append_images=frames[1:],
                    duration=100, loop=0,
                )
                logger.log_gif(
                    gif_path, key="final_episode", step=timesteps,
                    caption=f"Final eval | Reward: {total_reward:.1f}",
                )
                print(f"[train] Final GIF logged ({len(frames)} frames, reward={total_reward:.1f})")
                shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception as e:
            print(f"[train] Final GIF recording failed: {e}")

    finally:
        # ------------------------------------------------------------------
        # 11. Close W&B run (always, even on exception)
        # ------------------------------------------------------------------
        logger.finish()
        print("[train] W&B run closed.")
