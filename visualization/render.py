"""Visualisation utilities for the firefighting RL project.

Provides:
- save_episode_gif: Record an episode with an agent and save as an animated GIF.
- plot_training_curves: Read SB3 Monitor CSV logs and plot reward/length curves.
"""
import os
from typing import Optional

import numpy as np


def save_episode_gif(
    env,
    agent,
    save_path: str,
    num_episodes: int = 1,
) -> None:
    """Run episodes with ``agent`` and save an animated GIF of the fire spread.

    Enables SimFire's pygame rendering before resetting the environment so
    that all frames are captured, then calls ``env.sim.save_gif()`` at the
    end of each episode.

    Args:
        env: A FireEnv instance (must expose ``env.sim``).
        agent: An agent with a ``get_action(obs) -> int`` method.
        save_path: Directory path where the GIF(s) will be written.
        num_episodes: Number of episodes to record (default 1).
    """
    os.makedirs(save_path, exist_ok=True)

    for ep in range(num_episodes):
        # Enable rendering so SimFire captures frames
        env.sim.rendering = True

        obs, _ = env.reset()
        done = False

        while not done:
            action = agent.get_action(obs)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        # Save GIF for this episode
        ep_save_path = os.path.join(save_path, f"episode_{ep}")
        env.sim.save_gif(ep_save_path)
        print(f"[save_episode_gif] Saved GIF to {ep_save_path}")

        # Disable rendering after saving
        env.sim.rendering = False


def plot_training_curves(
    log_path: str,
    save_path: Optional[str] = None,
) -> None:
    """Plot episode reward and length curves from an SB3 Monitor CSV log.

    SB3's ``Monitor`` wrapper writes a CSV with columns ``r`` (episode reward)
    and ``l`` (episode length). This function reads that file and produces
    a two-panel matplotlib figure.

    Args:
        log_path: Path to the Monitor CSV file (e.g. ``"logs/monitor.csv"``).
        save_path: If provided, the figure is saved to this path instead of
            being displayed interactively. Accepts any extension supported by
            matplotlib (``".png"``, ``".pdf"``, etc.).

    Raises:
        ImportError: If matplotlib is not installed.
        FileNotFoundError: If ``log_path`` does not exist.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plot_training_curves. "
            "Install it with: pip install matplotlib"
        ) from exc

    import csv

    if not os.path.isfile(log_path):
        raise FileNotFoundError(f"Monitor CSV not found: {log_path}")

    rewards = []
    lengths = []

    with open(log_path, "r") as fh:
        # SB3 Monitor CSVs start with a comment line (#) then a header
        reader = csv.DictReader(
            (line for line in fh if not line.startswith("#"))
        )
        for row in reader:
            try:
                rewards.append(float(row["r"]))
                lengths.append(float(row["l"]))
            except (KeyError, ValueError):
                continue

    if not rewards:
        print("[plot_training_curves] No data found in log file.")
        return

    episodes = np.arange(1, len(rewards) + 1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax1.plot(episodes, rewards, linewidth=0.8, alpha=0.6, label="Episode reward")
    # Smoothed curve
    if len(rewards) >= 10:
        window = max(1, len(rewards) // 20)
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax1.plot(
            np.arange(window, len(rewards) + 1),
            smoothed,
            linewidth=2,
            label=f"Smoothed (window={window})",
        )
    ax1.set_ylabel("Episode Reward")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(episodes, lengths, linewidth=0.8, alpha=0.6, color="orange")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Episode Length (steps)")
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Training Curves")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[plot_training_curves] Figure saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)
