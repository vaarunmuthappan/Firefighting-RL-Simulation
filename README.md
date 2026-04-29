# Firefighting RL — Multi-Agent Wildfire Suppression

> **DS-GA 3001 · Reinforcement Learning · NYU Center for Data Science**
> Vaarun Muthappan · Simran Khullar · Rishabh Patil
> Instructor: Prof. Jeremy Curuksu

---

## Overview

This project trains a multi-agent reinforcement learning system to discover optimal fireline construction strategies on real Camp Fire (2018) terrain. Agents are deployed from real fire station locations and must intercept a historically recorded fire — derived from VIIRS satellite data — before it spreads across Butte County, CA.

Rather than using a synthetic fire simulator, the environment replays the actual 2018 Camp Fire arrival timeline at 375m resolution. Each agent observes a 55×55×8 spatial window and selects from 20 discrete actions (movement + mitigation type) every 3-hour fire timestep.

PPO (Proximal Policy Optimization) consistently outperformed DQN across all experimental runs. Both algorithms used an identical CNN architecture to ensure a fair comparison.

---

## Environment

| Property | Value |
|---|---|
| Fire data | 2018 Camp Fire, Butte County CA (VIIRS/FEDS satellite) |
| Grid resolution | 375m per cell |
| Observation | 55×55×8 per agent (220px window, 4× downsample) |
| Actions | 20 discrete (5 movement × 4 mitigation) |
| Sub-steps per fire tick | 36 agent steps per 3-hour fire timestep |
| Max episode steps | 5,000 |
| Wind | speed=2, direction=135°, moisture=3% |

### Observation Channels

| Channel | Description |
|---|---|
| 0 | `fire_map` — current burn / mitigation status |
| 1–4 | `fire_ahead_1…4` — predicted fire at t+3h / 6h / 9h / 12h |
| 5 | `other_agents` — binary, co-agent positions |
| 6 | `mitigation` — existing fireline / scratchline / wetline |
| 7 | `agent_id_norm` — unique constant per agent (breaks symmetry) |

---

## Reward Function

Rewards are individually attributed — each agent is only rewarded or penalized for outcomes in its own Voronoi sector (the fire cells closest to its home station).

| Signal | Condition |
|---|---|
| +5 / cell | Agent moves toward fire front in its assigned zone |
| +400 / +200 / +150 | Mitigation placed 6h / 3h / 9h before fire arrives |
| +300 / +150 / +75 | Fireline / scratchline / wetline placed at active fire edge |
| +600 | Fire arrived at a cell that already had mitigation (block held) |
| −50 / new cell | Fire spread within agent's own sector |
| −1 / step | Flat penalty to discourage inaction |

---

## Algorithms

Both algorithms use the identical `ChannelLastCNN` architecture:
`Conv(32, 3×3, s1) → Conv(64, 3×3, s2) → Conv(64, 3×3, s2) → Flatten → Linear(256) → ReLU`

### PPO (Primary)

On-policy algorithm. Each agent acts one at a time, so the environment changes after every action. As an on-policy algorithm, PPO always learns from the latest state of the world rather than from outdated past experiences.

Key hyperparameters (`config/train_config.yaml`):

| Parameter | Value |
|---|---|
| Total timesteps | 500,000 |
| Learning rate | 3e-4 |
| n_steps | 5,000 |
| Batch size | 256 |
| n_epochs | 10 |
| Gamma (γ) | 0.99 |
| GAE Lambda (λ) | 0.95 |
| Clip range (ε) | 0.2 |

### DQN (Baseline)

Off-policy algorithm. Learns by estimating the value of every possible action and selecting the best one, using a memory bank of past experiences to improve over time. Underperformed in this setting because replayed transitions quickly became stale as the multi-agent environment shifted.

---

## Installation

```bash
git clone https://github.com/vaarunmuthappan/Firefighting-RL-Simulation
cd Firefighting-RL-Simulation
pip install -r requirements.txt
```

**Requirements:** Python 3.9+, PyTorch 2.0+, Stable-Baselines3 2.0+, Gymnasium, W&B

---

## Training

```bash
# Train with default config (PPO, 500k steps)
python main_train.py

# Override algorithm
python main_train.py --algorithm PPO

# Override timesteps
python main_train.py --timesteps 100000

# Resume from checkpoint
python main_train.py --resume checkpoints/model_50000_steps.zip
```

Training logs, checkpoints, and GIFs are saved every 10,000 steps. W&B integration is enabled by default — set your API key via `WANDB_API_KEY` environment variable or `.env` file.