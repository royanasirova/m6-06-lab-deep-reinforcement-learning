![logo_ironhack_blue 7](https://user-images.githubusercontent.com/23629340/40541063-a07a0a8a-601a-11e8-91b5-2f13e4e6b441.png)

# Lab | Deep Reinforcement Learning

## Overview

Today's lesson took you from tabular Q-learning all the way to PPO, the algorithm that powers everything from AlphaGo to RLHF for LLMs. In this lab you'll get your hands on two of the most important Deep RL algorithms in practice: you'll **implement DQN from scratch** on `CartPole-v1`, then train a **PPO** agent with **Stable-Baselines3** and compare the two on the same environment plus one that's substantially harder (`LunarLander-v2`).

By the end you'll have a working mental model of what each algorithm is good at, what its training dynamics actually look like, and when to reach for which.

## Learning Goals

By the end of this lab you should be able to:

- Use **Gymnasium** to set up a standard Deep RL environment.
- Implement a small **DQN** agent from scratch with a replay buffer and a target network.
- Train a **PPO** agent with **Stable-Baselines3** in a few lines of code.
- Plot, smooth, and interpret reward curves from a Deep RL training run.
- Recognise when DQN-style value-based methods are the right choice and when policy-gradient methods like PPO are.

## Setup and Context

You'll work in a single Jupyter Notebook. The environments are bundled with `gymnasium`. `CartPole-v1` is a classic balancing task — keep a pole upright on a moving cart. `LunarLander-v2` is harder — land a 2D spacecraft on a designated pad without crashing.

CPU is fine for everything in this lab. A DQN run on CartPole takes about 5 minutes; PPO on LunarLander takes 10–20 minutes.

## Requirements

### Fork and clone

1. Fork this repository to your own GitHub account.
2. Clone the fork to your local machine.
3. Navigate into the project directory.

### Python environment

```bash
pip install numpy matplotlib torch "gymnasium[box2d,classic-control]" stable-baselines3
```

> `LunarLander-v2` lives in the `box2d` extras of Gymnasium; if you skip the `box2d` extra, only Task 3's LunarLander run will fail to load. On Linux you may also need to install `swig` system-wide (`brew install swig` on macOS) before the `box2d` install succeeds.

## Getting Started

1. Create a notebook called **`m6-06-deep-reinforcement-learning.ipynb`**.
2. Standard imports:

```python
import collections
import random
import time

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
```

3. As a warm-up, instantiate `CartPole-v1`, reset it, and print the observation and action spaces. Run a short loop with **random actions** and print the total reward of the episode — this is the baseline a learning agent has to beat.

```python
env = gym.make("CartPole-v1")
obs, _ = env.reset(seed=42)
print("observation space:", env.observation_space)
print("action space:", env.action_space)
```

## Tasks

### Task 1 — DQN from Scratch on CartPole-v1

Build a DQN agent following the recipe from the lesson — Q-network, replay buffer, target network, ε-greedy exploration.

1. Implement a small Q-network in PyTorch: input dimension = `env.observation_space.shape[0]`, two hidden layers of 128 units with ReLU, output dimension = `env.action_space.n`.
2. Implement a `ReplayBuffer` class that supports `push(state, action, reward, next_state, done)` and `sample(batch_size)`. A `collections.deque(maxlen=50_000)` is plenty for CartPole.
3. Implement the training loop:
   - Keep two networks: `q_net` (trained) and `target_net` (frozen, periodically synced from `q_net`).
   - Use ε-greedy action selection with linear ε decay from `1.0` to `0.05` over the first ~5 000 steps.
   - After at least 1 000 transitions are in the buffer, do one gradient step per environment step on a random mini-batch of 64.
   - Sync `target_net ← q_net` every 1 000 steps.
   - Use `Adam(lr=1e-3)` and γ = 0.99.
4. Run training for **30 000 environment steps** and store the **total reward per episode** in a list.

**Expected behaviour.** A correctly implemented DQN solves `CartPole-v1` (average reward ≥ 475 over 100 episodes) somewhere between 10 000 and 25 000 steps. If your reward is still flat at the end, double-check the target-network sync, the ε decay, and the loss (it should be MSE between `q_net(s).gather(action)` and `r + γ · target_net(s').max()` for non-terminal transitions; just `r` for terminal ones).

5. Plot the per-episode reward and a 100-episode moving average on the same axes. Report the average reward over the last 100 episodes.

### Task 2 — PPO with Stable-Baselines3

Now do the same task — and one harder one — with a well-tuned library implementation.

1. Train a **PPO** agent on `CartPole-v1`:

```python
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

env = gym.make("CartPole-v1")
ppo_cartpole = PPO("MlpPolicy", env, verbose=0, seed=42)
t0 = time.perf_counter()
ppo_cartpole.learn(total_timesteps=50_000)
cartpole_train_time = time.perf_counter() - t0

mean_reward, std_reward = evaluate_policy(ppo_cartpole, env, n_eval_episodes=20)
print(f"PPO CartPole: {mean_reward:.1f} ± {std_reward:.1f} over 20 episodes "
      f"(trained in {cartpole_train_time:.0f}s)")
```

2. Train a second PPO agent on `LunarLander-v2` with the same hyperparameters but **300 000 total timesteps** (this environment is much harder than CartPole).

```python
env_ll = gym.make("LunarLander-v2")
ppo_ll = PPO("MlpPolicy", env_ll, verbose=0, seed=42)
t0 = time.perf_counter()
ppo_ll.learn(total_timesteps=300_000)
ll_train_time = time.perf_counter() - t0

mean_reward, std_reward = evaluate_policy(ppo_ll, env_ll, n_eval_episodes=20)
print(f"PPO LunarLander: {mean_reward:.1f} ± {std_reward:.1f} over 20 episodes "
      f"(trained in {ll_train_time:.0f}s)")
```

3. Use Stable-Baselines3's built-in `Monitor` wrapper or its logger to capture the per-episode rewards during training for both runs, and plot 100-episode moving-average reward curves for both environments.

**Expected behaviour.** PPO solves `CartPole-v1` (reward ≥ 195 on average) within ~20 000 timesteps and reliably reaches the 500 cap. On `LunarLander-v2` PPO usually clears the 200-reward "solved" threshold somewhere between 150 000 and 300 000 timesteps.

### Task 3 — Comparison and Reflection

Fill in this table from your runs in Tasks 1 and 2:

| Agent | Environment | Wall-clock training time | Avg reward (last 100 episodes) |
|---|---|---|---|
| DQN (from scratch) | CartPole-v1 | … | … |
| PPO (SB3) | CartPole-v1 | … | … |
| PPO (SB3) | LunarLander-v2 | … | … |

Then, in a markdown cell, answer the following questions in 3–5 sentences each:

1. Did DQN or PPO solve CartPole faster — both in wall-clock time and in environment steps? Why do you think that is?
2. Could you imagine training the **same DQN code** on `LunarLander-v2` and getting a similar result? What would you expect to go wrong, and which Deep RL improvement from the lesson (Double DQN, Dueling, PER, Rainbow) would you reach for first?
3. Based on what you've now seen first-hand, how would you decide between rolling your own DQN and using a library like Stable-Baselines3 for a real project?

## Submission

### What to submit

- `m6-06-deep-reinforcement-learning.ipynb` — completed notebook.

### Definition of done (checklist)

- [ ] Gymnasium environment instantiated and random-action baseline printed.
- [ ] DQN from scratch trains on CartPole-v1 with reward curve plotted.
- [ ] PPO trains on CartPole-v1 and LunarLander-v2 with reward curves plotted.
- [ ] Comparison table filled in with training times and final rewards.
- [ ] Written reflection answering the three questions in Task 3.
- [ ] `Kernel → Restart & Run All` produces no errors.

### How to submit (Git workflow)

```bash
git add .
git commit -m "lab: complete deep reinforcement learning"
git push origin main
```

Then open a **Pull Request** on the original repository describing your reward curves and your comparison between value-based and policy-gradient Deep RL.
