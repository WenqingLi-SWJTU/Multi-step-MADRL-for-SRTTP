# Multi-step MADRL for Single-Track Railway Timetabling (SRTTP)

> A stable and scalable multi-step multi-agent deep reinforcement learning (MADRL) framework for the Single-Track Railway Timetabling Problem (SRTTP).
> Companion paper: *A Stable and Scalable Multi-Step Multi-Agent Deep Reinforcement Learning Framework for Single-Track Railway Timetabling*.

## Overview

This repository implements a **multi-step MADRL framework** for the SRTTP. It addresses three key bottlenecks of existing MADRL approaches: the "myopia" of single-step simulation and learning, unstable CTDE training, and the poor scalability of per-agent feed-forward networks (FNNs). The method introduces three core components:

1. **A multi-step MDP environment**: each timestep provides forward-looking information through
   - a **rule-based repair mechanism**, which checks for headway violations in the immediately following block and adjusts the dwell time of the affected trains to restore feasibility;
   - a **policy-conditioned forward search (n-step forward search)**, which samples actions from the current policy, simulates forward, and estimates the future conflict probability ρ;
   - a **unified weighted reward** that fuses the actual dwell time, the deviation from the repaired value, and the forward-search conflict probability into a single learning signal.
2. **Training algorithm**: **GAE** for multi-step value estimation and **HAPPO** for policy updates (following the MAT training protocol), improving stability under non-stationary multi-agent dynamics.
3. **A shared-parameter Transformer**: an encoder–decoder structure whose parameter size is independent of the number of active agents, enabling scalability to large fleets and cross-instance policy transfer.

## Requirements

- Python 3.x
- PyTorch
- NumPy
- Gymnasium
- TensorBoard / wandb (optional, for logging)

Suggested installation:

```bash
pip install torch numpy gymnasium tensorboard
# optional
pip install wandb


| File | Description |
| --- | --- |
| `get_config.py` | Hyperparameter configuration and the three scenario (line) configs: `10_10`, `30_15`, `70_20` |
| `env_util.py` | Low-level environment utilities: time computation, headway constraint checks, action repair |
| `MSTTLE.py` | Core multi-step simulation environment for single-track railways (TTLE) |
| `env_wrapper.py` | Multi-agent environment wrapper (parallel rollouts, observation/action shaping) |
| `ma_transformer.py` | Shared-parameter Transformer (encoder/decoder, self/cross/masked attention) |
| `transformer_policy.py` | Policy network wrapper (actor/critic) |
| `mat_trainer.py` | MAT training algorithm (GAE + HAPPO/PPO objectives, optimizers) |
| `shared_buffer.py` | Shared experience replay buffer |
| `base_runner.py` | Base class for the training loop (collect / compute / train / save) |
| `TTLE_runner.py` | Training runner for this environment (forward search, curriculum-style randomization, convergence detection) |
| `trainer.py` | Training entry point |
| `utils.py` | Network initialization and parallel/autoregressive action decoding utilities |

| Scenario | Trains (Down, Up) | Stations | Blocks |
| --- | --- | --- | --- |
| `10_10` | 10 (5, 5) | 10 | 9 |
| `30_15` | 30 (15, 15) | 15 | 14 |
| `70_20` | 70 (35, 35) | 20 | 19 |
