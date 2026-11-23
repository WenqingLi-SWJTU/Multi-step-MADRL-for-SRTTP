# Multi-step MADRL for Single-Track Railway Timetabling (SRTTP)

> A stable and scalable multi-step multi-agent deep reinforcement learning (MADRL) framework for the Single-Track Railway Timetabling Problem (SRTTP).  
> Companion paper: _A Stable and Scalable Multi-Step Multi-Agent Deep Reinforcement Learning Framework for Single-Track Railway Timetabling_.

## Overview

This repository implements a **multi-step MADRL framework** for the SRTTP. It addresses three key bottlenecks of existing MADRL approaches: the "myopia" of single-step simulation and learning, unstable CTDE training, and the poor scalability of per-agent feed-forward networks (FNNs). The method introduces three core components:

1.  **A multi-step MDP environment**: each timestep provides forward-looking information through
    *   a **rule-based repair mechanism**, which checks for headway violations in the immediately following block and adjusts the dwell time of the affected trains to restore feasibility;
        
    *   a **policy-conditioned forward search (n-step forward search)**, which samples actions from the current policy, simulates forward, and estimates the future conflict probability ρ;
        
    *   a **unified weighted reward** that fuses the actual dwell time, the deviation from the repaired value, and the forward-search conflict probability into a single learning signal.
        
2.  **Training algorithm**: **GAE** for multi-step value estimation and **HAPPO** for policy updates (following the MAT training protocol), improving stability under non-stationary multi-agent dynamics.
    
3.  **A shared-parameter Transformer**: an encoder–decoder structure whose parameter size is independent of the number of active agents, enabling scalability to large fleets and cross-instance policy transfer.
    

## Requirements

*   Python 3.x
    
*   PyTorch
    
*   NumPy
    
*   Gymnasium
    
*   TensorBoard / wandb (optional, for logging)
    

Suggested installation:

```bash
pip install torch numpy gymnasium tensorboard
# optional
pip install wandb
```

## Project Structure

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

## Quick Start

The three experiment scenarios correspond to different scales (drawn from a case on the Gui-Guang line):
| Scenario | Trains (Down, Up) | Stations | Blocks |
| --- | --- | --- | --- |
| `10_10` | 10 (5, 5) | 10 | 9 |
| `30_15` | 30 (15, 15) | 15 | 14 |
| `70_20` | 70 (35, 35) | 20 | 19 |

Training (e.g., the `10_10` scenario):

```bash
python trainer.py --scenario 10_10
```

Common hyperparameters (defaults in `get_config.py`):

```bash
python trainer.py \
  --scenario 30_15 \
  --n_rollout_threads 10 \
  --episode_length 200 \
  --search_depth_N 1 \
  --use_n_step_search --use_expert_rule \
  --weight_dwell 0.6 --weight_adjust 0.3 --weight_multistep 0.1 \
  --n_block 3 --n_embd 256 --n_head 8 \
  --gamma 0.99 --gae_lambda 0.99 --clip_param 0.2 \
  --lr 1e-4 --critic_lr 5e-4 \
  --save_interval 100
```

*   `--search_depth_N` / `--use_n_step_search`: forward-search depth (default 1).
    
*   `--use_expert_rule`: enable the rule-based repair.
    
*   `--weight_dwell / _adjust / _multistep`: reward weights for the three components (default 0.6 / 0.3 / 0.1).
    
*   `--model_dir`: load an existing model for inference / continued training.
    

## Key Results (from the paper)

*   **Case 1**: objective 196 min, 7.1% better than MAPPO and 31.8% better than Gurobi, with faster convergence.
    
*   **Case 2**: 587 min, 8.1% better than MAPPO and 24.1% better than Gurobi.
    
*   **Case 3 (70 trains)**: only this method converges within the computational budget (~2873.8 min); MAA2C / MAPPO / Gurobi all exceed the memory or time budget and do not converge, demonstrating the scalability of the shared-parameter architecture.
    

## Notes and Limitations

*   The environment focuses on the core safety-headway constraints; additional practical constraints such as station track capacity, train priorities, and maintenance windows can be added through the constraint-checking / reward modules in `MSTTLE.py` (see Appendix A of the paper).
    
*   The current implementation assumes deterministic transitions and uniform block sections, among other simplifications. Directions for extension—e.g., stochastic demand and variable departure times—are discussed in Section 7 of the paper.
    

## Citation

If you use this code, please cite the companion paper:

> W. Li, S. Ni, D. Chen, J. Pan, M. Lv, L. Zhang. _A Stable and Scalable Multi-Step Multi-Agent Deep Reinforcement Learning Framework for Single-Track Railway Timetabling_.