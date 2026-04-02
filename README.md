# Hamaya Assistive Arm in Modern Python

This repo focuses only on the assistive-arm extension inspired by Hamaya et al. (2017).

The implementation keeps the core ideas that matter for that paper:

- an augmented human-robot state `[q, qdot, Eb, Et]`
- a learned GP dynamics model over `(state, assist pressure) -> state delta`
- a shared-control assistive policy that depends on arm state and EMG
- an EMG-only long-term cost
- visualizations for rollouts, learned assistive policy maps, and predicted interaction surfaces

The stack is:

- `numpy` for simulation and dataset handling
- `gpytorch` for the learned interaction model
- `torch` for differentiable policy optimization
- `matplotlib` and `imageio` for diagnostics and animations
- `uv` for package management

## Project layout

- [`src/pilco_py/assistive_env.py`](/Users/weiliu/code/pilco-py/src/pilco_py/assistive_env.py): 1-DoF human-exoskeleton simulation
- [`src/pilco_py/gpytorch_dynamics.py`](/Users/weiliu/code/pilco-py/src/pilco_py/gpytorch_dynamics.py): exact GP dynamics model
- [`src/pilco_py/policy.py`](/Users/weiliu/code/pilco-py/src/pilco_py/policy.py): positive-output assistive policy
- [`src/pilco_py/assistive_trainer.py`](/Users/weiliu/code/pilco-py/src/pilco_py/assistive_trainer.py): training loop
- [`src/pilco_py/assistive_visualization.py`](/Users/weiliu/code/pilco-py/src/pilco_py/assistive_visualization.py): plots and GIF generation

## Install

```bash
uv sync
```

## Run

Default training run:

```bash
uv run pilco-train --episodes 10 --initial-rollouts 5 --policy-steps 200
```

Quick smoke run:

```bash
uv run pilco-train --episodes 2 --initial-rollouts 2 --policy-steps 20 --particles 32 --eval-particles 64 --gp-train-steps 25 --skip-video
```

## Outputs

Training writes to `artifacts/assistive/` by default:

- `training_curve.png`: observed vs predicted long-term EMG cost
- `episode_XX_diagnostics.png`: reference tracking, pressure, EMGs, and reward
- `episode_XX_policy_maps.png`: learned assistive policy map and one-step EMG interaction maps
- `episode_XX.gif`: rollout animation
- `history.json`: per-trial summary

## Source reference

The implementation is based on:

- the paper [`doc/Hamaya et al. - 2017 - Learning assistive strategies for exoskeleton robots from user-robot physical interaction.pdf`](/Users/weiliu/code/pilco-py/doc/Hamaya%20et%20al.%20-%202017%20-%20Learning%20assistive%20strategies%20for%20exoskeleton%20robots%20from%20user-robot%20physical%20interaction.pdf)
