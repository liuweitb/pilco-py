# PILCO in Modern Python

This repo contains two readable reproductions built from the original MATLAB `pilcoV0.9` code and the referenced papers:

- the original PILCO pendulum swing-up task
- a Hamaya-style assistive shared-control extension with augmented human-robot state `[q, qdot, Eb, Et]`

The code uses:

- `numpy` for simulation and dataset handling
- `gpytorch` for one-step GP dynamics models
- `torch` for differentiable policy optimization through particle rollouts
- `mujoco` for pendulum playback renders
- `matplotlib` and `imageio` for assistive-policy maps, diagnostics, and animations

## Included experiments

### Pendulum

This recreation keeps the original ingredients that matter for the classic PILCO experiment:

- a pendulum plant with the same state convention as the old code: `[dtheta, theta]`
- GP dynamics learned on state differences from observed transitions
- a bounded policy optimized against long-horizon predicted cost
- a saturating pendulum-tip cost matching the geometry used in the original paper
- saved diagnostics and a MuJoCo animation of learned rollouts

### Assistive Extension

The Hamaya et al. extension is implemented as a separate experiment path:

- integrated human-robot state `[q, qdot, Eb, Et]`
- assistive policy that depends on robot state and user EMG, matching the paper’s shared-control framing
- EMG-only long-term cost following Eq. (6) in the paper
- visualizations of learned assistive strategy maps and one-step EMG interaction surfaces inspired by Fig. 8 and Fig. 11

## What is faithful and what is modernized

The original MATLAB PILCO code computes analytic uncertain-input GP predictions and analytic policy gradients. Recreating every derivative helper from `pilcoV0.9` would turn this repo back into a dense symbolic port.

This version keeps the one-step GP model and long-horizon policy search, but modernizes policy evaluation:

- one-step dynamics are exact GPs trained with `gpytorch`
- policy search uses PyTorch autograd through particle rollouts
- predictive uncertainty is estimated from particles instead of re-implementing all MATLAB moment-matching derivatives

That keeps the code readable while preserving the core data-efficient model-based workflow.

## Project layout

- [`src/pilco_py/env.py`](/Users/weiliu/code/pilco-py/src/pilco_py/env.py): pendulum simulator and rollout collection
- [`src/pilco_py/gpytorch_dynamics.py`](/Users/weiliu/code/pilco-py/src/pilco_py/gpytorch_dynamics.py): independent GP dynamics model fit in GPyTorch and queried inside PyTorch rollouts
- [`src/pilco_py/policy.py`](/Users/weiliu/code/pilco-py/src/pilco_py/policy.py): squashed and positive-output PyTorch policies
- [`src/pilco_py/trainer.py`](/Users/weiliu/code/pilco-py/src/pilco_py/trainer.py): end-to-end pendulum training loop
- [`src/pilco_py/visualization.py`](/Users/weiliu/code/pilco-py/src/pilco_py/visualization.py): pendulum learning curves, diagnostics, and rendering
- [`src/pilco_py/assistive_env.py`](/Users/weiliu/code/pilco-py/src/pilco_py/assistive_env.py): 1-DoF exoskeleton-style shared-control simulation
- [`src/pilco_py/assistive_trainer.py`](/Users/weiliu/code/pilco-py/src/pilco_py/assistive_trainer.py): Hamaya-style assistive learning loop
- [`src/pilco_py/assistive_visualization.py`](/Users/weiliu/code/pilco-py/src/pilco_py/assistive_visualization.py): assistive rollout, policy-map, and interaction-model visualizations

## Package management with uv

Create the environment and install dependencies with `uv`:

```bash
uv sync
```

Run the pendulum experiment:

```bash
uv run pilco-train --experiment pendulum --episodes 8 --policy-steps 200
```

Run the assistive extension:

```bash
uv run pilco-train --experiment assistive --episodes 10 --initial-rollouts 5 --policy-steps 200
```

Faster smoke runs:

```bash
uv run pilco-train --experiment pendulum --episodes 2 --policy-steps 20 --particles 32 --eval-particles 64 --gp-train-steps 25 --skip-video
uv run pilco-train --experiment assistive --episodes 2 --initial-rollouts 2 --policy-steps 20 --particles 32 --eval-particles 64 --gp-train-steps 25 --skip-video
```

## Artifacts

Pendulum training writes to `artifacts/pendulum/` by default:

- `training_curve.png`: overall learning progress
- `episode_XX_diagnostics.png`: predicted versus realized rollout
- `episode_XX_tip_path.png`: pendulum-tip geometry plot
- `episode_XX.gif`: MuJoCo playback
- `history.json`: per-episode summary

Assistive training writes to `artifacts/assistive/` by default:

- `training_curve.png`: long-term EMG-cost learning curve
- `episode_XX_diagnostics.png`: reference tracking, pressure, and EMGs
- `episode_XX_policy_maps.png`: learned assistive policy map and predicted EMG interaction maps
- `episode_XX.gif`: rollout animation
- `history.json`: per-trial summary

## Source references

The reconstructions were based on:

- the pendulum scenario files, especially `settings_pendulum.m`, `dynamics_pendulum.m`, `loss_pendulum.m`, and `pendulum_learn.m`
- the paper [`doc/Deisenroth et al. - 2015 - Gaussian Processes for Data-Efficient Learning in Robotics and Control.pdf`](./doc/Deisenroth%20et%20al.%20-%202015%20-%20Gaussian%20Processes%20for%20Data-Efficient%20Learning%20in%20Robotics%20and%20Control.pdf)
- the paper [`doc/Hamaya et al. - 2017 - Learning assistive strategies for exoskeleton robots from user-robot physical interaction.pdf`](./doc/Hamaya%20et%20al.%20-%202017%20-%20Learning%20assistive%20strategies%20for%20exoskeleton%20robots%20from%20user-robot%20physical%20interaction.pdf)
