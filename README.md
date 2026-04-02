# PILCO Pendulum in Modern Python

This repo rebuilds the original `pilcoV0.9` pendulum example in readable Python instead of porting the MATLAB code line-by-line. The implementation keeps the original ingredients that matter for the experiment:

- a pendulum plant with the same state convention as the old code: `[dtheta, theta]`
- GP dynamics learned on state differences from observed transitions
- a bounded policy optimized against long-horizon predicted cost
- a saturating pendulum-tip cost matching the geometry used in the original paper
- saved diagnostics and a MuJoCo animation of learned rollouts

The code uses:

- `numpy` for simulation and dataset handling
- `gpytorch` for fitting the one-step GP dynamics model
- `torch` for differentiable policy optimization through particle rollouts
- `mujoco` for pendulum playback renders

## What is faithful and what is modernized

The original MATLAB PILCO code computes analytic uncertain-input GP predictions and analytic policy gradients. Recreating every derivative helper from `pilcoV0.9` would turn this repo back into a dense symbolic port.

This version keeps the one-step GP model and long-horizon policy search, but modernizes policy evaluation:

- one-step dynamics are exact GPs trained with `gpytorch`
- policy search uses PyTorch autograd through particle rollouts
- predictive uncertainty is estimated from particles instead of re-implementing all MATLAB moment-matching derivatives

That makes the code much easier to read and extend while staying close to the original pendulum workflow.

## Project layout

- [`src/pilco_py/env.py`](/Users/weiliu/code/pilco-py/src/pilco_py/env.py): pendulum simulator and rollout collection
- [`src/pilco_py/gpytorch_dynamics.py`](/Users/weiliu/code/pilco-py/src/pilco_py/gpytorch_dynamics.py): independent GP dynamics model fit in GPyTorch and queried inside PyTorch rollouts
- [`src/pilco_py/policy.py`](/Users/weiliu/code/pilco-py/src/pilco_py/policy.py): squashed PyTorch policy
- [`src/pilco_py/trainer.py`](/Users/weiliu/code/pilco-py/src/pilco_py/trainer.py): end-to-end training loop
- [`src/pilco_py/visualization.py`](/Users/weiliu/code/pilco-py/src/pilco_py/visualization.py): learning curves, diagnostics, and MuJoCo rendering

## Package management with uv

Create the environment and install dependencies with `uv`:

```bash
uv sync
```

Run training:

```bash
uv run pilco-train --episodes 8 --policy-steps 200
```

Faster smoke run:

```bash
uv run pilco-train --episodes 2 --policy-steps 20 --particles 32 --eval-particles 64 --gp-train-steps 25 --skip-video
```

## Artifacts

Training writes to `artifacts/pendulum/` by default:

- `training_curve.png`: overall learning progress
- `episode_XX_diagnostics.png`: predicted versus realized rollout
- `episode_XX_tip_path.png`: pendulum-tip geometry plot
- `episode_XX.gif`: MuJoCo playback
- `history.json`: per-episode summary

## Source references

The reconstruction was based on:

- original MATLAB code in `/Users/weiliu/code/pilcoV0.9`
- the pendulum scenario files, especially `settings_pendulum.m`, `dynamics_pendulum.m`, `loss_pendulum.m`, and `pendulum_learn.m`
- the paper [`doc/Deisenroth et al. - 2015 - Gaussian Processes for Data-Efficient Learning in Robotics and Control.pdf`](/Users/weiliu/code/pilco-py/doc/Deisenroth%20et%20al.%20-%202015%20-%20Gaussian%20Processes%20for%20Data-Efficient%20Learning%20in%20Robotics%20and%20Control.pdf)
