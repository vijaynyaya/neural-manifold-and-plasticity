# %%
import numpy as np
from typing import Literal

# %%
def create_stim_center_reach_out(
    tsteps: int,
    pulse_steps: int,
    n_targets: int = 6, # number of directions
    amplitude: float = 1.0
) -> np.ndarray:
  """
  Create a stimulus matrix for center-out reaching task.
  Stimulus is a one-hot encoding of target directions, presented at the
  start of the trial for a set number of time steps (pulse_steps).
  Parameters:
  * tsteps: total number of time steps in the trial
  * pulse_steps: number of time steps to present the stimulus
  * n_targets: number of target directions (default is 6 for center-out)
  * amplitude: amplitude of the stimulus (default is 1.0)
  Returns:
  * stim: stimulus matrix of shape (n_targets, tsteps, n_targets)
  """
  stim = np.zeros((n_targets, tsteps, n_targets))
  for i in range(n_targets):
    stim[i, :pulse_steps, i] = amplitude
  return stim


# %%
TrajectoryType = Literal[
  "linear", # linear increase in radius after pulse
  "normal", # normal distribution increase in radius after pulse
  "constant" # constant increase in radius after pulse
] 
def create_target_trajectories_center_reach_out(
    tsteps: int,
    pulse_steps: int,
    n_targets: int = 6, # number of directions
    target_max: float = .2,
    trajectory_type: TrajectoryType = "constant"
) -> np.ndarray:
  """
  Create target trajectories for center-out reaching task.
  Targets are positioned at equal angles around a circle, with a radial
  distance that increases after the pulse period.
  Parameters:
  * tsteps: total number of time steps in the trial
  * pulse_steps: number of time steps to present the stimulus
  * n_targets: number of target directions (default is 6 for center-out)
  * target_max: maximum radial distance of the targets (default is 0.2)
  * trajectory_type: type of trajectory to generate
    - "linear": linear increase in radius after pulse
    - "normal": normal distribution increase in radius after pulse
    - "constant": constant increase in radius after pulse
  Returns:
  * trajectories: target trajectories of shape (n_targets, tsteps, 2)
    where the last dimension contains x and y coordinates.
  """
  # Angles for center-out reaching task, equally spaced around a circle
  angles = np.linspace(0, 2 * np.pi, n_targets, endpoint=False)

  # Radial distance over time (0 during pulse, then increasing)
  radius = np.zeros(tsteps)
  match trajectory_type:
    case "linear":
      # Linear increase in radius after pulse
      radius[pulse_steps:] = np.linspace(0, target_max, tsteps - pulse_steps)
    case "normal":
      # Normal distribution increase in radius after pulse
      xx = np.linspace(0, target_max, tsteps - pulse_steps)
      mu = target_max / 2
      sigma = target_max / 8
      radius[pulse_steps:] = target_max * np.exp(-((xx - mu) ** 2) / (2 * sigma ** 2))
    case "constant":
      # Constant radius after pulse
      radius[:pulse_steps] = target_max
  # Create target trajectories for each direction
  trajectories = np.zeros((n_targets, tsteps, 2))
  for i, angle in enumerate(angles):
    trajectories[i, :, 0] = radius * np.cos(angle)  # x-coordinate
    trajectories[i, :, 1] = radius * np.sin(angle)  # y-coordinate
  return trajectories

# %%
def plot_target_trajectory(trajectory: np.ndarray, target_idx: int = 0, ax=None):
  """
  Plot a single target trajectory (x vs y) for the center-out reaching task.

  Parameters:
  * trajectory: np.ndarray of shape (n_targets, tsteps, 2)
  * target_idx: index of the target to plot (default 0)
  * ax: matplotlib axis to plot on (optional)
  """
  import matplotlib.pyplot as plt

  if ax is None:
    fig, ax = plt.subplots()
  traj = trajectory[target_idx]
  ax.plot(traj[:, 0], traj[:, 1], marker='o')
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_title(f'Target Trajectory {target_idx}')
  ax.axis('equal')
  return ax

# %%
def cost(result: np.ndarray, target_traj: np.ndarray, trial_order: np.ndarray) -> float:
  """
  Calculate the cost (mean squared error) between the network output and target trajectories.

  Parameters:
  * result: np.ndarray of shape (trials, tsteps, n_outputs)
    Network output for each trial
  * target_traj: np.ndarray of shape (n_targets, tsteps, n_targets)
    Target trajectories for each trial
  * trial_order: np.ndarray of shape (trials,)
    Order of trials mapping each trial to its corresponding target

  Returns:
  * cost: float
    Mean squared error across all trials
  """
  total_cost = 0.0
  trials = len(result.shape[0])
  for t in range(trials):
    target_cue_idx = trial_order[t]
    error = result[t, :, :] - target_traj[target_cue_idx, :, :]
    total_cost += np.mean(error ** 2)
  return total_cost