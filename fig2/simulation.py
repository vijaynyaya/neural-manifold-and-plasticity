"""
Simulation using Perfect Feedback Signal (Fig 2)
"""
# %% import packages
import numpy as np
from pathlib import Path
from .decoder import Decoder
from .task import (
  create_stim_center_reach_out,
  create_target_trajectories_center_reach_out,
  cost,
  TrajectoryType
)
from .rnn import RNN

# Output Directory
OUT_DIR = Path(__file__).parents[1] / 'data'
OUT_DIR.mkdir(exist_ok=True, parents=True)

# PARAMETERS
RANDOM_SEED = 2
DT = 0.01
T = 2
TIME = np.arange(0, T, DT)
T_STEPS =  len(TIME)
TARGETS = 6
TRAJECTORY_TYPE: TrajectoryType = "constant"
TARGET_MAX_RADIUS = 0.2
PULSE_LENGTH = int(TARGET_MAX_RADIUS / DT)
# network specifications
N = 800
G = 1.5 # recurrent coupling strength
P = 0.1 # connection probability
TAU = 0.1
# initial network learning
N_TRIALS_LEARN = 80
LEARNING_RATE = 20.
# BCI manifold calculation
N_TRIALS_BCI_MANIFOLD = 50
N_COMPONENTS = 10
# network relearning after perturbations
N_TRIALS_RELEARN = 80
LEARNING_RATE_REC = 20.


def main():
  np.random.seed(RANDOM_SEED)

  stim = create_stim_center_reach_out(T_STEPS, PULSE_LENGTH, TARGETS)
  target_traj = create_target_trajectories_center_reach_out(T_STEPS, PULSE_LENGTH, TARGETS, TARGET_MAX_RADIUS, TRAJECTORY_TYPE)

  network = RNN(N, g=G, p=P, N_in=TARGETS, dt=DT, tau=TAU)
  network.save(OUT_DIR / 'rnn')
  w0 = network.W.copy()  # initial weights

  # Network learns to reach out by getting feedback from a randomly initialized decoder
  decoder = Decoder(N=N, N_out=2, reduced_dim=N_COMPONENTS)
  bci_w0 = decoder.W.copy()  # initial decoder weights
  rnn_loss_traj_0 = network.train(N_TRIALS_LEARN, stim, PULSE_LENGTH, decoder, target_traj, lr=LEARNING_RATE)
  print(f"Initial loss: {rnn_loss_traj_0[-1]:.4f}")
  w1 = network.W.copy()  # weights after initial learning

  manifold_dict = network.compute_manifold(N_TRIALS_BCI_MANIFOLD, stim, PULSE_LENGTH)
  bci_loss = decoder.train(
    manifold_dict["activity"],
    target_traj[:,PULSE_LENGTH:,:],
    manifold_dict["trial_order"]
  )
  print(f"BCI loss after initial learning: {bci_loss:.4f}")

  result = decoder.decode(N_COMPONENTS, manifold_dict)
  rnn_loss_bci = cost(result, target_traj[:,PULSE_LENGTH:,:], manifold_dict["trial_order"])

  


