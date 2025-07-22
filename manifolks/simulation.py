"""
Simulation using Perfect Feedback Signal (Fig 2)
"""

# %% import packages
import numpy as np
from pathlib import Path
from manifolks.decoder import Decoder, ManifoldPerturbation
from manifolks.task import (
    create_stim_center_reach_out,
    create_target_trajectories_center_reach_out,
    cost,
    TrajectoryType,
)
from manifolks.rnn import RNN

# Output Directory
OUT_DIR = Path(__file__).parents[1] / "data"
OUT_DIR.mkdir(exist_ok=True, parents=True)

# PARAMETERS
RANDOM_SEED = 2
DT = 0.01
T = 2
TIME = np.arange(0, T, DT)
T_STEPS = len(TIME)
TARGETS = 6
TRAJECTORY_TYPE: TrajectoryType = "constant"
TARGET_MAX_RADIUS = 0.2
PULSE_LENGTH = int(TARGET_MAX_RADIUS / DT)
# network specifications
N = 800
G = 1.5  # recurrent coupling strength
P = 0.1  # connection probability
TAU = 0.1
# initial network learning
N_TRIALS_LEARN = 80
LEARNING_RATE = 20.0
# BCI manifold calculation
N_TRIALS_BCI_MANIFOLD = 50
N_COMPONENTS = 10
# network relearning after perturbations
N_TRIALS_RELEARN = 80
LEARNING_RATE_REC = 20.0


if __name__ == "__main__":
    
    np.random.seed(RANDOM_SEED)

    stim = create_stim_center_reach_out(T_STEPS, PULSE_LENGTH, TARGETS)
    target_traj = create_target_trajectories_center_reach_out(
        T_STEPS, PULSE_LENGTH, TARGETS, TARGET_MAX_RADIUS, TRAJECTORY_TYPE
    )

    network = RNN(N, g=G, p=P, N_in=TARGETS, dt=DT, tau=TAU)
    network.save(OUT_DIR / "rnn")
    w0 = network.W.copy()  # initial weights

    # Network learns to reach out by getting feedback from a randomly initialized decoder
    decoder = Decoder(N=N, N_out=2, reduced_dim=N_COMPONENTS)
    bci_w0 = decoder.W.copy()  # initial decoder weights
    rnn_loss_traj_0 = network.train(
        N_TRIALS_LEARN, stim, PULSE_LENGTH, decoder, target_traj, lr=LEARNING_RATE
    )
    print(f"Initial loss: {rnn_loss_traj_0[-1]:.4f}")
    w1 = network.W.copy()  # "stabilized" weights after initial learning

    manifold_dict = network.compute_manifold(N_TRIALS_BCI_MANIFOLD, stim, PULSE_LENGTH)
    bci_loss = decoder.train(
        manifold_dict["xi"][
            :, :, :N_COMPONENTS
        ],  # !! notice the use of only first n_dims of projected data
        target_traj[:, PULSE_LENGTH:, :],
        manifold_dict["trial_order"],
    )
    print(f"BCI loss after initial learning: {bci_loss:.4f}")

    result = decoder.decode(manifold_dict)
    rnn_loss_bci = cost(
        result, target_traj[:, PULSE_LENGTH:, :], manifold_dict["trial_order"]
    )

    perturber = ManifoldPerturbation(
        N,
        reduced_dim=N_COMPONENTS,
        evectors=manifold_dict["evectors"],
        decoder_weights=decoder,
    )
    perturbations = perturber.find_balanced_perturbations()

    # print perturbation costs
    print(f"Perturbation costs: {perturbations['costs'][0]}")

    # Within manifold perturbation (WMP)
    wmp_decoder = Decoder(N, N_out=2, reduced_dim=N_COMPONENTS, W=perturbations["within_manifold_perturbation"])
    network.W = w1.copy()  # reset to learned stabilized weights
    rnn_loss_wmp = network.train(
        N_TRIALS_RELEARN, stim, PULSE_LENGTH, wmp_decoder, target_traj, lr=LEARNING_RATE_REC)
    w2_wmp = network.W.copy()  # weights after within manifold perturbation relearning

    wmp_manifold_dict = network.compute_manifold(N_TRIALS_BCI_MANIFOLD, stim, PULSE_LENGTH)
    result = wmp_decoder.decode(wmp_manifold_dict)
    rnn_loss_wmp_bci = cost(
        result, target_traj[:, PULSE_LENGTH:, :], wmp_manifold_dict["trial_order"]
    )

    # Outside manifold perturbation (OMP)
    omp_decoder = Decoder(N, N_out=2, reduced_dim=N_COMPONENTS, W=perturbations["outside_manifold_perturbation"])
    network.W = w1.copy()  # reset to learned stabilized weights
    rnn_loss_omp = network.train(
        N_TRIALS_RELEARN, stim, PULSE_LENGTH, omp_decoder, target_traj, lr=LEARNING_RATE_REC)
    w2_omp = network.W.copy()  # weights after outside manifold perturbation relearning

    omp_manifold_dict = network.compute_manifold(N_TRIALS_BCI_MANIFOLD, stim, PULSE_LENGTH)
    result = omp_decoder.decode(omp_manifold_dict)
    rnn_loss_omp_bci = cost(
        result, target_traj[:, PULSE_LENGTH:, :], omp_manifold_dict["trial_order"])
    



