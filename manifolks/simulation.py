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
import matplotlib.pyplot as plt

# Simulation Parameters
RANDOM_SEED = 2
DT = 0.01 # intergration time step
T = 2 # total simulation time
TIME = np.arange(0, T, DT) # time vector
T_STEPS = len(TIME) # number of time steps
TARGETS = 4 # number of radial directions
TARGET_MAX_RADIUS = 0.2 # max reachout distance
PULSE_LENGTH = int(TARGET_MAX_RADIUS / DT) 
N_COMPONENTS = 4  # reduced dimensionality for decoding

# Network Parameters
N = 50 # number of neurons
G = 1.5  # recurrent coupling strength
P = 0.2  # connection probability
TAU = 0.01 # time constant for synaptic dynamics

# Training Parameters
N_TRIALS_TRAIN = 80  # number of trials for initial training
N_TRIALS_RELEARN = 80  # number of trials for relearning
N_TRIALS_BCI_MANIFOLD = 50  # number of trials for BCI manifold computation
LEARNING_RATE = 20. # learning rate for all network trainings


# Output Directory
sim_name = f"N{N}_g{G}_p{P}_tau{TAU}_T{T}_dt{DT}_Ncomp{N_COMPONENTS}_seed{RANDOM_SEED}_dir{TARGETS}"
OUT_DIR = Path(__file__).parents[1] / "data" / "sims" / "perfect_feedback" / sim_name
OUT_DIR.mkdir(exist_ok=True, parents=True)

def main():
    np.random.seed(RANDOM_SEED)


    stim = create_stim_center_reach_out(T_STEPS, PULSE_LENGTH, TARGETS)
    target_traj = create_target_trajectories_center_reach_out(
        T_STEPS, PULSE_LENGTH, TARGETS, TARGET_MAX_RADIUS
    )
    
    # print simulation parameters
    print(f"Simulation Parameters:\n"
          f"  Time: {T} s, Time Steps: {T_STEPS}, DT: {DT} s\n"
          f"  Target Radius: {TARGET_MAX_RADIUS}, Pulse Length: {PULSE_LENGTH}\n"
          f"  Neurons: {N}, Coupling Strength: {G}, Connection Probability: {P}, Tau: {TAU}\n"
          f"  Trials (Train): {N_TRIALS_TRAIN}, Trials (Relearn): {N_TRIALS_RELEARN}, "
          f"Trials (BCI Manifold): {N_TRIALS_BCI_MANIFOLD}\n")

    print("=== Phase 1: Initial Network Training ===")
    network = RNN(N, g=G, p=P, N_in=TARGETS, dt=DT, tau=TAU)
    network.save(OUT_DIR / f"rnn-N{N}-g{G}-p{P}-tau{TAU}")
    w0 = network.W.copy()  # initial weights

    # Network learns to reach out by getting feedback from a randomly initialized decoder
    decoder = Decoder(N=N, N_out=2, reduced_dim=N_COMPONENTS)
    decoder.set_initial_weights(TARGET_MAX_RADIUS)
    bci_w0 = decoder.W.copy()  # initial decoder weights
    rnn_loss_traj_0 = network.train(
        N_TRIALS_TRAIN, stim, PULSE_LENGTH, decoder, target_traj, lr=LEARNING_RATE
    )
    print(f"Initial loss: mean({np.mean(rnn_loss_traj_0):.4f}, std({np.std(rnn_loss_traj_0):.4f})")
    w1 = network.W.copy()  # "stabilized" weights after initial learning

    print("=== BCI Decoder Training ===")
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
        N_out=2,
        reduced_dim=N_COMPONENTS,
        evectors=manifold_dict["evectors"],
        decoder_weights=decoder.W,
    )
    perturbations = perturber.find_balanced_perturbations(
        manifold_dict["activity"],
        target_traj[:, PULSE_LENGTH:, :],
        manifold_dict["trial_order"],
    )

    print("=== Phase 2: Within Manifold Pertubations ===")
    # Within manifold perturbation (WMP)
    wmp_decoder = Decoder(N, N_out=2, reduced_dim=N_COMPONENTS, W=perturbations["within_manifold_perturbation"])
    network.W = w1.copy()  # reset to learned stabilized weights
    rnn_loss_wmp = network.train(
        N_TRIALS_RELEARN, stim, PULSE_LENGTH, wmp_decoder, target_traj, lr=LEARNING_RATE)
    w2_wmp = network.W.copy()  # weights after within manifold perturbation relearning

    wmp_manifold_dict = network.compute_manifold(N_TRIALS_BCI_MANIFOLD, stim, PULSE_LENGTH)
    result = wmp_decoder.decode(wmp_manifold_dict)
    rnn_loss_wmp_bci = cost(
        result, target_traj[:, PULSE_LENGTH:, :], wmp_manifold_dict["trial_order"]
    )

    print("=== Phase 3: Outside Manifold Perturbations ===")
    # Outside manifold perturbation (OMP)
    omp_decoder = Decoder(N, N_out=2, reduced_dim=N_COMPONENTS, W=perturbations["outside_manifold_perturbation"])
    network.W = w1.copy()  # reset to learned stabilized weights
    rnn_loss_omp = network.train(
        N_TRIALS_RELEARN, stim, PULSE_LENGTH, omp_decoder, target_traj, lr=LEARNING_RATE)
    w2_omp = network.W.copy()  # weights after outside manifold perturbation relearning

    omp_manifold_dict = network.compute_manifold(N_TRIALS_BCI_MANIFOLD, stim, PULSE_LENGTH)
    result = omp_decoder.decode(omp_manifold_dict)
    rnn_loss_omp_bci = cost(
        result, target_traj[:, PULSE_LENGTH:, :], omp_manifold_dict["trial_order"])

    # Save results: network weights, decoder weights, losses, and manifold data
    np.savez(
        OUT_DIR / "simulation_results.npz",
        w0=w0,
        w1=w1,
        w2_wmp=w2_wmp,
        w2_omp=w2_omp,
        bci_w0=bci_w0,
        rnn_loss_traj_0=rnn_loss_traj_0,
        rnn_loss_bci=rnn_loss_bci,
        rnn_loss_wmp=rnn_loss_wmp,
        rnn_loss_wmp_bci=rnn_loss_wmp_bci,
        rnn_loss_omp=rnn_loss_omp,
        rnn_loss_omp_bci=rnn_loss_omp_bci,
        manifold_dict=manifold_dict,
        wmp_manifold_dict=wmp_manifold_dict,
        omp_manifold_dict=omp_manifold_dict,
        perturbations=perturbations,
        stim=stim,
        target_traj=target_traj,
        time=TIME,
        pulse_length=PULSE_LENGTH,
    )

    # Save Visualizations
    print("=== Saving Visualizations ===")
    plot_network_weights(w0, w1, w2_wmp, w2_omp)
    plot_loss_trajectories(
        rnn_loss_traj_0, rnn_loss_wmp, rnn_loss_omp
    )
    plot_decoder_weights(bci_w0, decoder.W, wmp_decoder.W, omp_decoder.W)


def plot_network_weights(initial_weights, stabilized_weights, wmp_weights, omp_weights):
    """Plot network weights across different phases"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Network Weight Evolution', fontsize=16, fontweight='bold')
    
    weight_data = [
        (initial_weights, 'Initial Weights'),
        (stabilized_weights, 'After Initial Training'),
        (wmp_weights, 'After WMP Training'),
        (omp_weights, 'After OMP Training')
    ]
    
    vmin = min(w.min() for w, _ in weight_data)
    vmax = max(w.max() for w, _ in weight_data)
    
    for i, (weights, title) in enumerate(weight_data):
        row, col = i // 2, i % 2
        im = axes[row, col].imshow(weights, cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axes[row, col].set_title(title, fontweight='bold')
        axes[row, col].set_xlabel('Neuron Index')
        axes[row, col].set_ylabel('Neuron Index')
    
    fig.colorbar(im, ax=axes, shrink=0.8, label='Weight Strength')
    plt.savefig(OUT_DIR / 'network_weights_evolution.png', dpi=300, bbox_inches='tight')
    print("Network weights plotted and saved.")
    plt.close()


def plot_decoder_weights(initial_decoder_weights, trained_decoder_weights, wmp_decoder_weights, omp_decoder_weights):
    """Plot decoder weights across different phases"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Decoder Weight Evolution', fontsize=16, fontweight='bold')
    
    decoder_data = [
        (initial_decoder_weights, 'Initial Decoder'),
        (trained_decoder_weights, 'After BCI Training'),
        (wmp_decoder_weights, 'Within-Manifold Perturbation'),
        (omp_decoder_weights, 'Outside-Manifold Perturbation')
    ]
    
    vmin = min(w.min() for w, _ in decoder_data)
    vmax = max(w.max() for w, _ in decoder_data)
    
    for i, (weights, title) in enumerate(decoder_data):
        row, col = i // 2, i % 2
        im = axes[row, col].imshow(weights, cmap='RdBu_r', vmin=vmin, vmax=vmax, aspect='auto')
        axes[row, col].set_title(title, fontweight='bold')
        axes[row, col].set_xlabel('Neuron Index')
        axes[row, col].set_ylabel('Output (Readout)')
    
    fig.colorbar(im, ax=axes, shrink=0.8, label='Weight Strength')
    plt.savefig(OUT_DIR / 'decoder_weights_evolution.png', dpi=300, bbox_inches='tight')
    print("Decoder weights plotted and saved.")
    plt.close()



def plot_loss_trajectories(initial_losses, wmp_losses, omp_losses):
    """Plot learning loss trajectories"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Learning Loss Trajectories', fontsize=16, fontweight='bold')
    
    axes[0].plot(initial_losses, 'b-', linewidth=2)
    axes[0].set_title('Initial Training', fontweight='bold')
    axes[0].set_xlabel('Trial')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(wmp_losses, 'g-', linewidth=2)
    axes[1].set_title('Within-Manifold Perturbation', fontweight='bold')
    axes[1].set_xlabel('Trial')
    axes[1].set_ylabel('Loss')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(omp_losses, 'r-', linewidth=2)
    axes[2].set_title('Outside-Manifold Perturbation', fontweight='bold')
    axes[2].set_xlabel('Trial')
    axes[2].set_ylabel('Loss')
    axes[2].grid(True, alpha=0.3)
    
    plt.savefig(OUT_DIR / 'loss_trajectories.png', dpi=300, bbox_inches='tight')
    print("Loss trajectories plotted and saved.")
    plt.close()


if __name__ == "__main__":
    main()