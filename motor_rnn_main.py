"""
Motor RNN Reaching Task Experiment
==================================
This script implements a reaching task experiment using a Recurrent Neural Network (RNN).
The RNN learns to reach to 6 different targets in 2D space when given directional cues.

Experiment Overview:
1. Create reaching task stimuli (6 directional cues)
2. Define target trajectories for each direction
3. Build and train an RNN to perform the reaching task
4. Extract the neural manifold (low-dimensional representation)
5. Train a linear decoder to read out reaching trajectories
6. Analyze and visualize the results

Based on motor cortex reaching experiments where:
- A cue indicates which of 6 targets to reach to
- The network must generate appropriate x,y trajectories
- The decoder reads out motor commands from neural activity
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from motor_rnn_utils import *

# ============================================================================
# 1. EXPERIMENT CONFIGURATION
# ============================================================================

print("=" * 60)
print("MOTOR RNN REACHING EXPERIMENT")
print("=" * 60)

# Create project directories
proj_path = "proj_rnn/"
savedir = os.path.join(proj_path, 'data/fig2/')
for path in [proj_path, savedir]:
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")

# Set random seed for reproducibility
SEED = 2
np.random.seed(SEED)
print(f"\nRandom seed set to: {SEED}")

# ============================================================================
# 2. TEMPORAL PARAMETERS
# ============================================================================
print("\n" + "="*60)
print("TEMPORAL PARAMETERS")
print("="*60)

# Time discretization and trial structure
dt = 0.01                   # Time step in seconds (10ms bins)
T = 2.0                     # Total trial duration in seconds
time = np.arange(0, T, dt)  # Time vector
tsteps = len(time)          # Total number of time steps (200)

# Stimulus presentation
pulse_duration = 0.2        # How long the directional cue is shown (seconds)
pulse_length = int(pulse_duration/dt)  # Convert to timesteps (20 steps)

print(f"Time step (dt): {dt*1000:.0f} ms")
print(f"Trial duration: {T} seconds")
print(f"Total timesteps: {tsteps}")
print(f"Cue duration: {pulse_duration*1000:.0f} ms ({pulse_length} timesteps)")

# ============================================================================
# 3. NETWORK ARCHITECTURE PARAMETERS
# ============================================================================
print("\n" + "="*60)
print("NETWORK ARCHITECTURE")
print("="*60)

# RNN parameters
N = 800                     # Number of recurrent units
g = 1.5                     # Synaptic strength scaling (chaos parameter)
p = 0.1                     # Connection probability (10% connectivity)
tau = 0.1                   # Neural time constant (100ms)

# Input/Output configuration
N_INPUT_TARGETS = 6         # Number of possible reaching directions
N_OUTPUT_UNITS = 2          # Output dimensions (x and y coordinates)

print(f"Recurrent units: {N}")
print(f"Connectivity: {p*100:.0f}% ({int(p*N)} connections per neuron)")
print(f"Synaptic strength (g): {g} {'(chaotic regime)' if g > 1 else '(stable regime)'}")
print(f"Time constant (tau): {tau*1000:.0f} ms")
print(f"Input channels: {N_INPUT_TARGETS} (one per target)")
print(f"Output channels: {N_OUTPUT_UNITS} (x, y coordinates)")

# ============================================================================
# 4. TASK PARAMETERS
# ============================================================================
print("\n" + "="*60)
print("TASK PARAMETERS")
print("="*60)

# Reaching task configuration
stimulus_type = 'constant'  # How targets move ('constant', 'linear', 'normal')
target_max = 0.2           # Maximum reach distance (in normalized units)

# Training parameters
n_initial_training = 80    # Number of trials for initial learning
initial_learning_rate = 20.0  # FORCE learning rate parameter

# Analysis parameters
n_manifold_trials = 50     # Trials used to compute neural manifold
manifold_dimension = 10    # Reduced dimensionality for decoder

print(f"Target arrangement: {N_INPUT_TARGETS} targets in circle")
print(f"Target distance: {target_max} (normalized units)")
print(f"Movement type: {stimulus_type}")
print(f"Initial training: {n_initial_training} trials")
print(f"Manifold analysis: {n_manifold_trials} trials")
print(f"Manifold dimension: {manifold_dimension}D")

# ============================================================================
# 5. CREATE TASK STIMULI AND TARGETS
# ============================================================================
print("\n" + "="*60)
print("CREATING TASK STIMULI AND TARGETS")
print("="*60)

# Create stimuli: brief pulses indicating which target to reach to
# Shape: (6 targets, 200 timesteps, 6 input channels)
# Each stimulus has a pulse in one channel for the first 20 timesteps
print("\nCreating directional cue stimuli...")
stimulus = create_reaching_task_stimuli(
    tsteps=tsteps, 
    pulse_steps=pulse_length, 
    n_targets=N_INPUT_TARGETS,
    twod=False  # Use one-hot encoding instead of 2D coordinates
)
print(f"Stimulus shape: {stimulus.shape}")
print("- Each trial starts with a {:.0f}ms pulse in one input channel".format(pulse_duration*1000))

# Visualize the stimulus structure
plot_reaching_task_stimuli(stimulus, N_INPUT_TARGETS, tsteps, T)

# Create target trajectories: where the hand should move for each cue
# Shape: (6 targets, 200 timesteps, 2 coordinates)
print("\nCreating target reaching trajectories...")
target = create_reaching_task_targets(
    tsteps=tsteps,
    pulse_steps=pulse_length,
    n_targets=N_INPUT_TARGETS,
    stype=stimulus_type,
    target_max=target_max
)
print(f"Target shape: {target.shape}")
print("- Targets arranged in circle at radius {:.2f}".format(target_max))
print("- Movement begins after cue offset")

# Visualize the target trajectories
plot_reaching_task_targets(target, tsteps, T)

# ============================================================================
# 6. BUILD AND TRAIN THE NETWORK
# ============================================================================
print("\n" + "="*60)
print("BUILDING AND TRAINING RNN")
print("="*60)

# Initialize the RNN
print("\nInitializing RNN...")
reaching_network = RNN(N=N, g=g, p=p, tau=tau, dt=dt, N_in=N_INPUT_TARGETS)
print(f"Created RNN with {reaching_network.N} neurons")
print(f"Actual connectivity: {reaching_network.K} connections per neuron")

# Save initial network configuration
save_RNN(reaching_network, savedir)
print(f"Saved initial network to: {savedir}")

# Create decoder: maps RNN activity to (x,y) output
print("\nCreating output decoder...")
reaching_decoder = create_reaching_task_decoder(
    reaching_network,
    n_output_units=N_OUTPUT_UNITS
)
print(f"Decoder shape: {reaching_decoder.shape}")

# Create feedback weights: allows output error to train the RNN
print("\nCreating feedback connections...")
reaching_feedback = get_feedback_weights(reaching_decoder)
print(f"Feedback shape: {reaching_feedback.shape}")

# Train the network using FORCE learning
print(f"\nTraining network for {n_initial_training} trials...")
print("Using FORCE (First-Order Reduced and Controlled Error) learning")
print("-" * 40)

reaching_loss = reaching_network.relearn(
    trials=n_initial_training,
    ext=stimulus,
    ntstart=pulse_length,
    decoder=reaching_decoder,
    feedback=reaching_feedback,
    target=target,
    delta=initial_learning_rate
)

# Save trained weights
np.save(f'{savedir}W_stabilized_reaching', reaching_network.W)
print(f"\nSaved trained weights to: {savedir}W_stabilized_reaching.npy")
print(f"Final loss: {reaching_loss[-1]:.5f}")

# ============================================================================
# 7. MANIFOLD ANALYSIS
# ============================================================================
print("\n" + "="*60)
print("NEURAL MANIFOLD ANALYSIS")
print("="*60)
print("Extracting low-dimensional structure from neural activity...")

# Compute the neural manifold using PCA on network activity
print(f"\nRunning {n_manifold_trials} trials to sample neural activity...")
manifold_out = get_manifold(
    network=reaching_network,
    trials=n_manifold_trials,
    stimulus=stimulus,
    pulse_length=pulse_length
)

print("\nManifold statistics:")
print(f"- Total variance captured: {np.sum(manifold_out['ev'][:manifold_dimension]):.1f}")
print(f"- Participation ratio: {manifold_out['pr']} dimensions")
print(f"- Activity shape: {manifold_out['activity'].shape}")
print(f"- Reduced activity shape: {manifold_out['xi2'].shape}")

# ============================================================================
# 8. TRAIN LINEAR DECODER
# ============================================================================
print("\n" + "="*60)
print("TRAINING LINEAR DECODER")
print("="*60)
print(f"Training decoder on {manifold_dimension}D manifold representation...")

# Train a linear decoder on the low-dimensional manifold
# This tests if the manifold preserves task-relevant information
W_bci, decoder_loss = train_reaching_decoder(
    inputP=manifold_out["xi2"][:, :, :manifold_dimension],  # Use only top PCs
    target=target[:, pulse_length:, :],  # Targets after cue period
    order=manifold_out["order"],  # Trial order used in manifold calculation
    n_output_units=N_OUTPUT_UNITS
)

print(f"\nDecoder performance:")
print(f"- Decoder weights shape: {W_bci.shape}")
print(f"- Mean squared error: {decoder_loss:.6f}")

# ============================================================================
# 9. TRANSFORM AND ANALYZE RESULTS
# ============================================================================
print("\n" + "="*60)
print("ANALYZING DECODED TRAJECTORIES")
print("="*60)

# Transform neural activity through the decoder
print("Transforming neural activity to motor output...")
transformed_decoder = transform_reaching(
    reaching_network=reaching_network,
    manifold_out=manifold_out,
    W_bci=W_bci,
    n_output_units=N_OUTPUT_UNITS,
    reduced_dim=manifold_dimension
)

print(f"Transformed decoder shape: {transformed_decoder.shape}")

# ============================================================================
# 10. SAVE RESULTS
# ============================================================================
print("\n" + "="*60)
print("SAVING EXPERIMENT RESULTS")
print("="*60)

# Compile all experiment data
experiment_data = {
    'params': {
        'dt': dt,
        'T': T,
        'time': time,
        'tsteps': tsteps,
        'pulse_length': pulse_length,
        'manifold_trials': n_manifold_trials,
        'target_max': target_max,
        'stimulus_type': stimulus_type,
        'N': N,
        'tau': tau,
        'g': g,
        'p': p
    },
    'stimulus': stimulus,
    'target': target,
    'training': {
        'n_trials': n_initial_training,
        'learning_rate': initial_learning_rate,
        'decoder': reaching_decoder,
        'feedback': reaching_feedback,
        'loss_history': reaching_loss
    },
    'manifold': {
        'activity': manifold_out["activity"],
        'activity_reshaped': manifold_out["activity_reshaped"],
        'xi': manifold_out["xi"],
        'xi2': manifold_out["xi2"],
        'covariance': manifold_out["cov"],
        'eigenvalues': manifold_out["ev"],
        'eigenvectors': manifold_out["evec"],
        'participation_ratio': manifold_out["pr"],
        'trial_order': manifold_out["order"]
    },
    'decoding': {
        'reduced_dim': manifold_dimension,
        'weights': W_bci,
        'loss': decoder_loss
    }
}

# Save experiment data
np.save(f'{savedir}reaching_experiment_results', experiment_data)
print(f"Saved experiment data to: {savedir}reaching_experiment_results.npy")

# Save manifold analysis separately
save_reaching_manifold(experiment_data, transformed_decoder, savedir)
print(f"Saved manifold data to: {savedir}reaching_relearning_results.npy")

# ============================================================================
# 11. VISUALIZE RESULTS
# ============================================================================
print("\n" + "="*60)
print("VISUALIZING REACHING TRAJECTORIES")
print("="*60)

# Simulate and plot reaching trajectories
print("Simulating reaching movements from decoded neural activity...")
trajectories = simulate_reaching(savedir, dt)

print("\n" + "="*60)
print("EXPERIMENT COMPLETE")
print("="*60)
print(f"Results saved to: {savedir}")
print("\nSummary:")
print(f"- Trained RNN with {N} neurons to reach to {N_INPUT_TARGETS} targets")
print(f"- Final training loss: {reaching_loss[-1]:.5f}")
print(f"- Manifold dimensionality: {manifold_out['pr']} effective dimensions")
print(f"- Decoder MSE: {decoder_loss:.6f}")
