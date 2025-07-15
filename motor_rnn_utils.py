"""
Motor RNN Utilities
===================
Support functions and classes for motor RNN experiments.

This module contains:
1. RNN class - Recurrent Neural Network with FORCE learning
2. Task functions - Create stimuli and targets for motor tasks
3. Decoder functions - Train and apply linear decoders
4. Analysis functions - Manifold analysis and visualization
5. Plotting functions - Visualize stimuli, targets, and results

The RNN uses FORCE (First-Order Reduced and Controlled Error) learning
to train recurrent weights for motor control tasks.
"""

import os
import numpy as np
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional, List

# ============================================================================
# RECURRENT NEURAL NETWORK CLASS
# ============================================================================


class RNN:
    """
    Recurrent Neural Network with FORCE learning capability.

    This RNN implementation:
    - Uses random sparse connectivity (not following Dale's law)
    - Implements continuous-time dynamics with Euler integration
    - Supports FORCE learning for supervised training
    - Can operate in chaotic regime (g > 1) for rich dynamics

    Parameters
    ----------
    N : int
        Number of neurons in the recurrent layer
    g : float
        Recurrent coupling strength (g > 1 for chaotic dynamics)
    p : float
        Connection probability (sparsity = 1 - p)
    tau : float
        Neuron time constant in seconds
    dt : float
        Integration time step in seconds
    N_in : int
        Number of input channels
    """

    def __init__(self, N=800, g=1.5, p=0.1, tau=0.1, dt=0.01, N_in=6):
        # Store parameters
        self.N = N  # Number of neurons
        self.g = g  # Coupling strength
        self.p = p  # Connection probability
        self.K = int(p * N)  # Average connections per neuron
        self.tau = tau  # Time constant
        self.dt = dt  # Time step

        # Create sparse recurrent weight matrix
        # Scale by 1/sqrt(K) to maintain stable dynamics
        print(f"Initializing sparse weight matrix ({p*100:.0f}% connectivity)...")
        mask = np.random.rand(N, N) < p
        np.fill_diagonal(mask, False)  # No self-connections
        self.mask = mask
        self.W = g / np.sqrt(self.K) * np.random.randn(N, N) * mask

        # Create input weights (uniform random in [-1, 1])
        self._N_in = N_in
        self.W_in = (np.random.rand(N, N_in) - 0.5) * 2.0

        # Initialize state variables
        self.r = None  # Firing rates
        self.z = None  # Activation (tanh of rates)

    def update_activation(self):
        """Apply activation function (tanh) to firing rates."""
        self.z = np.tanh(self.r)

    def update_neurons(self, ext):
        """
        Update neural dynamics for one timestep.

        Implements: tau * dr/dt = -r + W @ tanh(r) + W_in @ ext

        Parameters
        ----------
        ext : np.ndarray
            External input vector of shape (N_in,)
        """
        # Euler integration of continuous dynamics
        self.r = self.r + self.dt / self.tau * (
            -self.r  # Decay term
            + np.dot(self.W, self.z)  # Recurrent input
            + np.dot(self.W_in, ext)  # External input
        )
        self.update_activation()

    def simulate(self, T, ext=None, r0=None):
        """
        Simulate network dynamics for duration T.

        Parameters
        ----------
        T : float
            Simulation duration in seconds
        ext : np.ndarray, optional
            External input of shape (timesteps, N_in)
        r0 : np.ndarray, optional
            Initial rates of shape (N,)

        Returns
        -------
        time : np.ndarray
            Time vector
        rates : np.ndarray
            Firing rates over time (timesteps, N)
        activations : np.ndarray
            Tanh activations over time (timesteps, N)
        """
        # Setup time
        time = np.arange(0, T, self.dt)
        tsteps = int(T / self.dt)

        # Create zero input if none provided
        if ext is None:
            ext = np.zeros((tsteps, self.N_in))

        # Validate input shape
        if ext.shape[0] != tsteps or ext.shape[1] != self.N_in:
            raise ValueError(f"Input shape should be ({tsteps}, {self.N_in})")

        # Initialize network state
        if r0 is None:
            self.r = (np.random.rand(self.N) - 0.5) * 2.0  # Random in [-1, 1]
        else:
            self.r = r0
        self.update_activation()

        # Run simulation
        record_r = np.zeros((tsteps, self.N))
        record_r[0, :] = self.r

        for i in range(1, tsteps):
            self.update_neurons(ext=ext[i])
            record_r[i, :] = self.r

        return time, record_r, np.tanh(record_r)

    def relearn(
        self, trials, ext, ntstart, decoder, feedback, target, delta=1.0, wplastic=None
    ):
        """
        Train recurrent weights using FORCE learning.

        FORCE learning updates recurrent weights to minimize output error
        using recursive least squares (RLS) optimization.

        Parameters
        ----------
        trials : int
            Number of training trials
        ext : np.ndarray
            Stimuli of shape (n_targets, timesteps, n_inputs)
        ntstart : int
            Time step when learning begins (after cue period)
        decoder : np.ndarray
            Output weights mapping neurons to outputs (n_outputs, N)
        feedback : np.ndarray
            Feedback weights mapping error to neurons (N, n_outputs)
        target : np.ndarray
            Target outputs of shape (n_targets, timesteps, n_outputs)
        delta : float
            Initial learning rate parameter
        wplastic : list, optional
            Which connections are plastic (default: all non-zero)

        Returns
        -------
        loss : np.ndarray
            Loss per trial
        """
        # Get trial duration
        tsteps = ext.shape[1]

        # Identify plastic connections (non-zero weights)
        if wplastic is None:
            self.W_plastic = [np.where(self.W[i, :] != 0)[0] for i in range(self.N)]
        else:
            self.W_plastic = wplastic

        # Initialize RLS estimation matrices P (one per neuron)
        # P tracks inverse correlation matrix for each neuron's inputs
        self.P = [
            1.0 / delta * np.eye(len(self.W_plastic[i]))
            for i in range(len(self.W_plastic))
        ]

        # Random trial order
        order = np.random.choice(range(ext.shape[0]), trials, replace=True)

        # Track loss
        record_loss = np.zeros(trials)

        # Training loop
        for trial in range(trials):
            # Initialize for this trial
            loss = 0.0
            self.r = (np.random.rand(self.N) - 0.5) * 2.0
            self.update_activation()

            # Run one trial
            for t in range(1, tsteps):
                # Update network state
                self.update_neurons(ext=ext[order[trial], t])

                # FORCE learning after cue period, every other timestep
                if t > ntstart and t % 2 == 0:
                    # Decode current output from neural activity
                    output = decoder @ self.z

                    # Calculate error between decoded and target output
                    error_output = output - target[order[trial], t]

                    # Transform output error to neural error via feedback
                    error_neural = feedback @ error_output

                    # Accumulate loss
                    loss += np.mean(error_neural**2)

                    # Update plastic weights for each neuron
                    for j in range(self.N):
                        # Get activities of neurons connecting to neuron j
                        z_plastic = self.z[self.W_plastic[j]]

                        # RLS update: P @ z
                        pz = np.dot(self.P[j], z_plastic)

                        # Normalization factor
                        norm = 1.0 + np.dot(z_plastic.T, pz)

                        # Update correlation matrix estimate
                        self.P[j] -= np.outer(pz, pz) / norm

                        # Update weights proportional to error
                        self.W[j, self.W_plastic[j]] -= error_neural[j] * pz / norm

            # Record trial loss
            record_loss[trial] = loss
            print(f"Trial {trial+1}/{trials}: Loss = {loss:.5f}")

        return record_loss

    def calculate_manifold(self, trials, ext, ntstart):
        """
        Calculate neural manifold using PCA on network activity.

        Parameters
        ----------
        trials : int
            Number of trials to sample activity
        ext : np.ndarray
            Stimuli of shape (n_targets, timesteps, n_inputs)
        ntstart : int
            Start time for collecting activity (after cue)

        Returns
        -------
        Manifold analysis results (see get_manifold function)
        """
        tsteps = ext.shape[1]
        T = self.dt * tsteps
        points = tsteps - ntstart  # Activity points per trial

        # Collect activity from multiple trials
        activity = np.zeros((points * trials, self.N))
        order = np.random.choice(range(ext.shape[0]), trials, replace=True)

        for trial in range(trials):
            # Simulate one trial
            time, r, z = self.simulate(T, ext[order[trial]])
            # Store activity after cue period
            activity[trial * points : (trial + 1) * points, :] = z[ntstart:, :]

        # PCA analysis
        cov = np.cov(activity.T)  # Covariance matrix
        ev, evec = np.linalg.eig(cov)  # Eigendecomposition

        # Sort by eigenvalue
        idx = np.argsort(ev.real)[::-1]
        ev = ev.real[idx]
        evec = evec.real[:, idx]

        # Participation ratio: effective dimensionality
        pr = np.round(np.sum(ev) ** 2 / np.sum(ev**2)).astype(int)

        # Project activity onto principal components
        xi = activity @ evec

        return activity, cov, ev, evec, pr, xi, order

    def save(self, filename):
        """Save network parameters and weights."""
        np.savez(
            filename,
            N=self.N,
            K=self.K,
            tau=self.tau,
            g=self.g,
            p=self.p,
            dt=self.dt,
            W_in=self.W_in,
            W=self.W,
            N_in=self._N_in,
        )

    def load(self, filename):
        """Load network parameters and weights."""
        net = np.load(filename + ".npz")
        self.N = int(net["N"])
        self.K = int(net["K"])
        self.tau = float(net["tau"])
        self.g = float(net["g"])
        self.p = float(net["p"])
        self.dt = float(net["dt"])
        self.W_in = net["W_in"]
        self.W = net["W"]
        self._N_in = int(net["N_in"])


# ============================================================================
# TASK CREATION FUNCTIONS
# ============================================================================


def create_reaching_task_stimuli(
    tsteps: int,
    pulse_steps: int,
    n_targets: int = 6,
    amplitude: float = 1.0,
    twod: bool = False,
) -> np.ndarray:
    """
    Create directional cue stimuli for reaching task.

    Each stimulus is a brief pulse indicating which target to reach to.

    Parameters
    ----------
    tsteps : int
        Number of timesteps in a trial
    pulse_steps : int
        Duration of cue pulse in timesteps
    n_targets : int
        Number of reaching targets
    amplitude : float
        Amplitude of cue pulse
    twod : bool
        If True, use 2D coordinate encoding; if False, use one-hot

    Returns
    -------
    stimulus : np.ndarray
        Array of shape (n_targets, tsteps, n_inputs)
    """
    if twod:
        # 2D coordinate encoding (not used in main experiment)
        stimulus = np.zeros((n_targets, tsteps, n_targets))
        phis = np.linspace(0, 2 * np.pi, n_targets, endpoint=False)
        for j in range(n_targets):
            stimulus[j, :pulse_steps, 0] = amplitude * np.cos(phis[j])
            stimulus[j, :pulse_steps, 1] = amplitude * np.sin(phis[j])
            stimulus[j, :pulse_steps, 2:] = 0
    else:
        # One-hot encoding: each target gets its own input channel
        stimulus = np.zeros((n_targets, tsteps, n_targets))
        for j in range(n_targets):
            stimulus[j, :pulse_steps, j] = amplitude

    return stimulus


def create_reaching_task_targets(
    tsteps: int,
    pulse_steps: int,
    n_targets: int = 6,
    stype: str = "constant",
    target_max: float = 0.2,
) -> np.ndarray:
    """
    Create target trajectories for reaching task.

    Targets are arranged in a circle. Movement starts after cue offset.

    Parameters
    ----------
    tsteps : int
        Number of timesteps in trial
    pulse_steps : int
        When movement should begin (after cue)
    n_targets : int
        Number of targets arranged in circle
    stype : str
        Movement type ('constant' for step, 'linear' for ramp)
    target_max : float
        Final reach distance (radius of target circle)

    Returns
    -------
    trajectories : np.ndarray
        Target positions of shape (n_targets, tsteps, 2)
    """
    # Angular positions of targets (evenly spaced around circle)
    angles = np.linspace(0, 2 * np.pi, n_targets, endpoint=False)

    # Radial distance over time (0 during cue, then reach out)
    radius = np.zeros(tsteps)
    radius[pulse_steps:] = target_max

    # Convert to x,y coordinates
    trajectories = np.zeros((n_targets, tsteps, 2))
    for j in range(n_targets):
        trajectories[j, :, 0] = radius * np.cos(angles[j])  # x-coordinate
        trajectories[j, :, 1] = radius * np.sin(angles[j])  # y-coordinate

    return trajectories


# ============================================================================
# DECODER FUNCTIONS
# ============================================================================


def create_reaching_task_decoder(
    network: RNN, n_output_units: int = 2, target_max: float = 0.2
) -> np.ndarray:
    """
    Create initial random decoder weights.

    The decoder maps neural activity to motor output (x,y coordinates).
    Initial weights are scaled appropriately for the task.

    Parameters
    ----------
    network : RNN
        The recurrent network
    n_output_units : int
        Number of outputs (2 for x,y)
    target_max : float
        Scale factor based on target distance

    Returns
    -------
    decoder : np.ndarray
        Decoder weights of shape (n_outputs, N_neurons)
    """
    # Random initial weights
    decoder = np.random.randn(n_output_units, network.N)

    # Scale weights appropriately
    # These magic numbers come from empirical tuning
    SCALE = 0.04
    BASELINE = 0.2
    scale_factor = SCALE * (target_max / BASELINE)

    # Normalize by weight magnitude
    decoder *= scale_factor / np.linalg.norm(decoder)

    return decoder


def get_feedback_weights(decoder: np.ndarray) -> np.ndarray:
    """
    Calculate feedback weights using pseudoinverse of decoder.

    Feedback weights transform output error back to neural error
    for FORCE learning.

    Parameters
    ----------
    decoder : np.ndarray
        Forward decoder weights (n_outputs, N_neurons)

    Returns
    -------
    feedback : np.ndarray
        Feedback weights (N_neurons, n_outputs)
    """
    return np.linalg.pinv(decoder)


def train_reaching_decoder(
    neural_activity: np.ndarray,
    targets: np.ndarray,
    trial_order: np.ndarray,
    n_output_units: int = 2,
) -> Tuple[np.ndarray, float]:
    """
    Train linear decoder using ridge regression.

    Maps low-dimensional neural activity to motor output.

    Parameters
    ----------
    neural_activity : np.ndarray
        Neural data of shape (trials, timesteps, dimensions)
    targets : np.ndarray
        Target outputs of shape (n_targets, timesteps, n_outputs)
    trial_order : np.ndarray
        Which target was used in each trial
    n_output_units : int
        Number of outputs

    Returns
    -------
    weights : np.ndarray
        Decoder weights (n_outputs, dimensions)
    mse : float
        Mean squared error of decoder
    """
    # Reshape data for regression
    n_trials, n_steps, n_dims = neural_activity.shape

    # Stack all timepoints
    X = np.zeros((n_trials * n_steps, n_dims))
    Y = np.zeros((n_trials * n_steps, n_output_units))

    for trial in range(n_trials):
        start_idx = trial * n_steps
        end_idx = (trial + 1) * n_steps

        X[start_idx:end_idx, :] = neural_activity[trial]
        Y[start_idx:end_idx, :] = targets[trial_order[trial]]

    # Fit linear regression
    reg = lm.LinearRegression()
    reg.fit(X, Y)

    # Calculate performance
    Y_pred = reg.predict(X)
    mse = np.mean((Y_pred - Y) ** 2)

    return reg.coef_, mse


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================


def get_manifold(
    network: RNN, trials: int, stimulus: np.ndarray, pulse_length: int
) -> Dict:
    """
    Extract neural manifold using PCA.

    Runs multiple trials and performs dimensionality reduction
    to find low-dimensional structure in neural activity.

    Parameters
    ----------
    network : RNN
        Trained network
    trials : int
        Number of trials for sampling activity
    stimulus : np.ndarray
        Task stimuli
    pulse_length : int
        Cue duration (activity collected after this)

    Returns
    -------
    results : dict
        Contains activity, PCA results, eigenvalues, etc.
    """
    # Calculate manifold
    activity, cov, ev, evec, pr, xi, order = network.calculate_manifold(
        trials=trials, ext=stimulus, ntstart=pulse_length
    )

    # Reshape for easier use
    points_per_trial = activity.shape[0] // trials
    activity_reshaped = activity.reshape(trials, points_per_trial, network.N)
    xi_reshaped = xi.reshape(trials, points_per_trial, network.N)

    return {
        "activity": activity,
        "activity_reshaped": activity_reshaped,
        "xi": xi,
        "xi2": xi_reshaped,
        "cov": cov,
        "ev": ev,
        "evec": evec,
        "pr": pr,
        "order": order,
    }


def transform_reaching(
    reaching_network: RNN,
    manifold_out: Dict,
    W_bci: np.ndarray,
    n_output_units: int,
    reduced_dim: int,
) -> np.ndarray:
    """
    Transform neural activity through decoder in PC space.

    Tests if low-dimensional manifold preserves task information.

    Parameters
    ----------
    reaching_network : RNN
        The network
    manifold_out : dict
        Manifold analysis results
    W_bci : np.ndarray
        Decoder weights for PC space
    n_output_units : int
        Number of outputs
    reduced_dim : int
        Number of PCs used

    Returns
    -------
    transformed : np.ndarray
        Full decoder in original neural space
    """
    # Get PCA projection matrix
    P = manifold_out["evec"].real.T

    # Create full decoder: PC decoder → full space
    D = np.zeros((n_output_units, reaching_network.N))
    D[:, :reduced_dim] = W_bci

    # Transform to original space
    transformed = D @ P

    return transformed


def get_cost(result: np.ndarray, target: np.ndarray, order: np.ndarray) -> float:
    """Calculate mean squared error across trials."""
    cost = 0
    for j in range(result.shape[0]):
        error = result[j, :, :] - target[order[j], :, :]
        cost += np.mean(error**2)
    return cost


# ============================================================================
# SAVING AND LOADING FUNCTIONS
# ============================================================================


def save_RNN(network: RNN, savedir: str):
    """Save RNN configuration and initial weights."""
    network.save(os.path.join(savedir, "network"))
    np.save(os.path.join(savedir, "W_initial"), network.W)
    print(f"Saved network configuration to {savedir}")


def save_reaching_manifold(data: Dict, transformed: np.ndarray, savedir: str):
    """Save manifold analysis results."""
    results = {
        "manifold": {"original": data["manifold"]},
        "perturbations": {"transformed": transformed},
    }
    filepath = os.path.join(savedir, "reaching_relearning_results")
    np.save(filepath, results)
    print(f"Saved manifold results to {filepath}.npy")


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================


def simulate_reaching(savedir: str, dt: float) -> np.ndarray:
    """
    Simulate and visualize reaching trajectories.

    Loads saved results and plots hand trajectories.

    Parameters
    ----------
    savedir : str
        Directory with saved results
    dt : float
        Time step for integration

    Returns
    -------
    trajectories : np.ndarray
        Hand positions over time
    """
    # Load results
    filepath = os.path.join(savedir, "reaching_relearning_results.npy")
    data = np.load(filepath, allow_pickle=True).item()

    # Extract velocity data
    activity = data["manifold"]["original"]["activity2"]
    velocities = activity @ data["perturbations"]["transformed"].T

    # Integrate velocities to get positions
    positions = np.zeros(velocities.shape)
    for j in range(1, activity.shape[1]):
        positions[:, j, :] = positions[:, j - 1, :] + velocities[:, j, :] * dt

    # Plot trajectories
    plt.figure(figsize=(8, 8), dpi=100)

    n_trials = positions.shape[0]
    colors = plt.cm.rainbow(np.linspace(0, 1, n_trials))

    for trial in range(n_trials):
        plt.plot(
            positions[trial, :, 0],
            positions[trial, :, 1],
            color=colors[trial],
            alpha=0.7,
            linewidth=2,
        )

    plt.title("Simulated Reaching Trajectories", fontsize=16)
    plt.xlabel("X Position", fontsize=14)
    plt.ylabel("Y Position", fontsize=14)
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    return positions


def plot_reaching_task_stimuli(
    stimulus: np.ndarray, n_targets: int, tsteps: int, T: float
):
    """Visualize stimulus structure as heatmaps."""
    fig, axes = plt.subplots(n_targets, 1, figsize=(12, 6))

    for target in range(n_targets):
        im = axes[target].imshow(stimulus[target, :, :].T, aspect="auto", cmap="Blues")

        # Format axes
        axes[target].set_yticks(range(n_targets))
        axes[target].set_yticklabels(range(n_targets), fontsize=8)
        axes[target].set_ylabel(f"Target {target}", fontsize=9)

        # Only label x-axis on bottom plot
        if target == n_targets - 1:
            axes[target].set_xlabel("Time (s)", fontsize=10)
            axes[target].set_xticks([0, tsteps // 2, tsteps])
            axes[target].set_xticklabels([0, T / 2, T])
        else:
            axes[target].set_xticks([])

    fig.suptitle("Stimulus Structure: One-hot encoded directional cues", fontsize=12)
    plt.tight_layout()


def plot_reaching_task_targets(target: np.ndarray, tsteps: int, T: float):
    """Visualize target trajectories."""
    n_targets = target.shape[0]

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # Plot x and y coordinates separately
    for target_idx in range(n_targets):
        color = plt.cm.rainbow(target_idx / n_targets)
        axes[0].plot(
            target[target_idx, :, 0], color=color, label=f"Target {target_idx}"
        )
        axes[1].plot(target[target_idx, :, 1], color=color)

    # Format axes
    axes[0].set_ylabel("X coordinate", fontsize=12)
    axes[1].set_ylabel("Y coordinate", fontsize=12)
    axes[1].set_xlabel("Time (s)", fontsize=12)

    # Time labels
    time_ticks = [0, tsteps // 4, tsteps // 2, 3 * tsteps // 4, tsteps]
    time_labels = [f"{t*T/tsteps:.1f}" for t in time_ticks]
    axes[1].set_xticks(time_ticks)
    axes[1].set_xticklabels(time_labels)

    # Add grid and legend
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-0.25, 0.25])

    axes[0].legend(loc="upper right", ncol=3, fontsize=8)

    fig.suptitle("Target Trajectories: Reach to 6 directions", fontsize=14)
    plt.tight_layout()


# ============================================================================
# FORCE TASK FUNCTIONS (Alternative task implementation)
# ============================================================================


def create_force_task_stimuli(
    tsteps: int,
    pulse_steps: int,
    n_targets: int = 1,
    amplitude: float = 1.0,
    twod: bool = False,
) -> np.ndarray:
    """Create stimuli for force exertion task."""
    stimulus = np.zeros((n_targets, tsteps, n_targets))
    for j in range(n_targets):
        stimulus[j, :pulse_steps, j] = amplitude
    return stimulus


def create_force_task_targets(
    tsteps: int,
    pulse_steps: int,
    frequencies: List[float] = [1, 10],
    target_max: float = 0.2,
) -> np.ndarray:
    """
    Create oscillating force targets at different frequencies.

    Parameters
    ----------
    tsteps : int
        Number of timesteps
    pulse_steps : int
        When force should begin
    frequencies : list
        Oscillation frequencies to use
    target_max : float
        Maximum force amplitude

    Returns
    -------
    targets : np.ndarray
        Force profiles of shape (n_frequencies, tsteps, 1)
    """
    n_targets = len(frequencies)
    targets = np.zeros((n_targets, tsteps, 1))

    # Time vector for oscillations
    t = np.linspace(-2 * np.pi, 2 * np.pi, tsteps, endpoint=False)

    # Create sinusoidal force profiles
    for i, freq in enumerate(frequencies):
        # Zero force during cue, then oscillate
        amplitude = np.zeros(tsteps)
        amplitude[pulse_steps:] = target_max
        targets[i, :, 0] = amplitude * np.sin(freq * t)

    return targets
