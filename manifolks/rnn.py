import numpy as np
from .decoder import Decoder

class RNN:
    """
    Recurrent Neural Network (not following Dale's law)
    """

    def __init__(self, N=800, g=1.5, p=0.1, dt=0.01, N_in=6, tau=0.1):
        """
        Instantiate a RNN
        Parameters:
        * n: number of neurons
        * n_in: number of input channels to the network
        * g: connection strength aka recurrent coupling strength
        * p: sparsity of connections aka connection probability
        * tau: neuron time constant
        * dt: integration time step
        """
        self.N = N
        self.g = g
        self.p = p
        self.K = int(p*N) # average connections per neuron
        self.tau = tau
        self.dt = dt
        self._N_in = N_in
        self.act_fn = np.tanh # activation function

        # create input weights matrix [-1, 1]
        self.W_in = 2 * (np.random.rand(self.N, self._N_in) - 0.5)

        # create recurrent weights matrix
        mask = (np.random.rand(self.N, self.N) < self.p).astype(float)
        np.fill_diagonal(mask, np.zeros(self.N)) # No self-connections
        self.mask = mask
        self.W = np.random.randn(self.N, self.N) * mask
        # scaled by 1/sqrt(K) to maintain stable dynamics
        self.W *= self.g / np.sqrt(self.K)

    
    def forward(self, r0: np.ndarray, ext: np.ndarray):
        """
        Update neural dynamics for one time step.
        Implements: tau * dr/dt = -r + W @ tanh(r) + W_in @ I_ext

        Parameters
        * ext: external input vector of shape (N_in,)
        """
        # Euler integration of dynamics
        dr = self.dt/self.tau * (
            -r0 + # decay term
            np.dot(self.W, self.act_fn(r0)) + # recurrent input
            np.dot(self.W_in, ext) # external input (target cue)
        )
        r1 = r0 + dr
        z1 = self.act_fn(r1)
        return r1, z1


    def simulate(self, T: float, stim: np.ndarray = None, r0: np.ndarray = None):
        """
        Simulate neural dynamics for duration T
        """
        steps = np.arange(0, T, self.dt)
        nsteps = len(steps)
        
        # Stimulus (Target Cue)
        stim = np.zeros((nsteps, self._N_in)) if stim is None else stim
        assert stim.shape == (nsteps, self._N_in), "Stimulus shape does not match the expected number of input size"
        
        # Network's Initial Conditions (Neural Firing Rates)
        r = 2.0 * (np.random.randn(self.N) - 0.5) if r0 is None else r0
        z = self.act_fn(r)

        # Run Simulation
        # record changes in firing rate over time
        r_trajectory = np.zeros((nsteps, self.N))
        z_trajectory = np.zeros((nsteps, self.N))
        r_trajectory[0, :] = r
        z_trajectory[0, :] = z
        for i in range(1, nsteps):
            r, z = self.forward(r0=r, ext=stim[i])
            r_trajectory[i, :], z_trajectory[i, :] = r, z
        
        return steps, r_trajectory, z_trajectory


    def train(self, trials: int, stim: np.ndarray, pulse_length: int, decoder: Decoder, targets, lr=1., W_p=None):
        """
        Train recurrent weights using FORCE learning.

        FORCE learning updates recurrent weights to minimize output error
        using recursive least squares (RLS) optimization.

        This method simulates the network for a given number of trials,
        collects the response activity after the cue period, and updates the weights
        based on the difference between the decoded output and the target outputs.
        The learning rate is controlled by the lr parameter, which
        determines the step size for weight updates. Plasticity weights can be specified
        to control which connections are updated during training. If not specified,
        all non-zero connections are considered plastic.
        
        Parameters
        ----------
        trials : int
            Number of training trials
        stim : np.ndarray
            Stimuli of shape (n_targets, timesteps, n_targets)
        pulse_length : int
            Length of the stimulus pulse (cue period)
        decoder : Decoder
            Decoder object to decode network output to target coordinates
        targets : np.ndarray
            Target outputs of shape (n_targets, timesteps, n_outputs)
        lr : float
            Learning rate for weight updates (default: 1.0)
        W_p : list, optional
            Which connections are plastic (default: all non-zero). Allows freezing
            part of the network like only updating excitatory connections.
            
        Returns
        -------
        loss : np.ndarray
            Loss per trial
        """
        tsteps = stim.shape[1]
        # plasticity matrix
        W_p = [
            np.where(self.W[i,:] != 0)[0] # non-zero weights
            for i in range(self.N)
        ] if W_p is None else W_p

        # Initialize RLS estimation matrices R (one per neuron)
        # inverse correlation matrix for each neuron's inputs
        P = [
            1.0 / lr * np.eye(len(row))
            for row in W_p
        ]
        feedback_weights = decoder.get_feedback_weights()  # pseudo-inverse feedback weights

        # shuffle the order of trials to avoid training on the same target cue back-to-back
        trial_order = np.random.choice(range(stim.shape[0]), trials, replace=True)

        loss_trajectory = np.zeros(trials)

        for t in range(trials):
            # initialization
            loss = 0.
            r = 2.0 * (np.random.rand(self.N) - 0.5) # random initial condition
            stim_idx = trial_order[t] # stimulus index basically means which target cue to present

            # Run trial
            for i in range(1, tsteps):
                r, z = self.forward(r0=r, ext=stim[stim_idx, i])

                # FORCE learning, after cue period, every other timestep
                # Note that this trial-based RLS update for each neuron is different
                # from what Sussillo & Abbott (2009) did, where they used a linear
                # combination of activity across all neurons to match some waveform.
                if i > pulse_length and i % 2 == 0:
                    c = decoder.W @ z                        # decoded 2d output
                    err_2d = c - targets[stim_idx, i]         # diff from target
                    err_n = feedback_weights @ err_2d # neural error
                    loss += np.mean(err_n ** 2)
                    for j in range(self.N):
                        # per-neuron RLS update
                        # Each neuron is doing local learning, adjusting its own dendrites based on feedback
                        # So neuron j is wondering: "How should I mix my inputs to reduce the error?"
                        z_plastic = z[W_p[j]]                  # presynaptic inputs to neuron j
                        pz = np.dot(P[j], z_plastic)           # RLS projection
                        norm = (1. + np.dot(z_plastic.T, pz))  # denominator
                        P[j] -= np.outer(pz, pz) / norm        # update inverse correlation matrix
                        self.W[j, W_p[j]] -= err_n[j] * pz / norm # update connection strengths (weights)
            loss_trajectory[t] = loss
        return loss_trajectory


    def compute_manifold(self, trials: int, stim: np.ndarray, pulse_length: int):
        """
        Compute Manifold of Neural Activity

        Compute the neural activity manifold after the cue period.
        This method simulates the network for a given number of trials,
        collects the activity after the cue period, and computes the covariance
        matrix of the activity. It then performs eigen decomposition to find
        the principal components of the activity manifold.
        
        Parameters
        ----------
        trials : int
            Number of trials to simulate
        stim : np.ndarray
            Stimuli of shape (n_targets, timesteps, n_targets)
        pulse_length : int
            Length of the stimulus pulse (cue period)
        Returns
        -------
        manifold : dict
        """
        tsteps = stim.shape[1]
        T = self.dt*tsteps
        post_cue_steps = tsteps - pulse_length
        activity_matrix = np.zeros((trials, post_cue_steps, self.N))
        # Randomise trial order to avoid presenting the same target cue back-to-back
        trial_order = np.random.choice(range(stim.shape[0]), trials, replace=True)
        for t in range(trials):
            stim_idx = trial_order[t]
            _, _, z_traj = self.simulate(T=T, stim=stim[stim_idx])
            activity_matrix[t, :, :] = z_traj[pulse_length:, :]
        activity_matrix = activity_matrix.reshape(-1, self.N)  # Flatten to (trials * post_cue_steps, N)
        cov = np.cov(activity_matrix.T)  # Compute covariance matrix
        evals, evectors = np.linalg.eig(cov) # Eigen decomposition
        idx = evals.argsort()[::-1] # Index the eigenvalues in descending order
        evals, evectors = evals[idx], evectors[:, idx] # Sort eigenvaleus and eigenvectors
        pr = np.round(np.sum(evals) ** 2 / np.sum(evals ** 2)).astype(int)  # Projected rank
        xi = activity_matrix @ evectors.real # Projected data
        return {
            "acivity_2d": activity_matrix,
            "activity": activity_matrix.reshape(trials, -1, self.N),
            "cov": cov,
            "evals": evals,
            "evectors": evectors,
            "pr": pr,
            "xi_2d": xi,
            "xi": xi.reshape(trials, -1, self.N),
            "trial_order": trial_order,
        }

    
    def save(self, filename):
        """Save network parameters and weights."""
        np.savez(
            filename,
            N=self.N, K=self.K, tau=self.tau, g=self.g, p=self.p,
            dt=self.dt, W_in=self.W_in, W=self.W, N_in=self._N_in
        )
        

    def load(self, filename):
        """Load network parameters and weights."""
        net = np.load(filename + '.npz')
        self.N = int(net['N'])
        self.K = int(net['K'])
        self.tau = float(net['tau'])
        self.g = float(net['g'])
        self.p = float(net['p'])
        self.dt = float(net['dt'])
        self.W_in = net['W_in']
        self.W = net['W']
        self._N_in = int(net['N_in'])