import numpy as np


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
        self.N = N_in
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
        mask = np.random.rand(self.N, self.N) < self.p
        np.fill_diagonal(mask, np.zeros(self.N)) # No self-connections
        self.mask = mask
        # scaled by 1/sqrt(K) to maintain stable dynamics
        self.W = self.g / np.sqrt(self.K) * np.random.randn(self.N, self.N) * mask

        # state variables
        self.r = None # Firing rates
        self.z = None # Activation (tanh of rates)
    
    def forward(self, r0: np.ndarray, x: np.ndarray):
        """
        Update neural dynamics for one time step.
        Implements: tau * dr/dt = -r + W @ tanh(r) + W_in @ X

        Parameters
        * x: external input vector of shape (N_in,)
        """
        # Euler integration of continuous dynamics
        r1 = r0 + self.dt/self.tau * (
            -r0 + # decay term
            np.dot(self.W, self.act_fn(r0)) + # recurrent input
            np.dot(self.W_in, x) # external input (target cue)
        )
        z1 = self.act_fn(z0)
        return r1, z1

    def simulate(self, t: float, stim: np.ndarray = None, r0: np.ndarray = None):
        """
        Simulate neural dynamics for duration t
        """
        steps = np.arange(0, t, self.dt)
        nsteps = len(steps)
        
        # Stimulus (Target Cue)
        stim = np.zeros((nsteps, self._N_in)) if stim is None else stim
        assert stim.shape == (nsteps, self._N_in), "Stimulus shape does not match the expected number of input size"
        
        # Network's Initial Conditions (Neural Firing Rates)
        r0 = 2.0 * (np.random.randn(self.N) - 0.5) if r0 is None else r0
        z0 = self.act_fn(r0)

        # Run Simulation
        # record changes in firing rate over time
        r_trajectory = np.zeros((nsteps, self.N))
        z_trajectory = np.zeros((nsteps, self.N))
        r_trajectory[0, :] = r0
        z_trajectory[0, :] = z0
        for i in range(1, nsteps):
            r, z = self.forward(x=stim[i])
            r_trajectory[i, :], z_trajectory[i, :] = r, z
        
        return steps, r_trajectory, z_trajectory

    def train(self, trials, stim, ntstart, decoder, feedback, target, delta=1., wplastic=None):
        """
        Train recurrent weights using FORCE learning.
        
        FORCE learning updates recurrent weights to minimize output error
        using recursive least squares (RLS) optimization.
        
        Parameters
        ----------
        trials : int
            Number of training trials
        stim : np.ndarray
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
        tsteps = stim.shape[1]
        # plasticity matrix
        W_p = [
            np.where(self.W[i,:] != 0)[0] # non-zero weights
            for i in range(self.N)
        ] if wplastic is None else wplastic

        # Initialize RLS estimation matrices R (one per neuron)
        # R tracks inverse correlation matrix for each neuron's inputs
        R = [
            1.0 / delta * np.eye(len(row))
            for row in W_p
        ]

        trial_order = np.random.choice(range(ext.shape[0]), trials, replace=True)

        loss_trajectory = np.zeros(trials)

        for t in range(trials):
            # initialization
            loss = 0.
            r = 2.0 * (np.random.rand(self.N) - 0.5) # random initial condition
            order = trial_order[t]

            # Run trial
            for i in range(1, tsteps):
                r, z = self.forward(r0=r, x=stim[order, i])

                # FORCE learning, after cue period, every other timestep
                if t > ntstart and t % 2 == 0:
                    c = decoder @ z # decoded 2d output
                    err_c = c - target[order, i] # diff from target
                    err_n = feedback @ errc # neural error
                    loss += np.mean(err_n ** 2)
                    for j in range(self.N):
                        z_plastic = z[W_p[j]]
                        pz = np.dot(self.R[j], z_plastic)
                        norm = (1. + np.dot(z_plastic.T, pz))
                        self.P[j] -= np.outer(pz, pz) / norm
                        







    
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


    