"""
Figure 2: Perfect Feedback Signal Simulation
"""

# %% Imports
import numpy as np
from pathlib import Path
import sklearn.linear_model as lm

# %% Setup
# output directory
OUT_DIR = Path(__file__).parents[1] / "data"
OUT_DIR.mkdir(exist_ok=True, parents=True)

# Simulation Parameters
RANDOM_SEED = 2
DT = 0.01  # intergration time step
T = 2  # total simulation time
TIME = np.arange(0, T, DT)  # time vector
T_STEPS = len(TIME)  # number of time steps
TARGETS = 6  # number of radial directions
TARGET_MAX_RADIUS = 0.2  # max reachout distance
PULSE_LENGTH = int(TARGET_MAX_RADIUS / DT)
N_COMPONENTS = 5  # reduced dimensionality for decoding

# Network Parameters
N = 30  # number of neurons
G = 1.5  # recurrent coupling strength
P = 0.2  # connection probability
TAU = 0.01  # time constant for synaptic dynamics

# Training Parameters
N_TRIALS_TRAIN = 80  # number of trials for initial training
N_TRIALS_RELEARN = 80  # number of trials for relearning
N_TRIALS_BCI_MANIFOLD = 50  # number of trials for BCI manifold computation
LEARNING_RATE = 20.0  # learning rate for all network trainings

np.random.seed(RANDOM_SEED)


# %%
class RNN(object):
    """
    Class implementing a recurrent network (not following Dale's law).

    Parameters:
    -----------
    * N: number of neurons
    * N_in: how many inputs can the network have
    * N_out: how many neurons are recorded by external device
    * g: recurrent coupling strength
    * p: connection probability
    * tau: neuron time constant
    * dt: set dt for simulation
    * delta: defines initial learning rate for FORCE
    * P_plastic: how many neurons are plastic in the recurrent network
    """

    def __init__(self, N=800, g=1.5, p=0.1, tau=0.1, dt=0.01, N_in=6):
        # set parameters
        self.N = N
        self.g = g
        self.p = p
        self.K = int(p * N)
        self.tau = tau
        self.dt = dt

        # create recurrent W
        mask = np.random.rand(self.N, self.N) < self.p
        np.fill_diagonal(mask, np.zeros(self.N))
        self.mask = mask
        self.W = self.g / np.sqrt(self.K) * np.random.randn(self.N, self.N) * mask

        # create Win and Wout
        self._N_in = N_in
        self.W_in = (np.random.rand(self.N, self._N_in) - 0.5) * 2.0

    @property
    def N_in(self):
        return self._N_in

    @N_in.setter
    def N_in(self, value):
        self._N_in = value
        self.W_in = (np.random.rand(self.N, self._N_in) - 0.5) * 2.0

    def save(self, filename):
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
        net = np.load(filename + ".npz")
        self.N = int(net["N"])
        self.dt = float(net["dt"])
        self.K = int(net["K"])
        self.tau = float(net["tau"])
        self.g = float(net["g"])
        self.p = float(net["p"])
        self.W_in = net["W_in"]
        self.W = net["W"]
        self._N_in = int(net["N_in"])

    def update_activation(self):
        self.z = np.tanh(self.r)

    def update_neurons(self, ext):
        self.r = self.r + self.dt / self.tau * (
            -self.r + np.dot(self.W, self.z) + np.dot(self.W_in, ext)
        )

        self.update_activation()

    def simulate(self, T, ext=None, r0=None):

        # define time
        time = np.arange(0, T, self.dt)
        tsteps = int(T / self.dt)

        # create input in case no input is given
        if ext is None:
            ext = np.zeros((tsteps, self.N_in))

        # check if input has the right shape
        if ext.shape[0] != tsteps or ext.shape[1] != self.N_in:
            print("ERROR: stimulus shape should be (time x number of input nodes)")
            return

        # set initial condition
        if r0 is None:
            self.r = (np.random.rand(self.N) - 0.5) * 2.0
        else:
            self.r = r0
        self.update_activation()

        # start simulation
        record_r = np.zeros((tsteps, self.N))
        record_r[0, :] = self.r
        for i in range(1, tsteps):
            self.update_neurons(ext=ext[i])
            # store activity
            record_r[i, :] = self.r

        return time, record_r, np.tanh(record_r)

    def relearn(
        self, trials, ext, ntstart, decoder, feedback, target, delta=1.0, wplastic=None
    ):
        tsteps = ext.shape[1]
        # set up learning
        if wplastic is None:
            self.W_plastic = [np.where(self.W[i, :] != 0)[0] for i in range(self.N)]
        else:
            self.W_plastic = wplastic
        self.P = [
            1.0 / delta * np.eye(len(self.W_plastic[i]))
            for i in range(len(self.W_plastic))
        ]
        order = np.random.choice(range(ext.shape[0]), trials, replace=True)
        record_loss = np.zeros(trials)
        # start learning
        for t in range(trials):
            loss = 0.0
            self.r = (np.random.rand(self.N) - 0.5) * 2.0
            self.update_activation()
            # loop over time
            for i in range(1, tsteps):
                self.update_neurons(ext=ext[order[t], i])
                # learning part
                if i > ntstart and i % 2 == 0:
                    c = decoder @ self.z
                    errc = c - target[order[t], i]
                    err1 = feedback @ errc
                    loss += np.mean(err1**2)
                    # ONLY RECURRENT WEIGHT UPDATE
                    for j in range(self.N):
                        z_plastic = self.z[self.W_plastic[j]]
                        pz = np.dot(self.P[j], z_plastic)
                        norm = 1.0 + np.dot(z_plastic.T, pz)
                        self.P[j] -= np.outer(pz, pz) / norm
                        self.W[j, self.W_plastic[j]] -= err1[j] * pz / norm
            record_loss[t] = loss
            print("Loss in Trial %d is %.5f" % (t + 1, loss))
        return record_loss

    def calculate_manifold(self, trials, ext, ntstart):
        tsteps = ext.shape[1]
        T = self.dt * tsteps
        points = tsteps - ntstart
        activity = np.zeros((points * trials, self.N))
        order = np.random.choice(range(ext.shape[0]), trials, replace=True)
        for t in range(trials):
            time, r, z = self.simulate(T, ext[order[t]])
            activity[t * points : (t + 1) * points, :] = z[ntstart:, :]
        cov = np.cov(activity.T)
        ev, evec = np.linalg.eig(cov)
        pr = np.round(np.sum(ev.real) ** 2 / np.sum(ev.real**2)).astype(int)
        xi = activity @ evec.real
        return activity, cov, ev.real, evec.real, pr, xi, order


# %%
def create_stimulus(tsteps, pulse_steps, n_targets=6, amplitude=1.0, twod=False):
    # create stimulus
    stimulus = np.zeros((n_targets, tsteps, n_targets))
    if twod:
        phis = np.linspace(0, 2 * np.pi, TARGETS, endpoint=False)
        for j in range(stimulus.shape[0]):
            stimulus[j, :PULSE_LENGTH, 0] = amplitude * np.cos(phis[j])
            stimulus[j, :PULSE_LENGTH, 1] = amplitude * np.sin(phis[j])
            stimulus[j, :PULSE_LENGTH, 2:] = 0
    else:
        for j in range(n_targets):
            stimulus[j, :pulse_steps, j] = amplitude
    return stimulus


def create_target(tsteps, pulse_steps, n_targets=6, stype="constant", target_max=0.2):
    # create target trajectories
    phis = np.linspace(0, 2 * np.pi, n_targets, endpoint=False)
    rs = np.zeros(tsteps)
    # TARGET DEFINITION
    if stype == "linear":
        # OPTION 1) linear for position
        rs[pulse_steps:] = np.linspace(0, target_max, tsteps - pulse_steps)
    elif stype == "normal":
        # OPTION 2) Gaussian speed profile
        xx = np.linspace(0, target_max, tsteps - pulse_steps)
        mu = target_max / 2.0
        sigma = target_max / 8.0
        rs[pulse_steps:] = target_max * np.exp(-((xx - mu) ** 2) / (2 * sigma**2))
    elif stype == "constant":
        # OPTION 3) constant speed
        rs[pulse_steps:] = np.ones(tsteps - pulse_steps) * target_max
    traj = np.zeros((n_targets, tsteps, 2))
    for j in range(n_targets):
        traj[j, :, 0] = rs * np.cos(phis[j])
        traj[j, :, 1] = rs * np.sin(phis[j])

    return traj


def decoder_training(inputP, target, order):
    X = np.zeros((inputP.shape[0] * inputP.shape[1], inputP.shape[-1]))
    Y = np.zeros((inputP.shape[0] * inputP.shape[1], 2))
    for j in range(inputP.shape[0]):
        X[j * inputP.shape[1] : (j + 1) * inputP.shape[1], :] = inputP[j]
        Y[j * inputP.shape[1] : (j + 1) * inputP.shape[1], :] = target[order[j]]
    reg = lm.LinearRegression()
    reg.fit(X, Y)
    y = reg.predict(X)
    mse = np.mean((y - Y) ** 2)
    #    print('MSE = %.4f'%mse)
    return reg.coef_, mse


def get_cost(result, target, order):
    cost = 0
    for j in range(result.shape[0]):
        error = result[j, :, :] - target[order[j], :, :]
        cost += np.mean(error**2)
    return cost


def select_random_perturbations(activity2, D, P, network, target, order):
    runs = 200
    cost = np.zeros((runs, 2))
    for j in range(runs):
        # set random seed for reproduction
        np.random.seed(j)
        # within-manifold perturbation
        perm_matrix_within = np.eye(N_COMPONENTS)
        np.random.shuffle(perm_matrix_within)
        D_permute = D.copy()
        D_permute[:N_COMPONENTS, :N_COMPONENTS] = (
            D[:N_COMPONENTS, :N_COMPONENTS] @ perm_matrix_within
        )
        T_within = D_permute @ P
        result_within = activity2 @ T_within.T
        cost[j, 0] = get_cost(result_within, target[:, PULSE_LENGTH:, :], order)
        # set random seed for reproduction
        np.random.seed(j)
        # outside-manifold perturbation
        perm_matrix_outside = np.eye(network.N)
        np.random.shuffle(perm_matrix_outside)
        P_permute = P @ perm_matrix_outside
        T_outside = D @ P_permute
        result_outside = activity2 @ T_outside.T
        cost[j, 1] = get_cost(result_outside, target[:, PULSE_LENGTH:, :], order)
    # select closest to 1 perturbations
    dif = abs(cost - np.mean(cost))
    idx = np.argsort(dif, axis=0)
    return idx  # first column seeds for within, second column seeds for outside

# %%
stimulus = create_stimulus(T_STEPS, PULSE_LENGTH, n_targets=TARGETS, twod=False)  

# create target
target = create_target(T_STEPS, PULSE_LENGTH, n_targets=TARGETS,
                                  stype="constant", target_max=TARGET_MAX_RADIUS)  
    
# create network
network = RNN(N=N,g=G,p=P,tau=TAU,dt=DT,N_in=TARGETS)            
# %%
decoder = np.random.randn(2,network.N)
initial_decoder_fac = 0.04 * (TARGET_MAX_RADIUS/0.2)
decoder *= (initial_decoder_fac / np.linalg.norm(decoder))
feedback = np.linalg.pinv(decoder)
stabilize_loss = network.relearn(N_TRIALS_TRAIN, stimulus, PULSE_LENGTH, 
                      decoder, feedback, target, delta=LEARNING_RATE)
# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(stabilize_loss, label="Stabilization Loss")
ax.set_xlabel("Training Trial")
ax.set_ylabel("Loss")
ax.set_title("Network Stabilization Loss")
ax.legend()
plt.show()
# %%
# visualize initial decoder weights and feedback
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(decoder, aspect='auto', cmap='coolwarm')
plt.title("Initial Decoder Weights")
plt.colorbar()
plt.subplot(1, 2, 2)
plt.imshow(feedback.T, aspect='auto', cmap='coolwarm')
plt.title("Feedback Weights")
plt.colorbar()
plt.tight_layout()
plt.show()

# %%
activity,cov,ev,evec,pr,xi,order = network.calculate_manifold(trials=N_TRIALS_BCI_MANIFOLD, 
                                                              ext=stimulus, ntstart=PULSE_LENGTH)
# rearrange the activity a bit
activity2 = activity.reshape(N_TRIALS_BCI_MANIFOLD,-1,network.N)
xi2 = xi.reshape(N_TRIALS_BCI_MANIFOLD,-1,network.N)

# %%
W_bci4,l4 = decoder_training(xi2[:,:,:N_COMPONENTS],target[:,PULSE_LENGTH:,:],order)
# %%
# visualize trained decoder weights and feedback
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(W_bci4, aspect='auto', cmap='coolwarm')
plt.title("Trained Decoder Weights")
plt.colorbar()
plt.subplot(1, 2, 2)
bci4_feedback = np.linalg.pinv(W_bci4)
plt.imshow(bci4_feedback.T, aspect='auto', cmap='coolwarm')
plt.title("Feedback Weights")
plt.colorbar()
plt.tight_layout()
plt.show()