# %%
import numpy as np
import sklearn.linear_model as lm
from .task import cost

# %%
class Decoder:
    """
    Decoder for a recurrent neural network.
    """
    def __init__(self, N: int = 800, N_out: int = 2, reduced_dim: int = 10, W: np.ndarray = None):
        """
        Instantiate a decoder for the RNN. Responsible for decoding the output
        of th   e RNN to target coordinates. Randomly initializes the output weights,
        then scales them to have a norm of 1.
        Parameters:
          N: number of neurons in the RNN
          N_out: number of output dimensions (e.g., 2 for x and y coordinates)
          reduced_dim: number of principal components to use for decoding
        """
        self.N = N
        self.N_out = N_out
        self.reduced_dim = reduced_dim
        self.W = W
        self.is_trained = False

    def set_initial_weights(self, target_max_radius: float = 0.2):
        """
        Set initial weights for the decoder.
        Randomly initializes the output weights and scales them based on the
        target maximum radius.
        Parameters:
          target_max_radius: maximum radius for scaling the weights
        """
        self.W = np.random.randn(self.N_out, self.N)
        self.W *= (0.04 * (target_max_radius / 0.2)) / np.linalg.norm(self.W)
  

    def get_feedback_weights(self) -> np.ndarray:
        """
        Get pseudo-inverse feedback weights.
        Returns:
          W: feedback weights matrix of shape (N_out, N)
        """
        if self.W is None:
            raise ValueError("Decoder weights are not initialized.")
        return np.linalg.pinv(self.W)


    def train(self, neural_activity: np.ndarray, targets: np.ndarray, trial_order: np.ndarray):
        """
        Train the decoder using linear regression.

        Parameters:
          neural_activity: neural activity  of shape (trials, tsteps, N)
          targets: stimulus matrix of shape (n_targets, tsteps, n_targets)
          trial_order: order of trials, mapping each trial to its corresponding stimulus
        Returns:
          W: trained output weights matrix of shape (N_out, N)
          mse: mean squared error of the predictions
        """
        n_trials, n_timesteps, _n_dims = neural_activity.shape

        # flatten data for regression
        X = np.zeros((n_trials * n_timesteps, _n_dims))
        y = np.zeros((n_trials * n_timesteps, self.N_out))
        
        for i in range(n_trials):
            start_idx = i * n_timesteps
            end_idx = (i + 1) * n_timesteps
            X[start_idx:end_idx, :] = neural_activity[i]
            y [start_idx:end_idx, :] = targets[trial_order[i]]

        # Train linear regression
        reg = lm.LinearRegression()
        reg.fit(X, y)
        
        self.W = reg.coef_
        self.is_trained = True

        # calculate training MSE
        y_hat = reg.predict(X)
        mse = np.mean((y - y_hat) ** 2)

        print(f"Decoder trained with MSE: {mse:.4f}")
        
        return mse

    def predict(self, neural_activity: np.ndarray) -> np.ndarray:
        """Predict target coordinates from neural activity."""
        if not self.is_trained:
            raise ValueError("Decoder is not trained yet.")
        
        return neural_activity @ self.W.T

    def decode(self, manifold_dict: dict, readout_matrix: np.ndarray = None) -> np.ndarray:
        """
        Decode the neural activity manifold to target coordinates.
        Parameters:
          n_dims: number of principal components to use for decoding
          manifold_dict: dictionary containing the neural activity manifold and eigenvectors
        Returns:
          result: decoded target coordinates of shape (trials, tsteps, N_out)
        """
        if readout_matrix is None:    
          P = manifold_dict["evectors"].real.T
          D = np.zeros((self.N_out, self.N))
          D[:, :self.reduced_dim] = self.W
          T = D @ P
        else:
          T = readout_matrix
        # what is the shape of result?
        result = manifold_dict["activity"] @ T.T
        return result


class ManifoldPerturbation:
    """
    Handles manifold perturbations for the decoder.
    Supports both within-manifold and outside-manifold perturbations.
    """

    def __init__(self, N: int, N_out: int, reduced_dim: int, evectors: np.ndarray, decoder_weights: np.ndarray):
        """
        Initialize the perturbation handler.
        Parameters:
          N: number of neurons in the RNN
          N_out: number of output dimensions (e.g., 2 for x and y coordinates)
          reduced_dim: number of principal components to use for perturbations
          evectors: eigenvectors of the covariance matrix of the neural activity
          decoder_weights: weights of the decoder
        """
        self.N = N
        self.N_out = N_out
        self.reduced_dim = reduced_dim
        self.P = evectors.real.T # P
        self.D = np.zeros((self.N_out, N)) # D
        self.D[:, :reduced_dim] = decoder_weights[:, :reduced_dim]
        self.T = self.D @ self.P # T

    def within_manifold_perturbation(self, seed: int) -> np.ndarray:
        """
        Apply a within-manifold perturbation by shuffling within the reduced dimensions.
        
        Parameters:
          seed: random seed for reproducibility

        Returns:
          T_within: perturbed transformation matrix of shape (reduced_dim, N)
          perm_matrix: permutation matrix used for shuffling
        """
        # create permutation matrix for reduced dimensions
        perm_matrix = np.eye(self.reduced_dim)
        np.random.shuffle(perm_matrix)

        # apply permutation to the decoder matrix
        D_permute = self.D.copy()
        D_permute[:self.reduced_dim, :self.reduced_dim] = \
            self.D[:self.reduced_dim, :self.reduced_dim] @ perm_matrix
        
        # computer perturbed transform
        T_within = D_permute @ self.P

        return T_within, perm_matrix

    def outside_manifold_perturbation(self, seed: int) -> np.ndarray:
        """
        Apply an outside-manifold perturbation by shuffling the full network dimensions.

        Parameters:
          seed: random seed for reproducibility

        Returns:
          T_outside: perturbed transformation matrix of shape (N, N)
          perm_matrix: permutation matrix used for shuffling
        """

        # Create permutation matrix for full network
        perm_matrix = np.eye(self.N)
        np.random.shuffle(perm_matrix)

        # Apply permutation to the manifold basis
        P_permute = self.P @ perm_matrix

        # Compute perturbed transform
        T_outside = self.D @ P_permute

        return T_outside, perm_matrix

    def find_balanced_perturbations(self, activity, targets, trial_order, runs = 200):
        """
        Find perturbations seeds that give similar performance degradation.

        Parameters:
            activity: neural activity matrix of shape (trials, tsteps, N)
            targets: target coordinates of shape (n_targets, tsteps, n_targets)
            trial_order: order of trials, mapping each trial to its corresponding stimulus
            runs: number of perturbation runs to evaluate
        Returns:
            dict: containing seeds for within and outside perturbations, and all costs
        """
        costs = np.zeros((runs, 2))

        for i in range(runs):
            T_within, _ = self.within_manifold_perturbation(seed=i)
            result_within = activity @ T_within.T
            costs[i, 0] = cost(result_within, targets, trial_order)

            T_outside, _ = self.outside_manifold_perturbation(seed=i)
            result_outside = activity @ T_outside.T
            costs[i, 1] = cost(result_outside, targets, trial_order)
        
        # Find perturbations closest to the mean cost
        dif = np.abs(costs - np.mean(costs))
        idx = np.argsort(dif, axis=0)

        T_within, permute_within = self.within_manifold_perturbation(seed=idx[0, 0])
        T_outside, permute_outside = self.outside_manifold_perturbation(seed=idx[0, 1])
        return {
            "within_manifold_perturbation": T_within,
            "outside_manifold_perturbation": T_outside,
            "permute_within": permute_within,
            "permute_outside": permute_outside,
            # "costs": costs,
            "seed_within": idx[0, 0],
            "seed_outside": idx[0, 1]
        }
        
# %%
