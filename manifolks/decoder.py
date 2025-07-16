# %%
import numpy as np
import sklearn.linear_model as lm
from .task import cost

# %%
class Decoder:
    """
    Decoder for a recurrent neural network.
    """
    def __init__(self, N: int = 800, N_out: int = 2, reduced_dim: int = 10):
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
        self.W = None
        self.is_trained = False
  

    def get_feedback_weights(self) -> np.ndarray:
        """
        Get pseudo-inverse feedback weights.
        Returns:
          W: feedback weights matrix of shape (N_out, N)
        """
        if not self.is_trained:
            raise ValueError("Decoder is not trained yet.")
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
        n_trials, n_timesteps, n_neurons = neural_activity.shape

        # flatten data for regression
        X = np.zeros((n_trials * n_timesteps, self.N))
        y = np.zeros((n_trials * n_timesteps, self.N_out))
        
        for i in range(n_trials):
            start_idx = i * n_timesteps
            end_idx = (i + 1) * n_timesteps
            X[start_idx:end_idx, :] = neural_activity[i, :, :]
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

    def decode(self, manifold_dict: dict) -> np.ndarray:
        """
        Decode the neural activity manifold to target coordinates.
        Parameters:
          n_dims: number of principal components to use for decoding
          manifold_dict: dictionary containing the neural activity manifold and eigenvectors
        Returns:
          result: decoded target coordinates of shape (trials, tsteps, N_out)
        """
        D = self.W[:, :self.reduced_dim]
        P = manifold_dict["evectors"].real.T
        T = D @ P
        # what is the shape of result?
        result = manifold_dict["activity"] @ T.T
        return result


class ManifoldPerturbation:
    """
    Handles manifold perturbations for the decoder.
    Supports both within-manifold and outside-manifold perturbations.
    """

    def __init__(self, N: int, reduced_dim: int):
        """
        Initialize the perturbation handler.
        Parameters:
          N: number of neurons in the RNN
          reduced_dim: number of principal components to use for perturbations
        """
        self.N = N
        self.reduced_dim = reduced_dim
        self.manifold_basis = None
        self.decoder_matrix = None
        self.original_transform = None


    def set_manifold_basis(self, eigenvectors):
        """Set the manifold basis from PCA eigenvectors of neural activity."""
        self.manifold_basis = eigenvectors.real.T

    def set_decoder_matrix(self, decoder_weights: np.ndarray):
        """Set the decoder weights matrix (from a trained decoder)."""
        self.decoder_matrix = np.zeros((decoder_weights.shape[0], self.N))
        self.decoder_matrix[:, :self.reduced_dim] = decoder_weights

    def compute_original_transform(self):
        """Compute original transformation matrix."""
        if self.manifold_basis is None or self.decoder_matrix is None:
            raise ValueError("Set manifold basis and decoder matrix first.")
        
        self.original_transform = self.decoder_matrix @ self.manifold_basis
        return self.original_transform

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
        D_permute = self.decoder_matrix.copy()
        D_permute[:self.reduced_dim, :self.reduced_dim] = \
            self.decoder_matrix[:self.reduced_dim, :self.reduced_dim] @ perm_matrix
        
        # computer perturbed transform
        T_within = D_permute @ self.manifold_basis

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
        perm_matrix = np.eye(self.network_size)
        np.random.shuffle(perm_matrix)

        # Apply permutation to the manifold basis
        P_permute = self.manifold_basis @ perm_matrix

        # Compute perturbed transform
        T_outside = self.decoder_matrix @ P_permute

        return T_outside, perm_matrix

    def find_balanced_perturbations(self, activity, targets, trial_order, runs):
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

        return {
            "within_seed": idx[0, 0],
            "outside_seed": idx[0, 1],
            "all_costs": costs,
        }
        
# %%
