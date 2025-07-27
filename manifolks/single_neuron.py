import numpy as np
import matplotlib.pyplot as plt

def plot_neuron_activity_range(manifold_data, neuron_idx, time_vec=None, save_path=None):
    """
    Visualize the range of activity for a single neuron across all trials
    
    Args:
        manifold_data: ndarray (n_trials, n_timesteps, n_neurons)
        neuron_idx: int, which neuron to analyze
        time_vec: optional time vector for x-axis
        save_path: optional path to save figure
    """
    n_trials, n_timesteps, n_neurons = manifold_data.shape
    
    if neuron_idx >= n_neurons:
        raise ValueError(f"Neuron index {neuron_idx} exceeds number of neurons {n_neurons}")
    
    # Extract data for this neuron
    neuron_activity = manifold_data[:, :, neuron_idx]  # (n_trials, n_timesteps)
    
    # Calculate statistics across trials
    mean_activity = np.mean(neuron_activity, axis=0)
    std_activity = np.std(neuron_activity, axis=0)
    min_activity = np.min(neuron_activity, axis=0)
    max_activity = np.max(neuron_activity, axis=0)
    
    # Set up time axis
    if time_vec is None:
        time_vec = np.arange(n_timesteps)
    
    plt.figure(figsize=(12, 6))
    
    # Plot individual trials (light gray)
    for trial in range(n_trials):
        plt.plot(time_vec, neuron_activity[trial, :], 'lightgray', alpha=0.3, linewidth=0.5)
    
    # Plot statistics
    plt.fill_between(time_vec, min_activity, max_activity, alpha=0.2, color='blue', label='Min-Max Range')
    plt.fill_between(time_vec, mean_activity - std_activity, mean_activity + std_activity, 
                     alpha=0.4, color='orange', label='Mean ± STD')
    plt.plot(time_vec, mean_activity, 'red', linewidth=2, label='Mean Activity')
    
    plt.xlabel('Time')
    plt.ylabel('Neural Activity')
    plt.title(f'Activity Range for Neuron {neuron_idx}', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add summary stats as text
    overall_mean = np.mean(neuron_activity)
    overall_std = np.std(neuron_activity)
    plt.text(0.02, 0.98, f'Overall Mean: {overall_mean:.3f}\nOverall STD: {overall_std:.3f}', 
             transform=plt.gca().transAxes, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f"{save_path}/neuron_{neuron_idx}_activity_range.png", dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_neuron_fluctuations(manifold_data, neuron_idx, time_vec=None, save_path=None):
    """
    Visualize fluctuations (trial-to-trial variability) for a single neuron
    
    Args:
        manifold_data: ndarray (n_trials, n_timesteps, n_neurons)
        neuron_idx: int, which neuron to analyze
        time_vec: optional time vector for x-axis
        save_path: optional path to save figure
    """
    n_trials, n_timesteps, n_neurons = manifold_data.shape
    
    if neuron_idx >= n_neurons:
        raise ValueError(f"Neuron index {neuron_idx} exceeds number of neurons {n_neurons}")
    
    # Extract data for this neuron
    neuron_activity = manifold_data[:, :, neuron_idx]  # (n_trials, n_timesteps)
    
    # Calculate mean and fluctuations
    mean_activity = np.mean(neuron_activity, axis=0)
    fluctuations = neuron_activity - mean_activity[np.newaxis, :]  # deviations from mean
    
    # Set up time axis
    if time_vec is None:
        time_vec = np.arange(n_timesteps)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Top plot: Raw activity with mean
    for trial in range(min(10, n_trials)):  # show max 10 trials for clarity
        ax1.plot(time_vec, neuron_activity[trial, :], alpha=0.6, linewidth=1, 
                label=f'Trial {trial+1}' if trial < 5 else '')
    ax1.plot(time_vec, mean_activity, 'black', linewidth=3, label='Mean')
    ax1.set_ylabel('Neural Activity')
    ax1.set_title(f'Raw Activity for Neuron {neuron_idx}', fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Bottom plot: Fluctuations from mean
    for trial in range(min(10, n_trials)):
        ax2.plot(time_vec, fluctuations[trial, :], alpha=0.6, linewidth=1)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=2)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Fluctuation from Mean')
    ax2.set_title('Trial-to-Trial Fluctuations', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add RMS fluctuation as text
    rms_fluctuation = np.sqrt(np.mean(fluctuations**2))
    ax2.text(0.02, 0.98, f'RMS Fluctuation: {rms_fluctuation:.3f}', 
             transform=ax2.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f"{save_path}/neuron_{neuron_idx}_fluctuations.png", dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_neuron_variance_analysis(manifold_data, neuron_idx, time_vec=None, save_path=None):
    """
    Analyze variance in activations across trials for a single neuron
    
    Args:
        manifold_data: ndarray (n_trials, n_timesteps, n_neurons)
        neuron_idx: int, which neuron to analyze
        time_vec: optional time vector for x-axis
        save_path: optional path to save figure
    """
    n_trials, n_timesteps, n_neurons = manifold_data.shape
    
    if neuron_idx >= n_neurons:
        raise ValueError(f"Neuron index {neuron_idx} exceeds number of neurons {n_neurons}")
    
    # Extract data for this neuron
    neuron_activity = manifold_data[:, :, neuron_idx]  # (n_trials, n_timesteps)
    
    # Calculate variance across trials at each time point
    variance_over_time = np.var(neuron_activity, axis=0)
    cv_over_time = np.std(neuron_activity, axis=0) / (np.abs(np.mean(neuron_activity, axis=0)) + 1e-8)  # coefficient of variation
    
    # Calculate variance across time for each trial
    variance_per_trial = np.var(neuron_activity, axis=1)
    
    # Set up time axis
    if time_vec is None:
        time_vec = np.arange(n_timesteps)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Variance Analysis for Neuron {neuron_idx}', fontsize=16, fontweight='bold')
    
    # Top left: Variance over time
    ax1.plot(time_vec, variance_over_time, 'purple', linewidth=2)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Variance Across Trials')
    ax1.set_title('Variance Over Time', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Top right: Coefficient of variation over time
    ax2.plot(time_vec, cv_over_time, 'green', linewidth=2)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Coefficient of Variation')
    ax2.set_title('CV Over Time', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Bottom left: Histogram of variance per trial
    ax3.hist(variance_per_trial, bins=15, alpha=0.7, color='orange', edgecolor='black')
    ax3.set_xlabel('Variance Within Trial')
    ax3.set_ylabel('Count')
    ax3.set_title('Distribution of Within-Trial Variance', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Bottom right: Scatter plot of mean vs variance for each trial
    mean_per_trial = np.mean(neuron_activity, axis=1)
    ax4.scatter(mean_per_trial, variance_per_trial, alpha=0.6, color='red', s=50)
    ax4.set_xlabel('Mean Activity (per trial)')
    ax4.set_ylabel('Variance (per trial)')
    ax4.set_title('Mean vs Variance Relationship', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Add correlation coefficient
    correlation = np.corrcoef(mean_per_trial, variance_per_trial)[0, 1]
    ax4.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
             transform=ax4.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    # Add summary statistics
    summary_text = f"""Summary Statistics:
    Mean Variance: {np.mean(variance_over_time):.3f}
    Max Variance: {np.max(variance_over_time):.3f}
    Mean CV: {np.mean(cv_over_time):.3f}
    Trial-to-trial Variability: {np.std(mean_per_trial):.3f}"""
    
    fig.text(0.02, 0.02, summary_text, fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f"{save_path}/neuron_{neuron_idx}_variance_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def analyze_multiple_neurons(manifold_data, neuron_indices, time_vec=None, save_path=None):
    """
    Quick comparison of activity patterns across multiple neurons
    
    Args:
        manifold_data: ndarray (n_trials, n_timesteps, n_neurons)
        neuron_indices: list of neuron indices to compare
        time_vec: optional time vector for x-axis
        save_path: optional path to save figure
    """
    n_trials, n_timesteps, n_neurons = manifold_data.shape
    
    if time_vec is None:
        time_vec = np.arange(n_timesteps)
    
    n_neurons_to_plot = len(neuron_indices)
    fig, axes = plt.subplots(n_neurons_to_plot, 1, figsize=(12, 3*n_neurons_to_plot))
    if n_neurons_to_plot == 1:
        axes = [axes]
    
    fig.suptitle('Multi-Neuron Activity Comparison', fontsize=16, fontweight='bold')
    
    for i, neuron_idx in enumerate(neuron_indices):
        if neuron_idx >= n_neurons:
            continue
            
        neuron_activity = manifold_data[:, :, neuron_idx]
        mean_activity = np.mean(neuron_activity, axis=0)
        std_activity = np.std(neuron_activity, axis=0)
        
        # Plot mean ± std
        axes[i].fill_between(time_vec, mean_activity - std_activity, 
                           mean_activity + std_activity, alpha=0.3)
        axes[i].plot(time_vec, mean_activity, linewidth=2, label=f'Neuron {neuron_idx}')
        axes[i].set_ylabel('Activity')
        axes[i].set_title(f'Neuron {neuron_idx}', fontweight='bold')
        axes[i].grid(True, alpha=0.3)
        
        # Add variance info
        total_var = np.var(neuron_activity)
        axes[i].text(0.98, 0.95, f'Total Var: {total_var:.3f}', 
                    transform=axes[i].transAxes, ha='right', va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    axes[-1].set_xlabel('Time')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f"{save_path}/multi_neuron_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# Example usage:
"""
# Assuming you have manifold_data from your simulation
manifold_data = your_manifold_dict["activity"]  # shape: (n_trials, n_timesteps, n_neurons)
time_vector = np.arange(manifold_data.shape[1]) * 0.01  # assuming dt=0.01

# Analyze a single neuron
plot_neuron_activity_range(manifold_data, neuron_idx=5, time_vec=time_vector, save_path="./plots")
plot_neuron_fluctuations(manifold_data, neuron_idx=5, time_vec=time_vector, save_path="./plots")
plot_neuron_variance_analysis(manifold_data, neuron_idx=5, time_vec=time_vector, save_path="./plots")

# Compare multiple neurons
analyze_multiple_neurons(manifold_data, [0, 5, 10, 15], time_vec=time_vector, save_path="./plots")
"""