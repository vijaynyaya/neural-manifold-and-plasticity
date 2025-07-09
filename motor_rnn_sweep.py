"""
Motor RNN Parameter Sweep Pipeline
==================================
Systematically vary experiment parameters and analyze results.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
from typing import Dict, List, Tuple, Any
import pickle
from motor_rnn_utils import *

# ============================================================================
# SWEEP CONFIGURATION
# ============================================================================

# Choose parameter to sweep (uncomment one)
SWEEP_PARAM = 'N'  # Options: 'g', 'N', 'p', 'tau', 'target_max', 'learning_rate', 'manifold_dim'

# Define sweep values for each parameter
PARAM_RANGES = {
    'g': np.linspace(0.5, 2.5, 10),  # Synaptic strength
    'N': np.array([200, 400, 600, 800, 1000, 1200]),  # Network size
    'p': np.linspace(0.05, 0.3, 8),  # Connectivity
    'tau': np.logspace(-2, 0, 8),  # Time constant (log scale)
    'target_max': np.linspace(0.1, 0.5, 8),  # Reach distance
    'learning_rate': np.logspace(0.5, 2.5, 8),  # FORCE learning rate
    'manifold_dim': np.arange(5, 31, 5),  # Decoder dimensionality
}

# Get sweep values
SWEEP_VALUES = PARAM_RANGES[SWEEP_PARAM]
N_RUNS = len(SWEEP_VALUES)
N_SEEDS = 3  # Repetitions per parameter value

# Fixed parameters (override as needed)
BASE_PARAMS = {
    'dt': 0.01,
    'T': 2.0,
    'pulse_duration': 0.2,
    'N': 800,
    'g': 1.5,
    'p': 0.1,
    'tau': 0.1,
    'n_targets': 6,
    'n_output': 2,
    'target_max': 0.2,
    'n_training': 80,
    'learning_rate': 20.0,
    'n_manifold_trials': 50,
    'manifold_dim': 10,
}

# Output configuration
SAVE_RESULTS = True
PLOT_RESULTS = True
SWEEP_NAME = f"{SWEEP_PARAM}_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
SWEEP_DIR = f"proj_rnn/sweeps/{SWEEP_NAME}/"

# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

def run_single_experiment(params: Dict, seed: int) -> Dict:
    """Run one experiment with given parameters."""
    np.random.seed(seed)
    
    # Extract parameters
    dt = params['dt']
    T = params['T']
    pulse_duration = params['pulse_duration']
    tsteps = int(T / dt)
    pulse_length = int(pulse_duration / dt)
    
    # Create stimuli and targets
    stimulus = create_reaching_task_stimuli(tsteps, pulse_length, params['n_targets'])
    target = create_reaching_task_targets(tsteps, pulse_length, params['n_targets'], 
                                        target_max=params['target_max'])
    
    # Build network
    network = RNN(N=params['N'], g=params['g'], p=params['p'], 
                  tau=params['tau'], dt=dt, N_in=params['n_targets'])
    
    # Create decoder and feedback
    decoder = create_reaching_task_decoder(network, params['n_output'])
    feedback = get_feedback_weights(decoder)
    
    # Train network
    loss_history = network.relearn(
        trials=params['n_training'],
        ext=stimulus,
        ntstart=pulse_length,
        decoder=decoder,
        feedback=feedback,
        target=target,
        delta=params['learning_rate']
    )
    
    # Manifold analysis
    manifold = get_manifold(network, params['n_manifold_trials'], stimulus, pulse_length)
    
    # Train manifold decoder
    decoder_weights, decoder_mse = train_reaching_decoder(
        manifold["xi2"][:, :, :params['manifold_dim']],
        target[:, pulse_length:, :],
        manifold["order"],
        params['n_output']
    )
    
    # Compute performance metrics
    final_loss = loss_history[-5:].mean()  # Average last 5 trials
    convergence_rate = -np.polyfit(range(len(loss_history)), np.log(loss_history + 1e-10), 1)[0]
    
    # Network statistics
    weight_stats = {
        'mean': np.mean(network.W[network.W != 0]),
        'std': np.std(network.W[network.W != 0]),
        'sparsity': np.mean(network.W != 0),
    }
    
    # Activity statistics
    activity_stats = {
        'mean_rate': np.mean(manifold['activity']),
        'participation': manifold['pr'],
        'top_pc_variance': manifold['ev'][0] / np.sum(manifold['ev']),
        'dimensionality': np.sum(manifold['ev'])**2 / np.sum(manifold['ev']**2),
    }
    
    return {
        'seed': seed,
        'params': params,
        'final_loss': final_loss,
        'decoder_mse': decoder_mse,
        'convergence_rate': convergence_rate,
        'loss_history': loss_history,
        'weight_stats': weight_stats,
        'activity_stats': activity_stats,
        'manifold_eigenvalues': manifold['ev'][:50],  # Top 50
    }

# ============================================================================
# PARAMETER SWEEP
# ============================================================================

def run_parameter_sweep():
    """Run full parameter sweep."""
    
    # Create output directory
    if SAVE_RESULTS:
        os.makedirs(SWEEP_DIR, exist_ok=True)
        print(f"Saving results to: {SWEEP_DIR}")
    
    # Initialize results storage
    results = {
        'sweep_param': SWEEP_PARAM,
        'sweep_values': SWEEP_VALUES,
        'base_params': BASE_PARAMS,
        'experiments': []
    }
    
    # Progress tracking
    total_runs = N_RUNS * N_SEEDS
    run_count = 0
    
    print(f"\nSweeping {SWEEP_PARAM}: {N_RUNS} values × {N_SEEDS} seeds = {total_runs} runs")
    print("=" * 60)
    
    # Sweep loop
    for i, param_value in enumerate(SWEEP_VALUES):
        print(f"\n{SWEEP_PARAM} = {param_value:.3f} ({i+1}/{N_RUNS})")
        
        # Update parameters
        experiment_params = BASE_PARAMS.copy()
        experiment_params[SWEEP_PARAM] = param_value
        
        # Run multiple seeds
        param_results = []
        for seed in range(N_SEEDS):
            run_count += 1
            print(f"  Seed {seed+1}/{N_SEEDS} (Total: {run_count}/{total_runs})", end='')
            
            try:
                result = run_single_experiment(experiment_params, seed)
                param_results.append(result)
                print(f" - Loss: {result['final_loss']:.4f}")
            except Exception as e:
                print(f" - FAILED: {str(e)}")
                param_results.append({'failed': True, 'error': str(e)})
        
        results['experiments'].append({
            'param_value': param_value,
            'runs': param_results
        })
        
        # Save intermediate results
        if SAVE_RESULTS and (i + 1) % 5 == 0:
            save_results(results, SWEEP_DIR)
    
    # Final save
    if SAVE_RESULTS:
        save_results(results, SWEEP_DIR)
    
    return results

def save_results(results: Dict, save_dir: str):
    """Save sweep results."""
    # Save as pickle for full data
    with open(os.path.join(save_dir, 'sweep_results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    # Save summary as JSON (excluding large arrays)
    summary = create_summary(results)
    with open(os.path.join(save_dir, 'sweep_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

def create_summary(results: Dict) -> Dict:
    """Create summary statistics."""
    summary = {
        'sweep_param': results['sweep_param'],
        'sweep_values': results['sweep_values'].tolist(),
        'statistics': []
    }
    
    for exp_group in results['experiments']:
        param_value = exp_group['param_value']
        runs = [r for r in exp_group['runs'] if 'failed' not in r]
        
        if runs:
            stats = {
                'param_value': param_value,
                'n_successful': len(runs),
                'final_loss': {
                    'mean': np.mean([r['final_loss'] for r in runs]),
                    'std': np.std([r['final_loss'] for r in runs]),
                    'min': np.min([r['final_loss'] for r in runs]),
                },
                'decoder_mse': {
                    'mean': np.mean([r['decoder_mse'] for r in runs]),
                    'std': np.std([r['decoder_mse'] for r in runs]),
                },
                'convergence_rate': {
                    'mean': np.mean([r['convergence_rate'] for r in runs]),
                    'std': np.std([r['convergence_rate'] for r in runs]),
                },
                'dimensionality': {
                    'mean': np.mean([r['activity_stats']['dimensionality'] for r in runs]),
                    'std': np.std([r['activity_stats']['dimensionality'] for r in runs]),
                },
                'participation_ratio': {
                    'mean': np.mean([r['activity_stats']['participation'] for r in runs]),
                    'std': np.std([r['activity_stats']['participation'] for r in runs]),
                },
            }
        else:
            stats = {'param_value': param_value, 'n_successful': 0}
        
        summary['statistics'].append(stats)
    
    return summary

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_sweep_results(results: Dict):
    """Create summary plots."""
    
    # Extract data
    param_values = []
    final_losses = []
    decoder_mses = []
    convergence_rates = []
    dimensionalities = []
    
    for exp_group in results['experiments']:
        runs = [r for r in exp_group['runs'] if 'failed' not in r]
        if runs:
            param_values.append(exp_group['param_value'])
            final_losses.append([r['final_loss'] for r in runs])
            decoder_mses.append([r['decoder_mse'] for r in runs])
            convergence_rates.append([r['convergence_rate'] for r in runs])
            dimensionalities.append([r['activity_stats']['dimensionality'] for r in runs])
    
    # Setup figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Parameter Sweep: {results["sweep_param"]}', fontsize=16)
    
    # Final loss
    ax = axes[0, 0]
    plot_with_error(ax, param_values, final_losses)
    ax.set_ylabel('Final Loss')
    ax.set_title('Training Performance')
    ax.set_yscale('log')
    
    # Decoder MSE
    ax = axes[0, 1]
    plot_with_error(ax, param_values, decoder_mses)
    ax.set_ylabel('Decoder MSE')
    ax.set_title('Decoding Performance')
    ax.set_yscale('log')
    
    # Convergence rate
    ax = axes[1, 0]
    plot_with_error(ax, param_values, convergence_rates)
    ax.set_ylabel('Convergence Rate')
    ax.set_title('Learning Speed')
    
    # Dimensionality
    ax = axes[1, 1]
    plot_with_error(ax, param_values, dimensionalities)
    ax.set_ylabel('Effective Dimensionality')
    ax.set_title('Neural Manifold')
    
    # Format axes
    for ax in axes.flat:
        ax.set_xlabel(results['sweep_param'])
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if SAVE_RESULTS:
        plt.savefig(os.path.join(SWEEP_DIR, 'sweep_summary.png'), dpi=150)
        plt.savefig(os.path.join(SWEEP_DIR, 'sweep_summary.pdf'))
    
    # Additional plots based on parameter
    if results['sweep_param'] == 'g':
        plot_chaos_transition(results)
    elif results['sweep_param'] == 'N':
        plot_scaling_analysis(results)
    
    return fig

def plot_with_error(ax, x_values, y_values_list):
    """Plot with error bars."""
    means = [np.mean(y) for y in y_values_list]
    stds = [np.std(y) for y in y_values_list]
    
    ax.errorbar(x_values, means, yerr=stds, marker='o', capsize=5, capthick=2)
    
    # Plot individual points
    for x, y_values in zip(x_values, y_values_list):
        ax.scatter([x] * len(y_values), y_values, alpha=0.3, s=30)

def plot_chaos_transition(results: Dict):
    """Special plot for g parameter showing chaos transition."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract weight statistics
    param_values = []
    weight_stds = []
    
    for exp_group in results['experiments']:
        runs = [r for r in exp_group['runs'] if 'failed' not in r]
        if runs:
            param_values.append(exp_group['param_value'])
            weight_stds.append([r['weight_stats']['std'] for r in runs])
    
    plot_with_error(ax, param_values, weight_stds)
    ax.axvline(x=1.0, color='red', linestyle='--', label='Chaos transition')
    ax.set_xlabel('Synaptic strength (g)')
    ax.set_ylabel('Weight standard deviation')
    ax.set_title('Network Dynamics vs Synaptic Strength')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if SAVE_RESULTS:
        plt.savefig(os.path.join(SWEEP_DIR, 'chaos_transition.png'), dpi=150)

def plot_scaling_analysis(results: Dict):
    """Special plot for N parameter showing scaling properties."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Extract data
    param_values = []
    final_losses = []
    dimensionalities = []
    
    for exp_group in results['experiments']:
        runs = [r for r in exp_group['runs'] if 'failed' not in r]
        if runs:
            param_values.append(exp_group['param_value'])
            final_losses.append([r['final_loss'] for r in runs])
            dimensionalities.append([r['activity_stats']['dimensionality'] for r in runs])
    
    # Loss scaling
    ax = axes[0]
    means = [np.mean(y) for y in final_losses]
    ax.loglog(param_values, means, 'o-')
    ax.set_xlabel('Network size (N)')
    ax.set_ylabel('Final loss')
    ax.set_title('Performance Scaling')
    ax.grid(True, alpha=0.3)
    
    # Dimensionality scaling
    ax = axes[1]
    means = [np.mean(y) for y in dimensionalities]
    ax.semilogx(param_values, means, 'o-')
    ax.set_xlabel('Network size (N)')
    ax.set_ylabel('Effective dimensionality')
    ax.set_title('Manifold Scaling')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if SAVE_RESULTS:
        plt.savefig(os.path.join(SWEEP_DIR, 'scaling_analysis.png'), dpi=150)

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def analyze_critical_points(results: Dict) -> Dict:
    """Find critical parameter values (phase transitions, optimal points)."""
    analysis = {}
    
    # Extract successful runs
    param_values = []
    final_losses = []
    
    for exp_group in results['experiments']:
        runs = [r for r in exp_group['runs'] if 'failed' not in r]
        if runs:
            param_values.append(exp_group['param_value'])
            final_losses.append(np.mean([r['final_loss'] for r in runs]))
    
    param_values = np.array(param_values)
    final_losses = np.array(final_losses)
    
    # Find optimal value
    optimal_idx = np.argmin(final_losses)
    analysis['optimal_value'] = param_values[optimal_idx]
    analysis['optimal_loss'] = final_losses[optimal_idx]
    
    # Find transition points (large derivatives)
    if len(param_values) > 3:
        derivatives = np.gradient(final_losses, param_values)
        transition_idx = np.argmax(np.abs(derivatives))
        analysis['transition_value'] = param_values[transition_idx]
        analysis['transition_derivative'] = derivatives[transition_idx]
    
    return analysis

def compute_correlations(results: Dict) -> Dict:
    """Compute correlations between metrics."""
    # Extract all metrics
    metrics = {
        'final_loss': [],
        'decoder_mse': [],
        'convergence_rate': [],
        'dimensionality': [],
        'weight_std': [],
    }
    
    for exp_group in results['experiments']:
        for run in exp_group['runs']:
            if 'failed' not in run:
                metrics['final_loss'].append(run['final_loss'])
                metrics['decoder_mse'].append(run['decoder_mse'])
                metrics['convergence_rate'].append(run['convergence_rate'])
                metrics['dimensionality'].append(run['activity_stats']['dimensionality'])
                metrics['weight_std'].append(run['weight_stats']['std'])
    
    # Compute correlation matrix
    metric_names = list(metrics.keys())
    n_metrics = len(metric_names)
    corr_matrix = np.zeros((n_metrics, n_metrics))
    
    for i, metric1 in enumerate(metric_names):
        for j, metric2 in enumerate(metric_names):
            corr_matrix[i, j] = np.corrcoef(metrics[metric1], metrics[metric2])[0, 1]
    
    return {'metrics': metric_names, 'correlation_matrix': corr_matrix}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print(f"Motor RNN Parameter Sweep: {SWEEP_PARAM}")
    print("=" * 60)
    
    # Run sweep
    results = run_parameter_sweep()
    
    # Analysis
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    
    # Find critical points
    critical = analyze_critical_points(results)
    print(f"\nOptimal {SWEEP_PARAM}: {critical['optimal_value']:.3f} (loss: {critical['optimal_loss']:.4f})")
    
    # Compute correlations
    correlations = compute_correlations(results)
    print("\nMetric correlations with final loss:")
    corr_with_loss = correlations['correlation_matrix'][0, :]
    for i, metric in enumerate(correlations['metrics'][1:], 1):
        print(f"  {metric}: {corr_with_loss[i]:.3f}")
    
    # Visualization
    if PLOT_RESULTS:
        print("\nGenerating plots...")
        fig = plot_sweep_results(results)
        plt.show()
    
    print(f"\nSweep complete. Results saved to: {SWEEP_DIR}")