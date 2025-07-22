import numpy as np
import matplotlib.pyplot as plt

def plot_explained_variance(eigenvalues, target_variance=0.8):

    total_components = len(eigenvalues)
    cumulative_explained_variance = np.append(0, np.cumsum(eigenvalues) / np.sum(eigenvalues)) 
    required_components = np.argmax(cumulative_explained_variance >= target_variance)

    print(f"Number of components needed for {target_variance} % explained - {required_components}")
    
    # Plot settings
    fig, ax = plt.subplots()
    ax.plot(cumulative_explained_variance, label="Cumulative Explained Variance")
    # plot a vertical line at the required components
    ax.axvline(required_components, linestyle="--", color="black", label=f"{target_variance} % Threshold")

    yticks = np.arange(0, 1.1, 0.25)
    ax.set_yticks(yticks, yticks, fontsize=14)
    ax.set_xlabel("Number of components", fontsize=18)
    ax.set_ylabel("Explained variance (%)", fontsize=18)
    ax.legend()
    plt.show()