import numpy as np
import matplotlib.pyplot as plt

def plot_explained_variance(eigenvalues, target_variance=0.8):

    total_components = len(eigenvalues)
    cumulative_explained_variance = np.append(0, np.cumsum(eigenvalues) / np.sum(eigenvalues)) 
    required_components = np.argmax(cumulative_explained_variance >= target_variance)

    print(f"Number of components needed for {target_variance} % explained - {required_components}")
    
    # Plot settings
    plt.plot(cumulative_explained_variance, label="Cumulative Explained Variance")
    plt.pltvline(required_components, linestyle="--", color="black", label=f"{target_variance} % Threshold")

    yticks = np.arange(0, 1.1, 0.25)
    plt.yticks(yticks, yticks, fontsize=14)
    plt.xlabel("Number of components", fontsize=18)
    plt.ylabel("Explained variance (%)", fontsize=18)
    plt.legend()
    plt.show()