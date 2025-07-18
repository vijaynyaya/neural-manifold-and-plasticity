# Exploring the learning dynamics of neural manifolds

Understanding how neural manifolds evolve during learning is essential for unraveling the neural basis of motor control and advancing applications in neuroprosthetics and robotics. This project aims to characterize how the task complexity, learning paradigm and learning phases affect the shape, separability and linearity of neural manifolds in recurrent neural networks (RNNs). 

By comparing manifold geometry and separability across these conditions, we aim to identify the conditions under which linear versus nonlinear dynamics dominate. Additionally, we will examine how architectural modifications such as biologically inspired decoding affect the latent structure of learned representations.

We will train RNNs to perform a center-out reach task inspired by Feulner and Clopath (2021), where network activity drives a cursor in a 2D space via a brain-computer interface (BCI). Task difficulty will be modulated by increasing the number of targets. Latent neural dynamics will be extracted using both linear (PCA) and nonlinear (ICA, Isomap) dimensionality reduction techniques.

We predict that: 1) manifold separability in linear latent spaces will decrease with increasing task complexity, 2) nonlinear dimensionality reduction will better capture manifold latent dynamics in more complex tasks, and 3) replacing a PCA decoder with a feedforward readout layer will generate distinct neural manifolds.

This work contributes to a deeper understanding of learning-related neural dynamics and could inform the design of more robust neural interfaces and motor rehabilitation protocols.

## Project Structure

```
neural-manifold-and-plasticity/
│
├── manifolks/             # Main Python package: core code, models, utilities
│   ├── __init__.py
│   ├── ...                # (other modules)
│
├── notebooks/             # Jupyter notebooks for experiments & simulation runs
│   ├── FORCE.ipynb
│   ├── RLS.ipynb
│   └── ...                # (other notebooks)
│
├── data/                  # (Optional) Data files for experiments
│   └── ...
│
├── figures/               # (Optional) Generated figures and plots
│   └── ...
│
├── LICENSE
└── README.md 
```

## Previous Work

1. Sadtler, P. T., Quick, K. M., Golub, M. D., Chase, S. M., Ryu, S. I., Tyler-Kabara, E. C., Yu, B. M., & Batista, A. P. (2014). Neural constraints on learning. Nature, 512(7515), 423–426. https://doi.org/10.1038/nature13665

2. Gallego, J. A., Perich, M. G., Miller, L. E., & Solla, S. A. (2017). Neural manifolds for the control of movement. Neuron, 94(5), 978–984. https://doi.org/10.1016/j.neuron.2017.05.025

3. Feulner, B., & Clopath, C. (2021). Neural manifold under plasticity in a goal-driven learning behaviour. PLOS Computational Biology, 17(2), e1008621. https://doi.org/10.1371/journal.pcbi.1008621

4. Feulner, B., Perich, M. G., Miller, L. E., Clopath, C., & Gallego, J. A. (2024). A neural implementation model of feedback-based motor learning. Nature Communications, 16, 1805. https://doi.org/10.1038/s41467-024-54738-5

5. Chang, J. C., Clopath, C., & Gallego, J. A. (2025). Neural signatures of motor memories emerge in neural network models. bioRxiv. https://doi.org/10.1101/2025.04.02.646788

6. Chang, J. C., Perich, M. G., Miller, L. E., Gallego, J. A., & Clopath, C. (2024). De novo motor learning creates structure in neural activity that shapes adaptation. Nature Communications, 15, 4084. https://doi.org/10.1038/s41467-024-48008-7
