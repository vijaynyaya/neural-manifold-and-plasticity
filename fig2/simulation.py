"""
Simulation using Perfect Feedback Signal (Fig 2)
"""
# %% import packages
import numpy as np
import os
from pathlib import Path
import sklearn.linear_model as lm
from typing import Literal

# Output Directory
OUT_DIR = Path(__file__).parents[1] / 'data'
OUT_DIR.mkdir(exist_ok=True, parents=True)


# Data Types
IStimulusType = Literal["constant", "linear", "normal"] 


# PARAMETERS
RANDOM_SEED = 2
DT = 0.01
T = 2
TIME = np.arange(0, T, DT)
T_STEPS =  len(TIME)
TARGETS = 6
STIMULUS_TYPE: IStimulusType = "constant"
TARGET_MAX = 0.2
PULSE_LENGTH = int(TARGET_MAX / DT)
# network specifications
N = 800
G = 1.5
P = 0.1
TAU = 0.1
# initial network learning
N_TRIALS_LEARN = 80
LEARNING_RATE = 20.
# BCI manifold calculation
N_TRIALS_BCI_MANIFOLD = 50
MANIFOLD_NDIM = 10
# network relearning after perturbations
N_TRIALS_RELEARN = 80
LEARNING_RATE_REC = 20.
