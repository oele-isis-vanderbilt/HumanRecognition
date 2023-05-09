import numpy as np
from dataclasses import dataclass

@dataclass
class ReIDResult:
    
    bbox: np.ndarray # Nx4
    scores: np.ndarray # Nx1
    ids: np.ndarray # Nx1
