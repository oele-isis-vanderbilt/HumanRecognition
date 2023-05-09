from typing import List

import numpy as np

class Track:

    def __init__(self, track: List):
        
        self.id = track[1]
        self.bbox = np.array(track[2:6])
        self.confidence = track[6]
