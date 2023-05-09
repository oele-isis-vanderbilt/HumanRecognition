from typing import List

import numpy as np

class Track:

    frame_id: int
    id: int
    bbox: np.ndarray # Shape (4,)
    confidence: float

    def __init__(self, track: List):
        
        self.frame_id = track[0]
        self.id = track[1]
        self.bbox = np.array(track[2:6])
        self.confidence = track[6]

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"<Track id={self.id}, confidence={self.confidence:.2f}>"

    def crop(self, frame: np.ndarray) -> np.ndarray:
        # Draw bounding box
        tl = self.bbox[:2].astype(int)
        br = (tl + self.bbox[2:]).astype(int)

        return frame[tl[1]:br[1], tl[0]:br[0]]
