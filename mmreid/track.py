from typing import List, Optional

import numpy as np

from .detection import Detection

class Track:

    frame_id: int
    id: int
    bbox: np.ndarray # Shape (4,)
    confidence: float
    embedding: Optional[np.ndarray] = None

    # Multimodal Data
    face: Optional[Detection] = None
    face_embedding: Optional[np.ndarray] = None
    

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
        """Crop the image given the bounding box"""
        tl = self.bbox[:2].astype(int)
        br = (tl + self.bbox[2:]).astype(int)

        return frame[tl[1]:br[1], tl[0]:br[0]]

    def wraps(self, d: Detection) -> bool:
        """Is the detection WRAPPED by the Track bounding box?"""
        
        tl = self.bbox[:2]
        br = (tl + self.bbox[2:])

        tl_d = d.tlwh[:2]
        br_d = (tl + d.tlwh[2:])

        return bool(np.all(tl < tl_d) and np.all(br_d < br))
