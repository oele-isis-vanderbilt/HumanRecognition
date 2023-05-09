import numpy as np
from typing import List

import cv2

from .track import Track

def render(frame: np.ndarray, tracks: List[Track]):
    
    for track in tracks:
      
        # Draw bounding box
        tl = track.bbox[:2]
        br = tl + track.bbox[2:]
        # import pdb; pdb.set_trace()
        cv2.rectangle(frame, tuple(tl.astype(int)), tuple(br.astype(int)), (0,255,0), 1)
        cv2.putText(
            frame,
            str(track.id),
            tuple(tl.astype(int)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,0,255),
            1,
            2
        )

    return frame
