from typing import List

import numpy as np

from .data_protocols import Detection, Track

def scale_fix(original_size, smaller_size, detections: List[Detection]):
    
    scale_x = original_size[1] / smaller_size[1]
    scale_y = original_size[0] / smaller_size[0]

    for d in detections:
        d.tlwh[0] *= scale_x
        d.tlwh[1] *= scale_y
        d.tlwh[2] *= scale_x
        d.tlwh[3] *= scale_y

    return detections

def wraps_detection(t: Track, d: Detection) -> bool:
    """Is the detection WRAPPED by the Track bounding box?"""
    
    tl = t.bbox[:2]
    br = (tl + t.bbox[2:])

    tl_d = d.tlwh[:2]
    br_d = (tl + d.tlwh[2:])

    return bool(np.all(tl < tl_d) and np.all(br_d < br))
