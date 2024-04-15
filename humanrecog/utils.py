from .data_protocols import Detection
from typing import List

def scale_fix(original_size, smaller_size, detections: List[Detection]):
    
    scale_x = original_size[1] / smaller_size[1]
    scale_y = original_size[0] / smaller_size[0]

    for d in detections:
        d.tlwh[0] *= scale_x
        d.tlwh[1] *= scale_y
        d.tlwh[2] *= scale_x
        d.tlwh[3] *= scale_y

    return detections