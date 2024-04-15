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

def compute_iou(x_tlwh, y_tlwh) -> float:
    """Compute the Intersection over Union (IoU) of two bounding boxes"""
    
    x_tl, x_br = x_tlwh[:2], x_tlwh[:2] + x_tlwh[2:]
    y_tl, y_br = y_tlwh[:2], y_tlwh[:2] + y_tlwh[2:]

    tl = np.maximum(x_tl, y_tl)
    br = np.minimum(x_br, y_br)

    intersection = np.maximum(br - tl, 0)
    intersection_area = intersection[0] * intersection[1]

    x_area = x_tlwh[2] * x_tlwh[3]
    y_area = y_tlwh[2] * y_tlwh[3]

    union_area = x_area + y_area - intersection_area

    return intersection_area / union_area



def wraps_detection(t: Track, d: Detection) -> bool:
    """Is the detection WRAPPED by the Track bounding box?"""
    
    tl = t.tlwh[:2]
    br = (tl + t.tlwh[2:])

    tl_d = d.tlwh[:2]
    br_d = (tl + d.tlwh[2:])

    return bool(np.all(tl < tl_d) and np.all(br_d < br))


def crop(track: Track, frame: np.ndarray) -> np.ndarray:
    """Crop the image given the bounding box"""
    tl = track.bbox[:2].astype(int)
    br = (tl + track.bbox[2:]).astype(int)

    return frame[tl[1]:br[1], tl[0]:br[0]]