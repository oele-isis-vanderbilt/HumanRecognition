from typing import List

import cv2
import numpy as np
from .data_protocols import Detection, Track

# from .track import Track
# from .detection import Detection

def render_tracks(frame: np.ndarray, tracks: List[Track]):
    
    for track in tracks:
      
        # Draw bounding box
        tl = track.tlwh[:2]
        br = tl + track.tlwh[2:]

        cv2.rectangle(frame, tuple(tl.astype(int)), tuple(br.astype(int)), (0,0,255), 2)
        cv2.putText(
            frame,
            str(track.id),
            tuple(tl.astype(int)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,0,255),
            2,
            2
        )

        # If head, draw that too
        if isinstance(track.face, Detection):
            tl = track.face.tlwh[:2]
            br = tl + track.face.tlwh[2:]
            cv2.rectangle(frame, tuple(tl.astype(int)), tuple(br.astype(int)), (0,0,255), 2)
            cv2.putText(
                frame,
                'f' + str(track.id),
                tuple(tl.astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,0,255),
                2,
                2
            )

    return frame

def render_detections_tracks(frame: np.ndarray, detections: List[Detection]):
    
    for detection in detections:
        tl = detection.tlwh[:2]
        br = tl + detection.tlwh[2:]

        cv2.rectangle(frame, tuple(tl.astype(int)), tuple(br.astype(int)), (0,0,255), 2)
        cv2.putText(
            frame,
            str(detection.id),
            tuple(tl.astype(int)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,0,255),
            2,
            2
        )

    return frame

def render_detections(frame: np.ndarray, detections: List[Detection], names: List[str]):
    
    for detection in detections:
        tl = detection.tlwh[:2]
        br = tl + detection.tlwh[2:]

        cv2.rectangle(frame, tuple(tl.astype(int)), tuple(br.astype(int)), (0,0,255), 2)
        cv2.putText(
            frame,
            names[int(detection.cls)],
            tuple(tl.astype(int)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,0,255),
            2,
            2
        )

    return frame