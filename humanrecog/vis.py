from typing import List

import cv2
import numpy as np
from .data_protocols import Detection, Track, ReIDTrack

# Keypoints
labels = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear", # 0, 1, 2, 3, 4
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", # 5, 6, 7, 8, 9, 10
    "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle", # 11, 12, 13, 14, 15, 16
]

def draw_skeleton_pose(image, keypoints, color=(0, 255, 0)):
    # Define the skeleton connections
    # connections = [
    #     (0, 1), (1, 2), (2, 3),
    #     (0, 4), (4, 5), (5, 6),
    #     (0, 7), (7, 8), (8, 9),
    #     (9, 10), (8, 11), (11, 12),
    #     (12, 13), (8, 14), (14, 15),
    #     (15, 16)
    # ]
    connections = [
        (0, 1), (0, 2), (1, 3), (2, 4), # Head
        (10, 8), (8, 6), (6, 5), (5, 7), (7, 9), # Sholder and Arms
        (5, 11), (11, 12), (12, 6), # Torso
        (11, 13), (13, 15), (11, 12), (12, 14), (14, 16) # Legs

    ]

    # Draw keypoints
    for point in keypoints[0]:
        if np.all(point == 0):
            continue
        cv2.circle(image, (int(point[0]), int(point[1])), 5, color, -1)

    # Draw skeleton connections
    for connection in connections:
        if np.all(keypoints[0][connection[0]] == 0) or np.all(keypoints[0][connection[1]] == 0):
            continue
        cv2.line(image, (int(keypoints[0][connection[0]][0]), int(keypoints[0][connection[0]][1])),
                 (int(keypoints[0][connection[1]][0]), int(keypoints[0][connection[1]][1])), color, 2)

    return image


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

        # If skeleton keypoints, draw them
        # import pdb; pdb.set_trace()
        if isinstance(detection.keypoints, np.ndarray):
            frame = draw_skeleton_pose(frame, detection.keypoints)

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

def render_face_reid(frame: np.ndarray, reid_tracks: List[ReIDTrack]):

    for reid_track in reid_tracks:
        tl = reid_track.track.tlwh[:2]
        br = tl + reid_track.track.tlwh[2:]

        cv2.rectangle(frame, tuple(tl.astype(int)), tuple(br.astype(int)), (0,0,255), 2)
        cv2.putText(
            frame,
            f"{reid_track.name} ({reid_track.cosine:.2f})",
            tuple(tl.astype(int)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,0,255),
            2,
            2
        )

    return frame