from typing import List
import hashlib
from math import cos, sin, pi

import cv2
import numpy as np
from .data_protocols import Detection, Track, ReIDTrack

# keypoints
labels = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear", # 0, 1, 2, 3, 4
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", # 5, 6, 7, 8, 9, 10
    "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle", # 11, 12, 13, 14, 15, 16
]


def generate_color_from_hash(seed):
    """
    Generate a color based on an input integer using a hashing function.
    
    Parameters:
    - seed: Input integer
    
    Returns:
    - color: Tuple (R, G, B) representing the generated color
    """
    # Convert the input integer to a string and encode it
    seed_str = str(seed).encode()
    
    # Generate a hash using hashlib
    hash_value = hashlib.sha256(seed_str).hexdigest()
    
    # Extract RGB values from the hash value
    r = int(hash_value[:2], 16)  # Extract the first two characters as the red component
    g = int(hash_value[2:4], 16)  # Extract the next two characters as the green component
    b = int(hash_value[4:6], 16)  # Extract the last two characters as the blue component
    
    return (r, g, b)


def draw_skeleton_pose(image, keypoints, color=(0, 255, 0)):
    # Define the skeleton connections
    connections = [
        (0, 1), (0, 2), (1, 3), (2, 4), # Head
        (10, 8), (8, 6), (6, 5), (5, 7), (7, 9), # Sholder and Arms
        (5, 11), (11, 12), (12, 6), # Torso
        (11, 13), (13, 15), (11, 12), (12, 14), (14, 16), # Legs
        (3, 5), (4, 6)
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


# def draw_head_pose(image, rvec, tvec, camera_matrix, dist_coeffs):
    
#     (s, _) = cv2.projectPoints(
#         np.array([(0.0, 0.0, 0)]),
#         rvec, 
#         tvec, 
#         camera_matrix,
#         dist_coeffs
#     )

#     (e, _) = cv2.projectPoints(
#         np.array([(0.0, 0.0, 100)]),
#         rvec, 
#         tvec, 
#         camera_matrix,
#         dist_coeffs
#     )

#     # Draw the line
#     cv2.line(image, tuple(s.ravel().astype(int)), tuple(e.ravel().astype(int)), (0, 255, 0), 2)

#     return image

def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size = 100):
    # Referenced from HopeNet https://github.com/natanielruiz/deep-head-pose
    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),2)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),2)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),2)
    return img


def render_tracks(frame: np.ndarray, tracks: List[Track], camera_matrix, dist_coeffs):
    
    for track in tracks:
      
        # Draw bounding box
        tl = track.tlwh[:2]
        br = tl + track.tlwh[2:]

        # Determine the color
        color = generate_color_from_hash(track.id)

        cv2.rectangle(frame, tuple(tl.astype(int)), tuple(br.astype(int)), color, 2)
        cv2.putText(
            frame,
            str(track.id),
            tuple(tl.astype(int)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2,
            2
        )

        # If skeleton keypoints, draw them
        if isinstance(track.keypoints, np.ndarray):
            frame = draw_skeleton_pose(frame, track.keypoints, color)

        # If head, draw that too
        if isinstance(track.face, Detection):
            tl = track.face.tlwh[:2]
            br = tl + track.face.tlwh[2:]
            cv2.rectangle(frame, tuple(tl.astype(int)), tuple(br.astype(int)), color, 2)
            cv2.putText(
                frame,
                'f' + str(track.id),
                tuple(tl.astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2,
                2
            )

        if isinstance(track.face_headpose, tuple):

            x_center = track.face.tlwh[0] + track.face.tlwh[2] / 2
            y_center = track.face.tlwh[1] + track.face.tlwh[3] / 2
            size = abs(track.face.tlwh[2] // 2)

            frame = draw_axis(
                frame,
                track.face_headpose[0], 
                track.face_headpose[1],
                track.face_headpose[2],
                tdx=x_center,
                tdy=y_center,
                size=size
            )

            # Draw the angular distance
            angular_distance = track.face_frontal
            cv2.putText(
                frame,
                f"{angular_distance:.2f}",
                (int(x_center), int(y_center)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
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