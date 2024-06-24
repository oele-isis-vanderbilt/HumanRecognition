import pathlib
import logging
from typing import List, Tuple, Dict

import imutils
import numpy as np
import pandas as pd
from collections import defaultdict
from headpose_estimation import Headpose

from .detector import Detector
from .data_protocols import PipelineResults, Track, Detection
from .reid import ReID
from .tracker import Tracker
from .utils import scale_fix, estimate_gaze_vector, crop

logger = logging.getLogger('')

class Pipeline:

    def __init__(
            self, 
            person_weights: pathlib.Path, 
            face_weights: pathlib.Path,
            db: pathlib.Path,
            device: str = 'cpu'
        ):

        # Default empty values
        self.focal_length = 0
        self.camera_matrix = np.zeros((3, 3))
        self.dist_coeffs = np.zeros((4, 1))
 
        # Create objects
        self.step_id = 0
        self.person_detector = Detector(person_weights, device=device)
        self.face_detector = Detector(face_weights, device=device)
        self.tracker = Tracker() 
        self.reid = ReID(db=db, device=device)
        self.head_pose = Headpose(face_detection=False)

    def update_camera_attributes(self, frame: np.ndarray):

        # Camera matrix estimation
        self.focal_length = frame.shape[1]
        center = (frame.shape[1] / 2, frame.shape[0] / 2)
        self.camera_matrix = np.array(
            [[self.focal_length, 0, center[0]],
            [0, self.focal_length, center[1]],
            [0, 0, 1]], dtype="double"
        )
        self.dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

    def split_body_and_face(self, detections: List[Detection]) -> Dict[str, List[Detection]]:
        
        head_detections = []
        body_detections = []

        # Split
        for d in detections:
            if d.cls == 0:
                body_detections.append(d)
            elif d.cls == 1:
                head_detections.append(d)

        return {'body': body_detections, 'head': head_detections}

    def match_head_to_track(self, tracks: List[Track], faces: List[Detection]) -> List[Track]:
       
        for face in faces:
            matches = defaultdict(list)

            for track_id, track in enumerate(tracks):
                
                # Determine if the nose is within the face bounding box
                if isinstance(track.keypoints, np.ndarray):
                    nose_keypoint = track.keypoints[0, 0]

                    # Compute if the nose is within the face bounding box
                    xmin = face.tlwh[0]
                    ymin = face.tlwh[1]
                    xmax = face.tlwh[0] + face.tlwh[2]
                    ymax = face.tlwh[1] + face.tlwh[3]
                    if xmin <= nose_keypoint[0] <= xmax and ymin <= nose_keypoint[1] <= ymax:

                        # If match, compute the distance between the nose and
                        # the center of the face bounding box
                        face_center = (xmin + xmax) // 2, (ymin + ymax) // 2
                        distance = np.linalg.norm(np.array(face_center) - np.array(nose_keypoint))

                        matches['track_id'].append(track_id)
                        matches['distance'].append(distance)

            matches_df = pd.DataFrame(matches)

            # Pick the minimum distance
            if not matches_df.empty:
                track_id = matches_df[matches_df['distance'] == matches_df['distance'].min()]['track_id'].values[0]
                tracks[track_id].face = face

        return tracks

    def compute_head_pose(self, frame: np.ndarray, tracks: List[Track]) -> List[Track]:

        track_imgs = [] 
        crops = []
        for track in tracks:
            if isinstance(track.face, Detection):

                # Only perform the head pose estimation if the face has all visible keypoints
                face_keypoints_idx = [0, 1, 2, 3, 4]
                if any([(track.keypoints[0][i] == np.zeros((2))).all() for i in face_keypoints_idx]):
                    continue

                # Get the crop of the track
                img = crop(track.face, frame)
                track_imgs.append(track)
                crops.append(img)

        # Estimate the head pose
        if len(crops) > 0:
            success, yaw, pitch, roll = self.head_pose.detect_multiple_headpose(crops)

            # Assign the head pose to the track
            if success:
                for i, track in enumerate(track_imgs):
                    track.face_headpose = (yaw[i], pitch[i], roll[i])

        return tracks

    def step(self, frame: np.ndarray) -> PipelineResults:

        # If the first frame, update the camera attributes
        if self.step_id == 0:
            self.update_camera_attributes(frame)
            self.step_id += 1

        # Reduce the image size to speed up the process
        reduce_size = imutils.resize(frame, width=640)

        # Perform detection
        person_detections = self.person_detector(reduce_size, persist=True)
        face_detections = self.face_detector(reduce_size)

        # Fix the detection to match the original size
        person_detections = scale_fix(frame.shape[:2], reduce_size.shape[:2], person_detections)
        face_detections = scale_fix(frame.shape[:2], reduce_size.shape[:2], face_detections)

        # Process the detections to perform simple tracking
        tracks = self.tracker.step(person_detections)
        
        # Match the body and face detections
        tracks = self.match_head_to_track(tracks, face_detections)

        # Compute head pose for each track
        tracks = self.compute_head_pose(frame, tracks)

        # Process Tracks to re-identify people
        # reid_tracks = self.reid.step(frame, tracks)
        reid_tracks = []
        
        # # Update the tracker's IDs if any possible re-identification
        # self.tracker.update(id_map)

        # Increment the step
        self.step_id += 1

        # return tracks

        return PipelineResults(
            person_detections=person_detections,
            face_detections=face_detections,
            tracks=tracks,
            reid_tracks=reid_tracks
        )