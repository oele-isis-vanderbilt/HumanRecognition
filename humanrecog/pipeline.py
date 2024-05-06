import pathlib
import logging
from typing import List, Tuple, Dict

import imutils
import numpy as np
import pandas as pd
from collections import defaultdict

from .detector import Detector
from .data_protocols import PipelineResults, Track, Detection
from .reid import ReID
from .tracker import Tracker
from .utils import scale_fix, wraps_detection, compute_iou

logger = logging.getLogger('')

class Pipeline:

    def __init__(
            self, 
            person_weights: pathlib.Path, 
            face_weights: pathlib.Path,
            db: pathlib.Path,
            device: str = 'cpu'
        ):
        
        # Create objects
        self.person_detector = Detector(person_weights, device=device)
        self.face_detector = Detector(face_weights, device=device)
        self.tracker = Tracker() 
        self.reid = ReID(db=db, device=device)

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
                iou = compute_iou(face.tlwh, track.tlwh)
                y_centroid_distance = abs(face.tlwh[1] - track.tlwh[1])
                matches['iou'].append(iou)
                matches['track_id'].append(track_id)
                matches['y_distance'].append(y_centroid_distance)

            matches_df = pd.DataFrame(matches)
            matches_df = matches_df[matches_df['iou'] > 0]

            # PIck the highest y_distance
            if not matches_df.empty:
                track_id = matches_df[matches_df['y_distance'] == matches_df['y_distance'].max()]['track_id'].values[0]
                tracks[track_id].face = face

        return tracks

    def step(self, frame: np.ndarray) -> PipelineResults:

        # Reduce the image size to speed up the process
        reduce_size = imutils.resize(frame, width=640)

        # Perform detection
        person_detections = self.person_detector(reduce_size)
        face_detections = self.face_detector(reduce_size)

        # Fix the detection to match the original size
        person_detections = scale_fix(frame.shape[:2], reduce_size.shape[:2], person_detections)
        face_detections = scale_fix(frame.shape[:2], reduce_size.shape[:2], face_detections)

        # Process the detections to perform simple tracking
        tracks = self.tracker.step(person_detections)
        
        # Match the body and face detections
        tracks = self.match_head_to_track(tracks, face_detections)

        # Process Tracks to re-identify people
        # reid_tracks = self.reid.step(frame, tracks)
        reid_tracks = []
        
        # # Update the tracker's IDs if any possible re-identification
        # self.tracker.update(id_map)

        # return tracks

        return PipelineResults(
            person_detections=person_detections,
            face_detections=face_detections,
            tracks=tracks,
            reid_tracks=reid_tracks
        )