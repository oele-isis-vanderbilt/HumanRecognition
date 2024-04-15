import pathlib
import logging
from typing import List, Tuple, Dict

import imutils
import numpy as np

from .detection import Detection
from .detector import Detector
from .data_protocols import PipelineResults, Track
from .reid import ReID
from .tracker import Tracker
from .utils import scale_fix, wraps_detection

logger = logging.getLogger('')

class Pipeline:

    def __init__(
            self, 
            person_weights: pathlib.Path, 
            face_weights: pathlib.Path, 
            device: str = 'cpu'
        ):
        
        # Create objects
        self.person_detector = Detector(person_weights, device=device)
        self.face_detector = Detector(face_weights, device=device)
        self.tracker = Tracker() 
        self.reid = ReID(device=device)

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

    def match_head_to_track(self, tracks: List[Track], heads: List[Detection]) -> List[Track]:
       
        # Match data
        for track in tracks:
            for head in heads:
                if wraps_detection(track,head):
                    track.face = head
                    break

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

        # # Obtain detections
        # detections = self.detector(frame)

        # # Split between body and head
        # body_head_dict = self.split_body_and_face(detections[0])

        # Process the detections to perform simple tracking
        tracks = self.tracker.step(person_detections)
        
        # Match the body and face detections
        tracks = self.match_head_to_track(tracks, face_detections)

        # # Process Tracks to re-identify people
        # id_map, tracks = self.reid.step(frame, tracks)
        
        # # Update the tracker's IDs if any possible re-identification
        # self.tracker.update(id_map)

        # return tracks

        return PipelineResults(
            person_detections=person_detections,
            face_detections=face_detections,
            tracks=tracks
        )