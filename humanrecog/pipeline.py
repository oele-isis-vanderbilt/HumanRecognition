import pathlib
import logging
from typing import List, Tuple, Dict

import numpy as np

from .detection import Detection
from .detector import Detector
from .data_protocols import PipelineResults
from .track import Track
from .reid import ReID
from .tracker import Tracker

logger = logging.getLogger('')

class Pipeline:

    def __init__(self, person_weights: pathlib.Path, face_weights: pathlib.Path, device: str = 'cpu'):
        
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
                if track.wraps(head):
                    track.face = head
                    break

        return tracks

    def step(self, frame: np.ndarray) -> PipelineResults:

        # Perform detection
        person_detections = self.person_detector(frame)
        face_detections = self.face_detector(frame)

        # # Obtain detections
        # detections = self.detector(frame)

        # # Split between body and head
        # body_head_dict = self.split_body_and_face(detections[0])

        # # Process the detections to perform simple tracking
        # tracks = self.tracker.step(body_head_dict['body'])
        
        # # Match the body and face detections
        # tracks = self.match_head_to_track(tracks, body_head_dict['head'])

        # # Process Tracks to re-identify people
        # id_map, tracks = self.reid.step(frame, tracks)
        
        # # Update the tracker's IDs if any possible re-identification
        # self.tracker.update(id_map)

        # return tracks

        return PipelineResults(
            person_detections=person_detections,
            face_detections=face_detections
        )