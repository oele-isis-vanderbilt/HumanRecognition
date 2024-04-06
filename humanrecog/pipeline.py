import pathlib
import logging
from typing import List, Tuple, Dict

import numpy as np

from .detection import Detection
from .detector import Detector
from .results import ReIDResult
from .track import Track
from .reid import ReID
from .tracker import Tracker

logger = logging.getLogger('')

class MMReIDPipeline:

    def __init__(self, weights: pathlib.Path, device: str = 'cpu'):

        # Save inputs
        self.weights = weights
        
        # Create objects
        self.detector = Detector(weights, device=device)
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

    def step(self, frame: np.ndarray) -> List[Track]:

        # Obtain detections
        detections = self.detector(frame)

        # Split between body and head
        body_head_dict = self.split_body_and_face(detections[0])

        # Process the detections to perform simple tracking
        tracks = self.tracker.step(body_head_dict['body'])
        
        # Match the body and face detections
        tracks = self.match_head_to_track(tracks, body_head_dict['head'])

        # Process Tracks to re-identify people
        id_map, tracks = self.reid.step(frame, tracks)
        
        # Update the tracker's IDs if any possible re-identification
        self.tracker.update(id_map)

        return tracks
