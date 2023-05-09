import pathlib
from typing import List, Tuple
import logging

import numpy as np
import motrackers as mt

from .detection import Detection
from .detector import Detector
from .results import ReIDResult
from .track import Track
from .reid import ReID

logger = logging.getLogger('')

class MMReIDPipeline:

    def __init__(self, weights: pathlib.Path, device: str = 'cpu'):

        # Save inputs
        self.weights = weights
        
        # Create objects
        self.detector = Detector(weights, device=device)
        self.tracker = mt.centroid_kf_tracker.CentroidKF_Tracker()
        self.reid = ReID(device=device)

    def _prep_for_tracker(self, detections: List[Detection], skip_cls:int=1):
        """Convert list of items into numpy array of items"""
        
        bboxes = []
        scores = []
        class_ids = []

        for d in detections:
            if d.cls != skip_cls:
                bboxes.append(np.array(d.tlwh))
                scores.append(d.confidence)
                class_ids.append(d.cls)
        
        bboxes = np.stack(bboxes)
        scores = np.stack(scores)
        class_ids = np.stack(class_ids)

        return bboxes, scores, class_ids

    def step(self, frame: np.ndarray) -> List[Track]:

        # Obtain detections
        detections = self.detector(frame)

        # Process detections with tracker
        bboxes, scores, class_ids = self._prep_for_tracker(detections[0], skip_cls=1)
        tracks = self.tracker.update(bboxes, scores, class_ids)
        tracks = [Track(x) for x in tracks]
        
        logger.debug(tracks)

        # Process Tracks to re-identify people
        id_map, tracks = self.reid.step(frame, tracks)

        # Update tracker to match new ID
        for old_id, new_id in id_map.items():
            if old_id != new_id:
                try:
                    track = self.tracker.tracks.pop(old_id)
                    track.id = new_id
                    self.tracker.tracks[new_id] = track
                except KeyError:
                    ...

        return tracks
