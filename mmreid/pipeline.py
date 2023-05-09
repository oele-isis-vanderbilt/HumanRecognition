import pathlib
from typing import List, Tuple

import numpy as np
import motrackers as mt

from .detection import Detection
from .detector import Detector
from .results import ReIDResult
from .track import Track

class MMReIDPipeline:

    def __init__(self, weights: pathlib.Path):

        # Save inputs
        self.weights = weights
        
        # Create objects
        self.detector = Detector(weights)
        self.tracker = mt.centroid_kf_tracker.CentroidKF_Tracker()

    def _prep_for_tracker(self, detections: List[Detection], skip_cls:int=1):
        
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

        return tracks

