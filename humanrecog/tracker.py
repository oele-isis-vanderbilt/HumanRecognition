from typing import List, Dict

import motrackers as mt
import numpy as np

from .data_protocols import Detection, Track

# from .detection import Detection
# from .track import Track

class Tracker():

    def __init__(self):

        self._tracker = mt.centroid_kf_tracker.CentroidKF_Tracker()
    
    def _prep_for_tracker(self, detections: List[Detection]):
        """Convert list of items into numpy array of items"""
        
        bboxes = []
        scores = []
        class_ids = []

        for d in detections:
            bboxes.append(np.array(d.tlwh))
            scores.append(d.confidence)
            class_ids.append(d.cls)
        
        bboxes = np.stack(bboxes)
        scores = np.stack(scores)
        class_ids = np.stack(class_ids)

        return bboxes, scores, class_ids

    def update(self, id_map: Dict[int, int]):
        
        # Update tracker to match new ID
        for old_id, new_id in id_map.items():
            if old_id != new_id:
                try:
                    track = self._tracker.tracks.pop(old_id)
                    track.id = new_id
                    self._tracker.tracks[new_id] = track
                except KeyError:
                    ...

    def step(self, detections: List[Detection]) -> List[Track]:
        
        # Process detections with tracker
        bboxes, scores, class_ids = self._prep_for_tracker(detections)
        tracks = self._tracker.update(bboxes, scores, class_ids)
        tracks = [Track(
            frame_id=x[0],
            id=x[1],
            tlwh=np.array(x[2:6]),
            confidence=x[6],
        ) for x in tracks]

        # Match tracks to keypoints
        for track in tracks:
            for detection in detections:
                if (track.tlwh == detection.tlwh).all():
                    track.keypoints = detection.keypoints
                    break

        return tracks
