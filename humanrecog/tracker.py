from typing import List, Dict

# import motrackers as mt
from boxmot import OCSORT
import numpy as np

from .data_protocols import Detection, Track

# from .detection import Detection
# from .track import Track

class Tracker():

    def __init__(self):
        self._tracker = OCSORT()
        # self._tracker = mt.centroid_kf_tracker.CentroidKF_Tracker()

    def step(self, detections: List[Detection], frame: np.ndarray) -> List[Track]:
        
        tracks = []
        for d in detections:
            tracks.append(Track(
                frame_id=0,
                id=d.id,
                tlwh=d.tlwh,
                confidence=d.confidence,
                # cls=d.cls,
                keypoints=d.keypoints
            ))

        # dets = []
        # for d in detections:
        #     x1 = d.tlwh[0]
        #     y1 = d.tlwh[1]
        #     x2 = d.tlwh[0] + d.tlwh[2]
        #     y2 = d.tlwh[1] + d.tlwh[3]
        #     score = d.confidence
        #     dets.append([x1, y1, x2, y2, score, d.cls])

        # if len(dets) == 0:
        #     dets = np.zeros((0, 6))
        # else:
        #     dets = np.array(dets)

        # try:
        #     np_tracks = self._tracker.update(dets, frame)
        # except Exception as e:
        #     print(e)
        #     import pdb; pdb.set_trace()

        # tracks = []
        # for t in np_tracks:
        #     # import pdb; pdb.set_trace()
        #     tracks.append(Track(
        #         frame_id=0,
        #         id=t[4],
        #         tlwh=np.array([t[0], t[1], t[2] - t[0], t[3] - t[1]]),
        #         confidence=t[5],
        #         keypoints=detections[int(t[-1])].keypoints
        #     ))

        return tracks
