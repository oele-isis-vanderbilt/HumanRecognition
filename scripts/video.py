import time
import pathlib
import os
import numpy as np
from collections import defaultdict
from tqdm import tqdm

import cv2
import humanrecog as hr

GIT_ROOT = pathlib.Path(os.path.abspath(__file__)).parent.parent
DATA_DIR = GIT_ROOT / "data"
WEIGHTS_DIR = GIT_ROOT / "weights"

def main():
    cap = cv2.VideoCapture(str(DATA_DIR / 'embodied_learning' / 'block-a-blue-day1-first-group-cam2.mp4'))
    pipeline = hr.Pipeline(
        WEIGHTS_DIR / 'yolov8n.pt', 
        WEIGHTS_DIR / 'yolov8n-face.pt', device='cpu',
        db=DATA_DIR / 'embodied_learning' / 'db'
    ) 

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in tqdm(range(length), total=length):

        tic = time.perf_counter()

        ret, frame = cap.read()

        # Inference
        results = pipeline.step(frame)

        # Render detection
        # frame = hr.vis.render_detections(frame, results.face_detections, pipeline.face_detector.model.names)
        # frame = hr.vis.render_detections(frame, results.person_detections, pipeline.person_detector.model.names)

        # Render tracking
        # frame = hr.vis.render_detections_tracks(frame, results.person_detections)
        # frame = hr.vis.render_tracks(frame, results.tracks)

        # Render ReID
        frame = hr.vis.render_face_reid(frame, results.reid_tracks)

        toc = time.perf_counter()
        cv2.putText(frame, f"FPS: {1 / (toc - tic):.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()