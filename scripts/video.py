import time
import pathlib
import os
import numpy as np
from collections import defaultdict

import cv2
import humanrecog as hr
# from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

GIT_ROOT = pathlib.Path(os.path.abspath(__file__)).parent.parent
DATA_DIR = GIT_ROOT / "data"
WEIGHTS_DIR = GIT_ROOT / "weights"

def render_detections(frame, results, names):
    
    boxes = results[0].boxes.xyxy.cpu()

    if results[0].boxes.id is not None:

        # Extract prediction results
        clss = results[0].boxes.cls.cpu().tolist()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        confs = results[0].boxes.conf.float().cpu().tolist()

        # Annotator Init
        annotator = Annotator(frame, line_width=2)

        for box, cls, track_id in zip(boxes, clss, track_ids):
            annotator.box_label(box, color=colors(int(cls), True), label=names[int(cls)])

            # Store tracking history
            # track = track_history[track_id]
            # track.append((int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)))
            # if len(track) > 30:
            #     track.pop(0)

            # Plot tracks
            # points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
            # cv2.circle(frame, (track[-1]), 7, colors(int(cls), True), -1)
            # cv2.polylines(frame, [points], isClosed=False, color=colors(int(cls), True), thickness=2)

def main():
    cap = cv2.VideoCapture(str(DATA_DIR / 'embodied_learning' / 'block-a-blue-day1-first-group-cam2.mp4'))
    pipeline = hr.Pipeline(WEIGHTS_DIR / 'yolov8n.pt', WEIGHTS_DIR / 'yolov8n-face.pt') 
    # model = YOLO(WEIGHTS_DIR / 'yolov8n.pt').to('cuda')
    # face_model = YOLO(WEIGHTS_DIR / 'yolov8n-face.pt').to('cuda')
    # names = model.model.names
    # face_names = face_model.model.names

    while True:

        tic = time.perf_counter()

        ret, frame = cap.read()

        # Inference
        results = pipeline.step(frame)
        # results = model.track(frame, verbose=False)
        # results_face = face_model.track(frame, verbose=False)

        # Render information
        frame = hr.vis.render_detections(frame, results.face_detections, pipeline.face_detector.model.names)
        frame = hr.vis.render_detections(frame, results.person_detections, pipeline.person_detector.model.names)
        # render_detections(frame, results, names)
        # render_detections(frame, results_face, face_names)


        # Perform tracking

        toc = time.perf_counter()
        cv2.putText(frame, f"FPS: {1 / (toc - tic):.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()