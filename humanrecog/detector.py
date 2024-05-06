import numpy as np
import ultralytics as ul

from .data_protocols import Detection

class Detector():
    def __init__(self, weights, device="cpu"):

        # Save parameters
        self.model = ul.YOLO(weights).to(device)

    def __call__(self, image: np.ndarray, persist=False):
        results = self.model.track(image, persist=persist, verbose=False)  
        all_detections = []
        bboxes = results[0].boxes.xywh.cpu()
        if results[0].boxes.id is not None:

            clss = results[0].boxes.cls.cpu().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            confs = results[0].boxes.conf.float().cpu().tolist()

            for id, (box, cls, conf, track_id) in enumerate(zip(bboxes, clss, confs, track_ids)):
  
                # Convert xywh (centroid) to tlwh
                box = box.numpy()
                box[0] = box[0] - box[2] / 2
                box[1] = box[1] - box[3] / 2


                if results[0].keypoints:
                    # import pdb; pdb.set_trace()
                    all_detections.append(Detection(
                        id=track_id,
                        tlwh=box, 
                        confidence=float(conf), 
                        cls=cls,
                        keypoints=results[0].keypoints[id].xy.cpu().numpy()
                    ))
                else:
                    all_detections.append(Detection(
                        id=track_id,
                        tlwh=box, 
                        confidence=float(conf), 
                        cls=cls
                    ))
        
        return all_detections