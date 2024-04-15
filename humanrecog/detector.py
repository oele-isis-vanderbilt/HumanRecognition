
import warnings
import ultralytics as ul
from .data_protocols import Detection

class Detector():
    def __init__(self, weights, imgsz=640, device="cpu", conf_thresh=0.4, iou_thresh=0.5):
        self.model = ul.YOLO(weights)
        # self.model.to(device)
        # self.imgsz = imgsz

        # self.model.conf = conf_thresh
        # self.model.iou = iou_thresh
        # self.model.agnostic = False  # NMS class-agnostic
        # self.model.multi_label = False  # NMS multiple labels per box
        # self.model.max_det = 1000  # maximum number of detections per image

    def predict(self, images):

        results = self.model.track(images, verbose=False) 

        all_detections = []
        bboxes = results[0].boxes.xywh.cpu()
        if results[0].boxes.id is not None:

            clss = results[0].boxes.cls.cpu().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            confs = results[0].boxes.conf.float().cpu().tolist()

            for box, cls, conf, track_id in zip(bboxes, clss, confs, track_ids):

                # Convert xywh (centroid) to tlwh
                box = box.numpy()
                box[0] = box[0] - box[2] / 2
                box[1] = box[1] - box[3] / 2

                all_detections.append(Detection(
                    tlwh=box, 
                    confidence=float(conf), 
                    cls=cls
                ))
                # import pdb; pdb.set_trace()
        
        return all_detections

    def __call__(self, images):
        return self.predict(images)