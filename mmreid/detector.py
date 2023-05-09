
import warnings
import yolov5
from .detection import Detection

class Detector():
    def __init__(self, weights, imgsz=640, device="cpu", conf_thresh=0.4, iou_thresh=0.5):
        self.model = yolov5.load(weights)
        self.model.to(device)
        self.imgsz = imgsz

        self.model.conf = conf_thresh
        self.model.iou = iou_thresh
        self.model.agnostic = False  # NMS class-agnostic
        self.model.multi_label = False  # NMS multiple labels per box
        self.model.max_det = 1000  # maximum number of detections per image

    def predict(self, images: list):
        results = self.model(images, size=self.imgsz)

        all_detections = []
        for predictions in results.pred:
            detections_per_image = []
            for det in predictions:
                if det is not None and len(det):
                    det = det.detach().cpu().numpy()
                    tlbr = det[:4]
                    tlbr[2:] -= tlbr[:2]
                    detections_per_image.append(Detection(det[:4], det[4], det[5]))
            all_detections.append(detections_per_image)
        
        return all_detections

    def __call__(self, images):
        return self.predict(images)
