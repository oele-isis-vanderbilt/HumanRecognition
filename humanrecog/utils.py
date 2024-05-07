from typing import List
import pathlib
import logging

import cv2
import torch
import pandas as pd
# from deepface import DeepFace
import facenet_pytorch
from tqdm import tqdm
import numpy as np

from .data_protocols import Detection, Track

logger = logging.getLogger('')

def scale_fix(original_size, smaller_size, detections: List[Detection]):
    
    scale_x = original_size[1] / smaller_size[1]
    scale_y = original_size[0] / smaller_size[0]

    for d in detections:
        d.tlwh[0] *= scale_x
        d.tlwh[1] *= scale_y
        d.tlwh[2] *= scale_x
        d.tlwh[3] *= scale_y

        if d.keypoints is not None:
            d.keypoints[..., 0] *= scale_x
            d.keypoints[..., 1] *= scale_y

    return detections

def compute_iou(x_tlwh, y_tlwh) -> float:
    """Compute the Intersection over Union (IoU) of two bounding boxes"""
    
    x_tl, x_br = x_tlwh[:2], x_tlwh[:2] + x_tlwh[2:]
    y_tl, y_br = y_tlwh[:2], y_tlwh[:2] + y_tlwh[2:]

    tl = np.maximum(x_tl, y_tl)
    br = np.minimum(x_br, y_br)

    intersection = np.maximum(br - tl, 0)
    intersection_area = intersection[0] * intersection[1]

    x_area = x_tlwh[2] * x_tlwh[3]
    y_area = y_tlwh[2] * y_tlwh[3]

    union_area = x_area + y_area - intersection_area

    return intersection_area / union_area



def wraps_detection(t: Track, d: Detection) -> bool:
    """Is the detection WRAPPED by the Track bounding box?"""
    
    tl = t.tlwh[:2]
    br = (tl + t.tlwh[2:])

    tl_d = d.tlwh[:2]
    br_d = (tl + d.tlwh[2:])

    return bool(np.all(tl < tl_d) and np.all(br_d < br))


def crop(track: Track, frame: np.ndarray) -> np.ndarray:
    """Crop the image given the bounding box"""
    tl = track.tlwh[:2].astype(int)
    br = (tl + track.tlwh[2:]).astype(int)

    return frame[tl[1]:br[1], tl[0]:br[0]]


def estimate_head_pose(keypoints):
    nose = keypoints[0]
    left_eye = keypoints[1]
    right_eye = keypoints[2]
    left_ear = keypoints[3]
    right_ear = keypoints[4]

    # Calculate vectors representing lines from eyes to ears
    left_eye_to_ear = np.array([left_ear[0] - left_eye[0], left_ear[1] - left_eye[1]])
    right_eye_to_ear = np.array([right_ear[0] - right_eye[0], right_ear[1] - right_eye[1]])

    # Calculate the angles between the eyes and ears
    angle_left = np.arctan2(left_eye_to_ear[1], left_eye_to_ear[0]) * 180 / np.pi
    angle_right = np.arctan2(right_eye_to_ear[1], right_eye_to_ear[0]) * 180 / np.pi

    # Calculate the average angle
    average_angle = (angle_left + angle_right) / 2

    return average_angle


def facenet_pytorch_preprocessing(img: np.ndarray) -> torch.Tensor:
    """Preprocess an image for the facenet_pytorch model"""
    img = cv2.resize(img.astype(np.float32), (160, 160))
    # img = cv2.resize(img.astype(np.float32), (100,100))
    img = np.moveaxis(img, -1, 0)
    tensor = facenet_pytorch.prewhiten(torch.from_numpy(img).unsqueeze(dim=0))
    return tensor


def load_db_representation(db: pathlib.Path, model_name: str) -> pd.DataFrame:
    pkl_representation = db / f'representation_{model_name}.pkl'

    # Create an inception resnet (in eval mode):
    resnet = facenet_pytorch.InceptionResnetV1(pretrained='vggface2').eval()

    # If cuda is available, use it
    if torch.cuda.is_available():
        resnet = resnet.cuda()

    # Load the database
    if pkl_representation.exists():
        reid_df = pd.read_pickle(pkl_representation)
        return reid_df

    # Create the database
    else:

        reid_dict = {
            'id': [],
            'name': [],
            'face_embeddings': [],
            'person_embeddings': [],
            'last_seen_step_id': []
        }
        logger.debug("Creating the database representation")
        for idx, folder in tqdm(enumerate(db.iterdir()), total=len(list(db.iterdir()))):

            if folder.is_dir():
                embeddings = []
                for img in folder.iterdir():
                    if img.suffix in ['.jpg', '.png']:


                        # Load the image
                        img = cv2.imread(str(img))

                        # Apply preprocessing
                        tensor = facenet_pytorch_preprocessing(img)

                        # If cuda is available, use it
                        if torch.cuda.is_available():
                            tensor = tensor.cuda()

                        # Save embedding as a numpy array
                        embedding = resnet(tensor)
                        embeddings.append(embedding.cpu().detach().numpy().squeeze())


                all_embeddings = np.stack([np.array(e) for e in embeddings])
                reid_dict['id'].append(idx)
                reid_dict['name'].append(folder.name)
                reid_dict['face_embeddings'].append(all_embeddings)
                reid_dict['person_embeddings'].append(np.empty(shape=(0, 512)))
                reid_dict['last_seen_step_id'].append(-1)

        reid_df = pd.DataFrame(reid_dict) 
        reid_df.to_pickle(pkl_representation)

        return reid_df