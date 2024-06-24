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

MODEL_POINTS = np.array([
    [-8.673, 0.1374, 19.843],   # Right ear
    [-3.334, -10.689, 22.039],  # Right eye ball center

    [ 8.673, 0.1374, 19.843],   # Left ear
    [ 3.334, -10.689, 22.039],  # Left eye ball center

    [-0.0305, -14.021, 17.594],              # Nose
])

# Make the noise (0,0,0) by shifting all points
MODEL_POINTS -= MODEL_POINTS[4]

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


def estimate_gaze_vector(keypoints, camera_matrix, dist_coeffs):

    # import pdb; pdb.set_trace()

    # Define 3D model points of the eyes and nose
    model_points = np.array([
        [-35, 32.7, -39.5],     # Right ear
        [-29.05, 32.7, -39.5],  # Right eye ball center

        [35, 32.7, -39.5],      # Left ear
        [29.05, 32.7, -39.5],   # Left eye ball center

        [0, 0, 0],              # Nose
    ])

    image_points = np.array([
        keypoints[4],  # Right ear
        keypoints[2],  # Right eye ball center
        keypoints[3],  # Left ear
        keypoints[1],  # Left eye ball center
        keypoints[0],   # Nose
    ], dtype='double')

    all_good = []
    for i in range(image_points.shape[0]):
        if not np.all(image_points[i] == 0):
            all_good.append(i)

    if len(all_good) < 4:
        return False, None, None
    
    image_points = image_points[all_good]
    # model_points = model_points[all_good]
    model_points = MODEL_POINTS[all_good]
    import pdb; pdb.set_trace()

    # Solve PnP problem to estimate rotation and translation vectors
    return cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)


def frontal_distance(yaw, pitch, roll):
    # Perfectly frontal head pose (0, 0, 0)
    yaw_frontal, pitch_frontal, roll_frontal = 0, 0, 0
    
    # Convert degrees to radians for computation
    yaw_rad = np.radians(yaw)
    pitch_rad = np.radians(pitch)
    roll_rad = np.radians(roll)
    
    # Compute the angular distance
    angular_distance = np.sqrt((yaw_rad - yaw_frontal)**2 +
                               (pitch_rad - pitch_frontal)**2 +
                               (roll_rad - roll_frontal)**2)
    
    # Convert the angular distance back to degrees
    angular_distance_deg = np.degrees(angular_distance)
    
    # Check if the angular distance is within the threshold
    return angular_distance_deg


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