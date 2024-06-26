from typing import List, Tuple, Dict, Optional, Union
import logging
import pathlib
import pickle
from collections import defaultdict
import time

import torch
import numpy as np
import torch.nn.functional as F
from torchreid.reid.utils import FeatureExtractor
import facenet_pytorch
# from deepface import DeepFace
import pandas as pd

from .data_protocols import Track, ReIDTrack, Detection
from .utils import crop, load_db_representation, facenet_pytorch_preprocessing
# from .database import Database

logger = logging.getLogger('humanrecog')

def find_cosine_distance(
    source_representation: Union[np.ndarray, list], test_representation: Union[np.ndarray, list]
) -> np.float64:
    """
    Find cosine distance between two given vectors
    Args:
        source_representation (np.ndarray or list): 1st vector
        test_representation (np.ndarray or list): 2nd vector
    Returns
        distance (np.float64): calculated cosine distance
    """
    if isinstance(source_representation, list):
        source_representation = np.array(source_representation)

    if isinstance(test_representation, list):
        test_representation = np.array(test_representation)

    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

def compute_matrix_cosine(m: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Cosine similarity between a matrix and a vector.

    Args:
        embeddings (np.ndarray): NxM matrix
        float_vector (np.ndarray): M vector

    Returns:
        np.ndarray: N vector of cosine similarities
    """
    
    return np.array([find_cosine_distance(row, v) for row in m])

def selection_procedure(cosine_vectors: List[np.ndarray], threshold: float) -> Tuple[bool, float, int]:

    if len(cosine_vectors) > 0:

        # Create a vector of the same length for the ids
        ids = [np.array([i] * cosine_vector.shape[0]) for i, cosine_vector in enumerate(cosine_vectors)]

        # Float cosine and ids
        flat_cosine = np.concatenate(cosine_vectors)
        flat_ids = np.concatenate(ids)

        # Find the top-k values
        top_k = np.argsort(flat_cosine)[:5]

        # Get the cosine and ids
        top_cosine = flat_cosine[top_k]
        top_ids = flat_ids[top_k]

        # Get the median cosine for each id
        medians = np.array([np.median(top_cosine[top_ids == i]) for i in range(len(cosine_vectors))])

        # Replace any Nan with infinities
        medians[np.isnan(medians)] = np.inf

        # Select the minimum median
        min_median = np.min(medians)
        min_index = np.argmin(medians)

        # import pdb; pdb.set_trace()
        # print(medians)
        # print(min_median, min_index)

        if min_median < threshold:
            return True, min_median, min_index

    return False, 0, -1

class ReID:

    def __init__(
            self, 
            person_model_name: str = 'osnet_x1_0', 
            face_model_name: str = 'Facenet512',
            db=pathlib.Path, 
            device: str = 'cuda', 
            update_knowns_interval: int = 10,
            face_threshold = 0.1,
            person_threshold = 0.1,
            headpose_threshold = 15,
        ):
       
        # Save parameters
        self.person_model_name = person_model_name
        self.face_model_name = face_model_name
        self.face_threshold = face_threshold
        self.person_threshold = person_threshold
        self.headpose_threshold = headpose_threshold
        self.step_id = 0
        self.update_knowns_interval = update_knowns_interval
        self.device = device
       
        # Creating feature extractor
        self.extractor = FeatureExtractor(
            model_name=person_model_name,
            model_path='weights/osnet_x1_0_imagenet.pth',
            device=device
        )

        # Create the face embedding model
        self.face_embed = facenet_pytorch.InceptionResnetV1(pretrained='vggface2').eval()
        self.face_embed.to(device)

        # Create simple database
        self.seen_ids = []
        self.tracklet_id_to_reid_id_map = {}
        self.reid_df = pd.DataFrame({
            'id': [],
            'name': [],
            'face_embeddings': [],
            'person_embeddings': [],
            'last_seen_step_id': []
        })

        # Load the database
        self.reid_df = load_db_representation(db, face_model_name)

    def compute_person_embedding(self, frame: np.ndarray, track: Track) -> Track:
        
        # Obtain their image
        img = crop(track, frame)
        track.embedding = F.normalize(self.extractor(img)).cpu().numpy().squeeze()
        return track
    
    def compute_face_embedding(self, frame: np.ndarray, track: Track) -> Track:
        
        # Obtain their image
        img = crop(track.face, frame)
        tensor = facenet_pytorch_preprocessing(img)
        tensor = tensor.to(self.device)

        # Compute the embedding
        embedding = self.face_embed(tensor).cpu().detach().numpy().squeeze()
        track.face_embedding = embedding

        return track
    
    def compare_person_embedding(self, frame: np.ndarray, track: Track) -> Tuple[bool, float, int]:

        # Compute the person embedding
        track = self.compute_person_embedding(frame, track)

        cosine_vectors = [] 
        for i, row in self.reid_df.iterrows():

            if row['person_embeddings'].shape[0] == 0:
                continue

            # Expand the incoming track embedding to match the multiple embeddings
            cosine_vector = compute_matrix_cosine(row['person_embeddings'],  track.embedding)
            cosine_vectors.append(cosine_vector)

        return selection_procedure(cosine_vectors, self.person_threshold)

    def compare_face_embedding(self, frame: np.ndarray, track: Track) -> Tuple[bool, float, int]:

        # Compute face embedding
        track = self.compute_face_embedding(frame, track)

        cosine_vectors = []
        for i, row in self.reid_df.iterrows():

            if row['face_embeddings'].shape[0] == 0:
                continue

            # Expand the incoming track embedding to match the multiple embeddings
            cosine_vector = compute_matrix_cosine(row['face_embeddings'],  track.face_embedding)
            cosine_vectors.append(cosine_vector)

        return selection_procedure(cosine_vectors, self.face_threshold)

    def handle_unknown_track(self, frame: np.ndarray, track: Track) -> Tuple[bool, Optional[ReIDTrack]]:
        
        # First, compare with the body embeddings
        success, cosine, id = self.compare_person_embedding(frame, track)

        # If we found a match, use it
        if success:
            logger.debug(f"PersonReID: Found a match with cosine: {cosine} and id: {id}")
            name = self.reid_df.loc[id, 'name']
            return True, ReIDTrack(reid=id, name=name, cosine=cosine, track=track)
        
        # If we didn't find a match, compare with the face embeddings (if face is available)
        if track.face and track.face_frontal_distance != None and track.face_frontal_distance < self.headpose_threshold:
            success, cosine, id = self.compare_face_embedding(frame, track)

            # If we found a match, use it
            if success:
                logger.debug(f"FaceReID: Found a match with cosine: {cosine} and id: {id}")

                # Update the personembeddings
                track = self.compute_person_embedding(frame, track)
                self.reid_df.at[id, 'person_embeddings'] = np.vstack([self.reid_df.loc[id, 'person_embeddings'], track.embedding])

                name = self.reid_df.loc[id, 'name']
                return True, ReIDTrack(reid=id, name=name, cosine=cosine, track=track)
        
        return False, None

    def step(self, frame: np.ndarray, tracks: List[Track]) -> List[ReIDTrack]:

        # First determine which Tracks need to be checked
        unknown_tracks: List[Track] = []
        known_tracks: List[Track] = []
        for track in tracks:
            if track.id not in self.seen_ids:
                unknown_tracks.append(track)
            else:
                known_tracks.append(track)

        # Create the container output
        reid_tracks: List[ReIDTrack] = []

        # Handle the known tracks
        if known_tracks:
            # Fetch the prior ReID ID
            for track in known_tracks:
                (reid_id, cosine) = self.tracklet_id_to_reid_id_map[track.id]
                reid_track = ReIDTrack(reid=reid_id, name=self.reid_df.loc[reid_id, 'name'], cosine=cosine, track=track)
                reid_tracks.append(reid_track)

        # Only if we have unknown tracks
        if unknown_tracks:

            # Only if we have people in the database
            if len(self.reid_df): 
                for track in unknown_tracks:

                    # First, compare with known tracks in the DB
                    success, reid_track = self.handle_unknown_track(frame, track)
                    # success, reid_track = False, None

                    # If we found a match, use it
                    if success:
                        reid_tracks.append(reid_track)
                        self.seen_ids.append(track.id)
                        self.tracklet_id_to_reid_id_map[track.id] = (reid_track.reid, reid_track.cosine)

                    break

        return reid_tracks