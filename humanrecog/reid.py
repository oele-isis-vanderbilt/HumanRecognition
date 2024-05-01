from typing import List, Tuple, Dict, Optional
import logging
import pathlib
import pickle
from collections import defaultdict

import torch
import numpy as np
import torch.nn.functional as F
from torchreid.reid.utils import FeatureExtractor
from deepface import DeepFace
import pandas as pd

from .data_protocols import Track, ReIDTrack, Detection
from .utils import crop, load_db_representation
# from .database import Database

logger = logging.getLogger('')

def compute_matrix_cosine(m: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Cosine similarity between a matrix and a vector.

    Args:
        embeddings (np.ndarray): NxM matrix
        float_vector (np.ndarray): M vector

    Returns:
        np.ndarray: N vector of cosine similarities
    """
    
    # Normalize the rows of the matrix A
    norm_m = np.linalg.norm(m, axis=1, keepdims=True)
    m_normalized = m / norm_m

    # Normalize the vector v
    norm_v = np.linalg.norm(m)
    v_normalized = v / norm_v

    # Compute the cosine similarity
    cosine_similarity = np.dot(m_normalized, v_normalized)
    return cosine_similarity

def selection_procedure(cosine_vectors: List[np.ndarray], threshold: float) -> Tuple[bool, float, int]:

    if len(cosine_vectors) > 0:
        medians = np.array([np.median(cosine_vector) for cosine_vector in cosine_vectors])
        max_medians = np.max(medians)
        print(max_medians)
        max_index = np.argmax(medians)
        if max_medians > threshold:
            return True, max_medians, max_index

    # for cosine_vector in cosine_vectors:
    #     if np.max(cosine_vector) > threshold:
    #         return True, np.max(cosine_vector), np.argmax(cosine_vector)
        
    return False, 0, -1

class ReID:

    def __init__(
            self, 
            person_model_name: str = 'osnet_x1_0', 
            face_model_name: str = 'Facenet512',
            db=pathlib.Path, 
            device: str = 'cpu', 
            update_knowns_interval: int = 10,
            face_threshold = 0.3,
            person_threshold = 0.1
        ):
       
        # Save parameters
        self.person_model_name = person_model_name
        self.face_model_name = face_model_name
        self.face_threshold = face_threshold
        self.person_threshold = person_threshold
        self.step_id = 0
        self.update_knowns_interval = update_knowns_interval
       
        # Creating feature extractor
        self.extractor = FeatureExtractor(
            model_name=person_model_name,
            model_path='weights/osnet_x1_0_imagenet.pth',
            device=device
        )

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
        embedding_dict = DeepFace.represent(img, model_name=self.face_model_name, detector_backend="skip", enforce_detection=False, normalization='Facenet')
        embedding = embedding_dict[0]['embedding']
        track.face_embedding = torch.tensor(embedding).cpu().numpy().squeeze()

        return track
    
    def compare_person_embedding(self, frame: np.ndarray, track: Track) -> Tuple[bool, float, int]:

        cosine_vectors = [] 
        for i, row in self.reid_df.iterrows():

            if row['person_embeddings'].shape[0] == 0:
                continue

            # Compute the person embedding
            track = self.compute_person_embedding(frame, track)

            # Expand the incoming track embedding to match the multiple embeddings
            cosine_vector = compute_matrix_cosine(row['person_embeddings'],  track.embedding)
            cosine_vectors.append(cosine_vector)

        return selection_procedure(cosine_vectors, self.person_threshold)

    def compare_face_embedding(self, frame: np.ndarray, track: Track) -> Tuple[bool, float, int]:

        cosine_vectors = []
        for i, row in self.reid_df.iterrows():

            if row['face_embeddings'].shape[0] == 0:
                continue

            # Compute face embedding
            track = self.compute_face_embedding(frame, track)

            # Expand the incoming track embedding to match the multiple embeddings
            cosine_vector = compute_matrix_cosine(row['face_embeddings'],  track.face_embedding)
            cosine_vectors.append(cosine_vector)

        return selection_procedure(cosine_vectors, self.face_threshold)

    def handle_unknown_track(self, frame: np.ndarray, track: Track) -> Tuple[bool, Optional[ReIDTrack]]:
        
        # First, compare with the body embeddings
        success, cosine, id = self.compare_person_embedding(frame, track)

        # If we found a match, use it
        if success:
            name = self.reid_df.loc[id, 'name']
            return True, ReIDTrack(reid=id, name=name, cosine=cosine, track=track)
        
        # If we didn't find a match, compare with the face embeddings (if face is available)
        if track.face:
            success, cosine, id = self.compare_face_embedding(frame, track)

            # If we found a match, use it
            if success:

                # Update the person embeddings
                # track = self.compute_person_embedding(frame, track)
                # self.reid_df.at[id, 'person_embeddings'] = np.vstack([self.reid_df.loc[id, 'person_embeddings'], track.embedding])

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

        return reid_tracks