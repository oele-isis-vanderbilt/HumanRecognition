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
from .utils import crop
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

class ReID:

    def __init__(
            self, 
            model_name: str = 'osnet_x1_0', 
            db=pathlib.Path, 
            device: str = 'cpu', 
            update_knowns_interval: int = 10,
            threshold = 0.1
        ):
       
        # Save parameters
        self.threshold = threshold
        self.step_id = 0
        self.update_knowns_interval = update_knowns_interval
       
        # Creating feature extractor
        self.extractor = FeatureExtractor(
            model_name=model_name,
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
        pkl_representations = db / 'representations_facenet512.pkl'
        if pkl_representations.exists():
            with open(pkl_representations, 'rb') as f:
                data = pickle.load(f)

            samples = defaultdict(list)
            for sample in data:
                path = pathlib.Path(sample[0])
                id = path.parent.name
                samples['id'].append(id)
                samples['embedding'].append(sample[1])

            samples = pd.DataFrame(samples)

            # Group by id and add to the reid_df
            for idx, (id, group) in enumerate(samples.groupby('id')):

                embeddings = group['embedding'].values
                embeddings_matrix = np.stack([np.array(e) for e in embeddings])

                self.reid_df = self.reid_df._append({
                    'id': idx,
                    'name': id,
                    'face_embeddings': embeddings_matrix,
                    'person_embeddings': np.empty(shape=(0, 512)),
                    'last_seen_step_id': -1
                }, ignore_index=True)


    def compute_person_embeddings(self, frame: np.ndarray, tracks: List[Track]) -> List[Track]:
        
        # For the tracks not found, find their features
        for track in tracks:
            # Obtain their image
            img = crop(track, frame)
            track.embedding = F.normalize(self.extractor(img)).cpu().numpy().squeeze()

        return tracks
    
    def compute_face_embedding(self, frame: np.ndarray, track: Track) -> Track:
        
        # Obtain their image
        img = crop(track.face, frame)
        embedding_dict = DeepFace.represent(img, model_name='Facenet512', detector_backend="skip", enforce_detection=False)
        embedding = embedding_dict[0]['embedding']
        track.face_embedding = torch.tensor(embedding).cpu().numpy().squeeze()

        return track
    
    def compare_person_embedding(self, frame: np.ndarray, track: Track) -> Tuple[bool, float, int]:
            
        for i, row in self.reid_df.iterrows():

            if row['person_embeddings'].shape[0] == 0:
                continue

            # Expand the incoming track embedding to match the multiple embeddings
            float_vector = np.full(shape=row['person_embeddings'].shape, fill_value=track.embedding)
            import pdb; pdb.set_trace()

        return False, 0, -1

    def compare_face_embedding(self, frame: np.ndarray, track: Track) -> Tuple[bool, float, int]:

        cosine_medians = []
        for i, row in self.reid_df.iterrows():

            if row['face_embeddings'].shape[0] == 0:
                continue

            # Compute face embedding
            track = self.compute_face_embedding(frame, track)

            # Expand the incoming track embedding to match the multiple embeddings
            cosine_vector = compute_matrix_cosine(row['face_embeddings'],  track.face_embedding)
            cosine_medians.append(np.median(cosine_vector))

        # Select the highest cosine similarity
        if cosine_medians:
            max_cosine = max(cosine_medians)
            if max_cosine > self.threshold:
                idx = np.argmax(cosine_medians)
                return True, max_cosine, idx
            
        return False, 0, -1

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