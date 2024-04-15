from typing import List, Tuple, Dict
import logging
import pathlib
import pickle
from collections import defaultdict

import torch
import numpy as np
import torch.nn.functional as F
from torchreid.reid.utils import FeatureExtractor
import deepface
import pandas as pd

from .data_protocols import Track, ReIDTrack, Detection
from .utils import crop
# from .database import Database

logger = logging.getLogger('')

def compute_cosine(embeddings, float_vector):
    # Compute consine similarity
    dot_product = np.dot(embeddings, float_vector)
    norm_series = np.linalg.norm(embeddings)
    norm_float_vector = np.linalg.norm(float_vector)
    cosine_similarity = dot_product / (norm_series * norm_float_vector)
    return cosine_similarity

class ReID:

    def __init__(self, model_name: str = 'osnet_x1_0', db=pathlib.Path, device: str = 'cpu', update_knowns_interval: int = 10):
       
        # Save parameters
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
                self.reid_df = self.reid_df._append({
                    'id': idx,
                    'name': id,
                    'face_embeddings': group['embedding'].values,
                    'person_embeddings': [],
                    'last_seen_step_id': -1
                }, ignore_index=True)


    def compute_person_embeddings(self, frame: np.ndarray, tracks: List[Track]) -> List[Track]:
        
        # For the tracks not found, find their features
        for track in tracks:
            # Obtain their image
            img = crop(track, frame)
            track.embedding = F.normalize(self.extractor(img)).cpu().numpy().squeeze()

        return tracks
    
    def compute_face_embeddings(self, frame: np.ndarray, tracks: List[Track]) -> List[Track]:
        
        # For the tracks not found, find their features
        for track in tracks:
            # Obtain their image
            img = crop(track.face, frame)
            embedding = deepface.DeepFace.represent(img, model_name='Facenet', detector_backend="skip", enforce_detection=False)
            track.face_embedding = F.normalize(torch.tensor(embedding)).cpu().numpy().squeeze()

        return tracks
    
    def compare_person_embedding(self, track: Track):
        
        embeddings = self.reid_df['person_embeddings'].values
        float_vector = np.full(shape=embeddings.shape, fill_value=track.embedding)
        cosine_similarity = compute_cosine(embeddings, float_vector)

    def compare_face_embedding(self, track: Track):
        ...

        # embeddings = self.reid_df['face_embeddings'].values
        # float_vector = np.full(shape=embeddings.shape, fill_value=track.face_embedding)
        # cosine_similarity = compute_cosine(embeddings, float_vector)

    def step(self, frame: np.ndarray, tracks: List[Track]) -> List[ReIDTrack]:
        ...

        # First determine which Tracks need to be checked
        unknown_tracks: List[Track] = []
        known_tracks: List[Track] = []
        for track in tracks:
            if track.id not in self.seen_ids:
                unknown_tracks.append(track)
            else:
                known_tracks.append(track)

        # Only if we have unknown tracks
        if unknown_tracks:

            # Only if we have people in the database
            if len(self.reid_df): 
                unknown_tracks = self.compute_person_embeddings(frame, unknown_tracks)
                for utrack in unknown_tracks:

                    # First compare with known tracks in the DB
                    self.compare_person_embedding(utrack)

            # Then check only with the faces
            unknown_tracks = self.compute_face_embeddings(frame, unknown_tracks)
            for utrack in unknown_tracks:
                
                # First, compare with known tracks in the DB
                self.compare_face_embedding(utrack)
        

        # # Process the incoming features to database to see if matches are possible
        # id_map, unknown_tracks = self.database.step(unknown_tracks, self.step_id) 

        # # Update step id
        # self.step_id += 1

        # return id_map, tracks
