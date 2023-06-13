from typing import List, Tuple, Dict
import logging

import torch
import numpy as np
import torch.nn.functional as F
from torchreid.reid.utils import FeatureExtractor

from .track import Track
from .database import Database

logger = logging.getLogger('')

class ReID:

    def __init__(self, model_name: str = 'osnet_x1_0', device: str = 'cpu', update_knowns_interval: int = 10):
       
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
        self.database = Database()

    def compute_embeddings(self, frame: np.ndarray, tracks: List[Track]) -> List[Track]:
        
        # For the tracks not found, find their features
        for track in tracks:
            # Obtain their image
            img = track.crop(frame)
            track.embedding = F.normalize(self.extractor(img)).cpu().numpy().squeeze()

            # if isinstance(track.head, np.ndarray):
            #     track.face_embedding = np.array([])
        
        return tracks

    def step(self, frame: np.ndarray, tracks: List[Track]) -> Tuple[Dict[int, int], List[Track]]:

        # First determine which Tracks need to be checked
        unknown_tracks: List[Track] = []
        known_tracks: List[Track] = []
        for track in tracks:
            if not self.database.has(track):
                unknown_tracks.append(track)
            else:
                known_tracks.append(track)

        logger.debug(f"STEP ID: {self.step_id}, Known tracks: {[t.id for t in known_tracks]}, Unknown: {[t.id for t in unknown_tracks]}")
        
        # Compute embeddings
        if self.step_id % self.update_knowns_interval == 0:
            known_tracks = self.compute_embeddings(frame, known_tracks)
        
        # Mark the known tracks
        self.database.mark_known(known_tracks, self.step_id)
        
        # Only perform re-identification if new ids:
        if not unknown_tracks:
            
            # Update step id
            self.step_id += 1

            return {}, tracks

        unknown_tracks = self.compute_embeddings(frame, unknown_tracks)

        # Process the incoming features to database to see if matches are possible
        id_map, unknown_tracks = self.database.step(unknown_tracks, self.step_id) 

        # Update step id
        self.step_id += 1

        return id_map, tracks
