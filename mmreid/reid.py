from typing import List, Tuple, Dict
import logging

import torch
import numpy as np
from torchreid.reid.utils import FeatureExtractor

from .track import Track
from .database import Database

logger = logging.getLogger('')

class ReID:

    def __init__(self, model_name: str = 'osnet_x1_0', device: str = 'cpu'):
       
        # Creating feature extractor
        self.extractor = FeatureExtractor(
            model_name='osnet_x1_0',
            model_path='weights/osnet_x1_0_imagenet.pth',
            device=device
        )

        # Create simple database
        self.database = Database()

    def step(self, frame: np.ndarray, tracks: List[Track]) -> Tuple[Dict[int, int], List[Track]]:

        # First determine which Tracks need to be checked
        unknown_tracks: List[Track] = []
        for track in tracks:
            if not self.database.has(track):
                unknown_tracks.append(track)
        
        # Only perform re-identification if new ids:
        if not unknown_tracks:
            return {}, tracks

        # Mark which tracks are being used
        self.database.unmark()
        self.database.mark(tracks)
        
        # For the tracks not found, find their features
        embeddings = []
        old_ids = []
        for track in unknown_tracks:
            # Obtain their image
            img = track.crop(frame)
            embeddings.append(self.extractor(img))
            old_ids.append(track.id)

        # Concatenate same-size tensors to make comparison fast
        embeddings = torch.cat(embeddings)

        # Process the incoming features to database to see if matches 
        # are possible
        new_ids = self.database.step(old_ids, embeddings)

        # Create a mapping of ids
        id_map = {o:n for o,n in zip(old_ids, new_ids)}

        # Overwrite ids
        for new_id, track in zip(new_ids, unknown_tracks):
            track.id = new_id

        return id_map, tracks
