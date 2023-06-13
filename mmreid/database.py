from typing import Optional, List, Tuple, Dict
import logging

import torch
import numpy as np
import pandas as pd

from .track import Track

logger = logging.getLogger('')

class Database:

    def __init__(self, threshold: float = 0.8, entry_ttl: Optional[int] = 30):

        # Input parameters
        self.threshold = threshold
        self.entry_ttl = entry_ttl

        # Container
        self.data = pd.DataFrame({
            'ttl': [], # int
            'tlhw': [], # np.ndarray
            'body_embedding': [], # np.ndarray
            'face_embedding': [], # Optional[np.ndarray]
            'used': [] # boolean
        })

    def has(self, track: Track):
        return track.id in self.data.index

    def mark_known(self, tracks: List[Track]):

        # Mark all as false
        self.data['used'] = False

        logger.debug(self.data)

        # Update all 'used'
        for track in tracks:
            logger.debug("Before: ", track.id)
            self.data.iloc[track.id].used = True
            logger.debug("After: ", track.id)

    def compare(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Cosine similarity comparison between arrays.

        Args:
            a (np.ndarray): Nx512
            b (np.ndarray): Mx512

        Returns:
            np.ndarray: NxM

        """
        # Transform and save
        b_t = b.T
        
        product = np.matmul(a, b_t)

        norm_a = np.linalg.norm(a,axis=1)
        norm_a = norm_a.reshape(norm_a.size,1)

        norm_b_t = np.linalg.norm(b_t,axis=0)
        norm_b_t = norm_b_t.reshape(1,norm_b_t.size)
        
        product_norms = np.matmul(norm_a,norm_b_t)
        
        similarity = np.subtract(1,np.divide(product,product_norms))
        
        return similarity

    def add(self, track: Track):
        self.data.loc[track.id] = [self.entry_ttl, track.bbox, track.embedding, track.face_embedding, 1]

    def update(self):
        
        # Select the ids not used
        unused_ids = self.data[self.data.used == False]

        # Decrease their TTL
        # unused_ids.ttl -= 1

        # Remove entries that go below 0
        # import pdb; pdb.set_trace()

    def step(self, tracks: List[Track]) -> Tuple[Dict, List[Track]]:

        # Container mapping old to new IDS if any change happen
        id_map: Dict[int, int] = {}

        # First determine if there are any unused ids to compare
        unused_ids = self.data[self.data.used == False]

        # Handling if no unused ids
        if len(unused_ids) == 0:
            for track in tracks:
                self.add(track)
            
            self.update()
            return id_map, tracks

        a = np.stack([x.embedding for x in tracks])
        b = np.stack(unused_ids.body_embedding.to_list())

        # Compute cosine similarity
        cosine = self.compare(a, b)

        # Determine matches
        argmax = np.argmax(cosine, axis=1)
        max_values = cosine[:,argmax.squeeze()]
        is_match = (max_values > self.threshold).reshape((-1,))
        logger.debug(max_values)
        logger.debug(is_match)

        for i, m in enumerate(is_match):
            if m:
                id_map[tracks[i].id] = argmax[i]
                tracks[i].id = argmax[i]
            else:
                self.add(tracks[i])

        self.update()
        return id_map, tracks
