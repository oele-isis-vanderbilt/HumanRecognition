from typing import Optional, List, Tuple, Dict
import logging

import torch
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from .track import Track

logger = logging.getLogger('')

class Database:

    def __init__(
            self, 
            threshold: float = 0.55, 
            entry_ttl: Optional[int] = 30
        ):

        # Input parameters
        self.threshold = threshold
        self.entry_ttl = entry_ttl
        self.step_id = 0

        # Container
        self.data = pd.DataFrame({
            'ttl': [], # int
            'tlhw': [], # np.ndarray
            'body_embedding': [], # np.ndarray
            'face_embedding': [], # Optional[np.ndarray]
            'used': [], # boolean
            "last_seen_step_id": [] # int
        })

    def has(self, track: Track):
        return track.id in self.data.index

    def mark_known(self, tracks: List[Track], step_id: int):

        # Mark all as false
        self.data['used'] = False

        # Update all 'used'
        for track in tracks:
            self.data.at[track.id, 'used'] = True
            self.data.at[track.id, 'last_seen_step_id'] = step_id

            # If they have embeddings, update those too
            if isinstance(track.embedding, np.ndarray):
                self.data.at[track.id, 'body_embedding'] = track.embedding

    def compare(self, tracks: List[Track], unused_ids: pd.DataFrame, id_map: Dict[int, int]) -> Dict[int, int]:

        # if len(tracks) >= 2:
        #     import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()
        
        # First compute the body features
        a = np.stack([x.embedding for x in tracks])
        b = np.stack(unused_ids.body_embedding.to_list())
        cosine = self.cosine(a, b)
        
        # Compute a distance
        a_bbox = np.stack([x.bbox[:2] for x in tracks])
        b_bbox = np.stack(unused_ids.tlhw.to_list())[:,:2]
        distance_matrix = cdist(a_bbox, b_bbox, metric='euclidean')
        delta_time = np.expand_dims(self.step_id - np.stack(unused_ids.last_seen_step_id.to_list()), axis=0)
        proxity_score = np.exp(-distance_matrix/200) * np.exp(-delta_time/1000)

        combined_scores = np.stack([cosine, proxity_score])
        scores = np.mean(combined_scores, axis=0)

        # import pdb; pdb.set_trace()

        # Determine matches
        argmax = np.argmax(scores, axis=1)
        max_values = scores[np.arange(scores.shape[0]),argmax.squeeze()]
        is_match = (max_values > self.threshold).reshape((-1,))
        logger.debug(max_values)
        logger.debug(is_match)

        for i, m in enumerate(is_match):
            if m:
                id_map[tracks[i].id] = argmax[i]
                tracks[i].id = argmax[i]
            else:
                self.add(tracks[i])
        
        return id_map

    def bbox_distance(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        ...

    def cosine(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
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
        self.data.loc[track.id] = [self.entry_ttl, track.bbox, track.embedding, track.face_embedding, 1, self.step_id]

    def update(self):
        
        # Select the ids not used
        unused_ids = self.data[self.data.used == False]

        # Decrease their TTL
        # unused_ids.ttl -= 1

        # Remove entries that go below 0
        # import pdb; pdb.set_trace()

    def step(self, tracks: List[Track], step_id: int) -> Tuple[Dict, List[Track]]:

        # Update the step id
        self.step_id = step_id

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

        logger.debug(self.data)

        # Compare the tracks to the database
        id_map = self.compare(tracks, unused_ids, id_map)
        
        self.update()
        return id_map, tracks
