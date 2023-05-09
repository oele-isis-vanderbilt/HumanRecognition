from typing import Optional, List
import logging

from dataclasses import dataclass
import torch
import numpy as np

from .track import Track

logger = logging.getLogger('')

@dataclass
class Entry:
    ttl: Optional[int]
    id: int
    embedding: torch.Tensor
    used: bool

class Database:

    def __init__(self, threshold: float = 0.8, entry_ttl: Optional[int] = 30):

        # Input parameters
        self.threshold = threshold
        self.entry_ttl = entry_ttl

        # Container
        self.data = {}

    def has(self, track: Track):
        return track.id in self.data

    def unmark(self):
        for entry in self.data.values():
            entry.used = False

    def mark(self, tracks: List[Track]):
        for track in tracks:
            if track.id in self.data:
                self.data[track.id].used = True

    def compare(self, a: torch.Tensor, b: torch.Tensor):

        # Compute the L2 norm of each tensor
        norm_a = torch.norm(a, p=2, dim=1, keepdim=True)
        norm_b = torch.norm(b, p=2, dim=1, keepdim=True)

        # Normalize the tensors to have unit L2 norm
        a_normed = a / norm_a
        b_normed = b / norm_b

        # Compute the cosine similarity between the two normalized tensors
        similarity = torch.mm(a_normed, b_normed.t())

        return similarity.item()

    def step(self, old_ids: List[int], embeddings: torch.Tensor) -> List[int]:
        """Processing incoming features and determine if matches.

        Determine if there is a high enough similarity to match previous
        tracks to incoming tracks. Need to assign them adds depending

        Args:
            features (torch.Tensor): features

        Returns:
            List[int]: New list of IDs
        """

        to_be_added_entries: List[Entry] = []
        new_ids = []
        for old_id, embedding in zip(old_ids, embeddings):

            results = {}
            ids = []
            for id, entry in self.data.items():
                if not entry.used:
                    results[id] = self.compare(torch.unsqueeze(embedding, dim=0), torch.unsqueeze(entry.embedding, dim=0))
                    ids.append(id)

            if not results:
                to_be_added_entries.append(Entry(
                    ttl=self.entry_ttl, 
                    id=old_id, 
                    embedding=embedding,
                    used=True
                ))
                new_ids.append(old_id)

            else:
                # logger.debug(f"{ids} - {results}")
                scores = np.array(results.values())
                best_score_idx = np.argmax(scores)
                new_ids.append(ids[best_score_idx])
                logger.debug(f"{ids} - {new_ids}")

        # After processing, add entries
        for entry in to_be_added_entries:
            self.data[entry.id] = entry

        return new_ids
