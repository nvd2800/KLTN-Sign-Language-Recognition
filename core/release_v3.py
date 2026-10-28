"""Physical no-hand release scheduling; marker is distinct from model REST.

Sends one synthetic REST per distinct frame in a no-hand run, at specified cadence.
Never emits a token itself. The Phase09 recognizer requires release_consecutive.
"""
from dataclasses import dataclass

@dataclass
class NoHandReleaseScheduler:
    threshold_frames:int=8
    interval_frames:int=2
    no_hand_frames:int=0

    def __post_init__(self):
        if self.threshold_frames<1 or self.interval_frames<1:
            raise ValueError('Both frame counts must be positive')

    def update(self,hand_present:bool)->bool:
        if hand_present:
            self.no_hand_frames=0
            return False
        self.no_hand_frames+=1
        return (self.no_hand_frames>=self.threshold_frames and
                (self.no_hand_frames-self.threshold_frames)%self.interval_frames==0)

    def reset(self):self.no_hand_frames=0
