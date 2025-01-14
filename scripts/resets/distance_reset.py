from scripts.resets.reset import Reset
from scripts.rewards.distance import Distance
from typing import Union, Tuple

_distReset = 0

class DistanceReset(Reset):
    def __init__(self, distance: Union[Distance, str], 
                 max_distance: float=None, 
                 max_angle: float=None, 
                 min_distance: float=None, 
                 min_angle: float=None, 
                 reward: float=0, 
                 name: str=None
                ) -> None:
        
        super().__init__(reward)

        # save the name of the distance
        if isinstance(distance, Distance):
            self.distance_name = distance.name
        else:
            self.distance_name = distance

        # save bounds
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.max_angle = max_angle
        self.min_angle = min_angle

        self.name = name if name else "DistReset_" + f"{_distReset+1}"