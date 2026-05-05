from dataclasses import dataclass, asdict
from enum import Enum

class Pose(str, Enum):
    FLAT  = "FLAT"
    PINCH = "PINCH"
    POINT = "POINT"
    IDLE  = "IDLE"
    AWAY  = "AWAY"

@dataclass
class Hand:
    x: float  # normalised pointer finger x [0, 1]
    y: float  # normalised pointer finger y [0, 1]
    vx: float  # px/s
    vy: float  # px/s
    theta: float
    pose: Pose

    def as_dict(self) -> dict:
        d = asdict(self)
        d["pose"] = self.pose.value
        return d


AWAY_HAND = Hand(x=0.0, y=0.0, vx=0.0, vy=0.0, theta=0.0, pose=Pose.AWAY)