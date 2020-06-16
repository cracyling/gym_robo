from enum import Enum, auto


class HyQState(Enum):
    Reached = auto()
    InProgress = auto()
    ApproachJointLimits = auto()
    Fallen = auto()
    Timeout = auto()
    Undefined = auto()
