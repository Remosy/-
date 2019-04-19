from Demo_gym.spaces.space import Space
from Demo_gym.spaces.box import Box
from Demo_gym.spaces.discrete import Discrete
from Demo_gym.spaces.multi_discrete import MultiDiscrete
from Demo_gym.spaces.multi_binary import MultiBinary
from Demo_gym.spaces.tuple import Tuple
from Demo_gym.spaces.dict import Dict

from Demo_gym.spaces.utils import flatdim
from Demo_gym.spaces.utils import flatten
from Demo_gym.spaces.utils import unflatten

__all__ = ["Space", "Box", "Discrete", "MultiDiscrete", "MultiBinary", "Tuple", "Dict", "flatdim", "flatten", "unflatten"]
