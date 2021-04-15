import numpy as np
from modules.entity import Entity
from modules.ray import Ray
from utils import normalize, find_intersection, is_soft_shadowed

class Plane(Entity):
    def __init__(self, params, materials):
        super().__init__(params, materials)
        self.normal = np.array(params[0:3], dtype=float)
        self.offset = float(params[3])