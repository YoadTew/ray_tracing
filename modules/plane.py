import numpy as np
from modules.entity import Entity
from modules.ray import Ray

class Plane(Entity):
    def __init__(self, params, materials):
        super().__init__(params, materials)
        self.normal = np.array(params[0:3], dtype=float)
        self.offset = float(params[3])

    def intersection(self, ray_origins, ray_directions):
        a = (ray_origins @ self.normal - self.offset)
        b = (ray_directions @ self.normal + 1e-8)
        t = -a / b

        mask_inter = t > 0

        return t, mask_inter

    def get_normal(self, point):
        return self.normal