import numpy as np
from modules.entity import Entity

class Sphere(Entity):
    def __init__(self, params, materials):
        super().__init__(params, materials)
        self.center = np.array(params[0:3], dtype=float)
        self.radius = float(params[3])