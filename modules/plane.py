import numpy as np
from modules.entity import Entity

class Plane(Entity):
    def __init__(self, params, materials):
        super().__init__(params, materials)
        self.normal = np.array(params[0:3], dtype=float)
        self.offset = float(params[3])