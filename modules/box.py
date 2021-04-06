import numpy as np
from modules.entity import Entity

class Box(Entity):
    def __init__(self, params, materials):
        super().__init__(params, materials)
        self.center = np.array(params[0:3], dtype=float)
        self.scale = float(params[3])