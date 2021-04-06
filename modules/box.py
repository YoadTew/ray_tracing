import numpy as np

class Box:
    def __init__(self, params):
        self.center = np.array(params[0:3], dtype=float)
        self.scale = float(params[3])
        self.material_index = int(params[4])