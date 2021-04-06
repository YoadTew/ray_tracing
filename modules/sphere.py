import numpy as np

class Sphere:
    def __init__(self, params):
        self.center = np.array(params[0:3], dtype=float)
        self.radius = float(params[3])
        self.material_index = int(params[4])