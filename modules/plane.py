import numpy as np

class Plane:
    def __init__(self, params):
        self.normal = np.array(params[0:3], dtype=float)
        self.offset = float(params[3])
        self.material_index = int(params[4])