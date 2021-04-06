import numpy as np


class Material:
    def __init__(self, params, index=0):
        self.index = index

        self.diffuse_color = np.array(params[0:3], dtype=float)
        self.specular_color = np.array(params[3:6], dtype=float)
        self.reflection_color = np.array(params[6:9], dtype=float)
        self.phong_specularity_coefficient = float(params[9])
        self.transparency = float(params[10])
