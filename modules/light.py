import numpy as np

class Light:
    def __init__(self, params):
        self.position = np.array(params[0:3], dtype=float)
        self.light_color = np.array(params[3:6], dtype=float)
        self.specular_intensity = float(params[6])
        self.shadow_intensity = float(params[7])
        self.light_radius = int(params[8])
