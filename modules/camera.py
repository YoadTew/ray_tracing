import numpy as np


class Camera:
    def __init__(self, params):
        self.position = np.array(params[0:3], dtype=float)
        self.look_at = np.array(params[3:6], dtype=float)
        self.up_vector = np.array(params[6:9], dtype=float)
        self.screen_distance = float(params[9])
        self.screen_width = float(params[10])

        self.fisheye = False
        if len(params) >= 12:
            self.fisheye = params[11] != 'false'

        self.fisheye_k = 0.5
        if len(params) >= 13:
            self.fisheye_k = float(params[12])
