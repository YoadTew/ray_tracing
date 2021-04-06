import numpy as np


class Settings:
    def __init__(self, params):
        self.background_color = np.array(params[0:3], dtype=int)
        self.soft_shadow_N = int(params[3])
        self.max_recursion = int(params[4])
