import numpy as np

from utils import normalize

class Ray:
    def __init__(self, camera, pixel):
        self.origin = camera
        self.screen_intersect = pixel

        self.direction = normalize(pixel - camera)

    def get_vec(self, t):
        point = self.origin + t * self.direction

        return point