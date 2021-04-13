import numpy as np

from utils import normalize

class Ray:
    def __init__(self, origin, intersect_point):
        self.origin = origin
        self.intersect_point = intersect_point

        self.direction = normalize(intersect_point - origin)

    def get_point(self, t):
        point = self.origin + t * self.direction

        return point