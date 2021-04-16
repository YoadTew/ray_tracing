import numpy as np
from modules.entity import Entity
from modules.ray import Ray

from utils import normalize, calc_diffuse_specular_color

class Sphere(Entity):
    def __init__(self, params, materials):
        super().__init__(params, materials)
        self.center = np.array(params[0:3], dtype=float)
        self.radius = float(params[3])
        self.radius_squared = self.radius ** 2

    def intersection(self, ray_origins, ray_directions):
        L = self.center - ray_origins
        t_ca = np.sum(L * ray_directions, axis=-1)

        mask_inter = (t_ca >= 0)

        d_squared = np.full(ray_origins.shape[0], None, dtype=float)
        d_squared[mask_inter] = np.sum(L[mask_inter] * L[mask_inter], axis=-1) - (t_ca[mask_inter] ** 2)

        mask_inter = d_squared < self.radius_squared

        t_hc = np.full(ray_origins.shape[0], None, dtype=float)
        t_hc[mask_inter] = np.sqrt(self.radius_squared - d_squared[mask_inter])

        return np.minimum(t_ca - t_hc, t_ca + t_hc), mask_inter

    def get_normal(self, point):
        normal = normalize(point - self.center)

        return normal