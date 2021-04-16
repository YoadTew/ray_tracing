import numpy as np
from modules.entity import Entity
from modules.ray import Ray
from utils import normalize, calc_diffuse_specular_color

class Plane(Entity):
    def __init__(self, params, materials):
        super().__init__(params, materials)
        self.normal = np.array(params[0:3], dtype=float)
        self.offset = float(params[3])

    def intersection(self, ray_origins, ray_directions):
        a = (ray_origins @ self.normal - self.offset)
        b = (ray_directions @ self.normal + 1e-8)
        t = -a / b

        mask_inter = t > 0

        return t, mask_inter

    def get_diffuse_specular_color(self, scene, inter_point, camera_ray_direction):
        color = calc_diffuse_specular_color(scene, inter_point, camera_ray_direction, self.normal,
                                            self.material.diffuse_color, self.material.specular_color,
                                            self.material.phong_specularity_coefficient)

        return color