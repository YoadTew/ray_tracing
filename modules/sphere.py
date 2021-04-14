import numpy as np
from modules.entity import Entity
from modules.ray import Ray

from utils import normalize, find_intersection, is_shadowed

class Sphere(Entity):
    def __init__(self, params, materials):
        super().__init__(params, materials)
        self.center = np.array(params[0:3], dtype=float)
        self.radius = float(params[3])

    def intersection(self, ray):
        b = 2 * np.dot(ray.direction, ray.origin - self.center)
        c = np.linalg.norm(ray.origin - self.center) ** 2 - self.radius ** 2
        delta = b ** 2 - 4 * c
        if delta > 0:
            t1 = (-b + np.sqrt(delta)) / 2
            t2 = (-b - np.sqrt(delta)) / 2
            if t1 > 0 and t2 > 0:
                return min(t1, t2)
        return None

    def get_diffuse_specular_color(self, scene, inter_point, camera_ray):
        normal = normalize(inter_point - self.center)
        diff_color = np.zeros(3, dtype=float)
        specular_color = np.zeros(3, dtype=float)

        for light in scene.lights[:]:
            need_shadow, ray = is_shadowed(light, inter_point, scene)

            ####### Diffuse color #######
            curr_diff_color = abs(normal @ ray.direction) * self.material.diffuse_color * light.light_color

            if need_shadow:
                curr_diff_color *= (1 - light.shadow_intensity)

            diff_color += curr_diff_color

            ####### Specular color #######
            R = 2 * (ray.direction @ normal) * normal - ray.direction
            curr_specular_color = self.material.specular_color * \
                                  ((R @ -camera_ray.direction) ** self.material.phong_specularity_coefficient) * \
                                  light.light_color * light.specular_intensity

            if need_shadow:
                curr_specular_color *= (1 - light.shadow_intensity)

            specular_color += curr_specular_color

        return diff_color + specular_color
