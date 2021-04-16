import numpy as np
from modules.entity import Entity
from modules.ray import Ray

from utils import normalize, is_soft_shadowed

class Sphere(Entity):
    def __init__(self, params, materials):
        super().__init__(params, materials)
        self.center = np.array(params[0:3], dtype=float)
        self.radius = float(params[3])
        self.radius_squared = self.radius ** 2

    def intersection_algebric(self, ray):
        b = 2 * np.dot(ray.direction, ray.origin - self.center)
        c = np.linalg.norm(ray.origin - self.center) ** 2 - self.radius ** 2
        delta = b ** 2 - 4 * c
        if delta > 0:
            t1 = (-b + np.sqrt(delta)) / 2
            t2 = (-b - np.sqrt(delta)) / 2
            if t1 > 0 and t2 > 0:
                return min(t1, t2)
        return None

    def intersection(self, ray):
        L = self.center - ray.origin
        t_ca = np.dot(L, ray.direction)

        if t_ca < 0:
            return None

        d_squared = np.dot(L, L) - (t_ca ** 2)

        if d_squared > self.radius_squared:
            return None

        t_hc = np.sqrt(self.radius_squared - d_squared)

        return min(t_ca - t_hc, t_ca + t_hc)

    def get_diffuse_specular_color(self, scene, inter_point, camera_ray):
        normal = normalize(inter_point - self.center)
        diff_color = np.zeros(3, dtype=float)
        specular_color = np.zeros(3, dtype=float)

        for light in scene.lights[:]:
            light_ray_hits, ray = is_soft_shadowed(light, inter_point, scene, normal)

            ####### Diffuse color #######
            curr_diff_color = abs(normal @ ray.direction) * self.material.diffuse_color * light.light_color

            # if need_shadow:
            curr_diff_color *= ((1 - light.shadow_intensity) + light.shadow_intensity * light_ray_hits)

            diff_color += curr_diff_color

            ####### Specular color #######
            R = 2 * (ray.direction @ normal) * normal - ray.direction
            curr_specular_color = self.material.specular_color * \
                                  ((R @ -camera_ray.direction) ** self.material.phong_specularity_coefficient) * \
                                  light.light_color * light.specular_intensity

            # if need_shadow:
            curr_specular_color *= ((1 - light.shadow_intensity) + light.shadow_intensity * light_ray_hits)

            specular_color += curr_specular_color

        return diff_color + specular_color
