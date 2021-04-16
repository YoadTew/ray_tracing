import numpy as np
from modules.entity import Entity
from modules.ray import Ray
from utils import normalize, find_intersection, is_soft_shadowed

class Plane(Entity):
    def __init__(self, params, materials):
        super().__init__(params, materials)
        self.normal = np.array(params[0:3], dtype=float)
        self.offset = float(params[3])

    def intersection(self, ray):
        t = -(ray.origin @ self.normal - self.offset) / (ray.direction @ self.normal + 1e-8)
        if t > 0:
            return t
        return None

    def get_diffuse_specular_color(self, scene, inter_point, camera_ray):
        diff_color = np.zeros(3, dtype=float)
        specular_color = np.zeros(3, dtype=float)

        for light in scene.lights[:]:
            light_ray_hits, ray = is_soft_shadowed(light, inter_point, scene, self.normal)

            ####### Diffuse color #######
            curr_diff_color = abs(self.normal @ ray.direction) * self.material.diffuse_color * light.light_color

            # if need_shadow:
            curr_diff_color *= ((1 - light.shadow_intensity) + light.shadow_intensity * light_ray_hits)

            diff_color += curr_diff_color

            ####### Specular color #######
            R = 2 * (ray.direction @ self.normal) * self.normal - ray.direction
            curr_specular_color = self.material.specular_color * \
                                  ((R @ -camera_ray.direction) ** self.material.phong_specularity_coefficient) * \
                                  light.light_color * light.specular_intensity

            # if need_shadow:
            curr_specular_color *= ((1 - light.shadow_intensity) + light.shadow_intensity * light_ray_hits)

            specular_color += curr_specular_color

        return diff_color + specular_color