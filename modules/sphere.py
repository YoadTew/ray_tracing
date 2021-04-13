import numpy as np
from modules.entity import Entity
from modules.ray import Ray

from utils import normalize, find_intersection

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

    def get_diffuse_color(self, scene, inter_point):
        normal = normalize(inter_point - self.center)
        diff_color = np.zeros(3, dtype=float)

        for light in scene.lights[:]:
            ray = Ray(light.position, inter_point)
            t, near_object = find_intersection(scene, ray)

            curr_color = abs(normal @ ray.direction) * self.material.diffuse_color * light.light_color

            # If this is not the first object we meet
            if abs(t - np.linalg.norm(inter_point - light.position)) > 1e-6:
                curr_color *= (1 - light.shadow_intensity)

            diff_color += curr_color

        return diff_color
