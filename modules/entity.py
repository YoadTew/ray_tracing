import numpy as np

class Entity:
    def __init__(self, params, materials):
        self.material_index = int(params[4])
        self.material = materials[self.material_index - 1]

    def intersection(self, ray):
        return None

    def get_diffuse_specular_color(self, scene, inter_point, camera_ray):
        return None
