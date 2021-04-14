import numpy as np

def normalize(vector):
    return vector / np.linalg.norm(vector)

def find_intersection(scene, ray):
    min_t = np.inf
    nearest_object = None

    for entity in scene.spheres + scene.planes[:]:
        t = entity.intersection(ray)

        if t and t < min_t:
            min_t = t
            nearest_object = entity

    return min_t, nearest_object

def is_shadowed(light, inter_point, scene):
    from modules.ray import Ray

    # ray = Ray(inter_point + (self.normal * 1e-4), light.position)
    # t, near_object = find_intersection(scene, ray)
    # need_shadow = near_object is not None

    ray = Ray(light.position, inter_point)
    t, near_object = find_intersection(scene, ray)
    need_shadow = abs(t - np.linalg.norm(inter_point - light.position)) > 1e-6

    return need_shadow, ray