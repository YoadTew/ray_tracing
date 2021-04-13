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