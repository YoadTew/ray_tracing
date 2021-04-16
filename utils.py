import numpy as np

def normalize(vector):
    return vector / np.linalg.norm(vector)

def find_intersection(scene, ray, stop_at_first=False):
    min_t = np.inf
    nearest_object = None

    for entity in scene.planes + scene.spheres:
        t = entity.intersection(ray)

        if t and t < min_t:
            min_t = t
            nearest_object = entity

            if stop_at_first:
                break

    return min_t, nearest_object

def is_soft_shadowed(light, inter_point, scene, normal):
    from modules.ray import Ray
    ray = Ray(light.position, inter_point)

    Vz = ray.direction
    Vx = normalize(np.cross(scene.camera.up_vector, Vz))
    Vy = normalize(np.cross(Vx, Vz))

    p_center = light.position
    p_radius = light.light_radius

    P_0 = p_center - (p_radius / 2) * Vx - (p_radius / 2) * Vy
    move_x = (Vx * p_radius) / scene.settings.soft_shadow_N
    move_y = (Vy * p_radius) / scene.settings.soft_shadow_N

    rays_hit = 0

    for i in range(scene.settings.soft_shadow_N):
        point = np.copy(P_0)
        for j in range(scene.settings.soft_shadow_N):
            if not is_shadowed(point, inter_point, scene, normal):
                rays_hit += 1

            point += move_x
        P_0 += move_y

    percent_hit = rays_hit / scene.settings.soft_shadow_N**2

    return percent_hit, ray

def is_shadowed(light_position, inter_point, scene, normal):
    from modules.ray import Ray

    ray = Ray(inter_point + (normal * 1e-3), light_position)
    t, near_object = find_intersection(scene, ray)
    need_shadow = (near_object is not None) and t < np.linalg.norm(light_position - inter_point)

    # ray = Ray(light_position, inter_point)
    # t, near_object = find_intersection(scene, ray, stop_at_first=False)
    # need_shadow = abs(t - np.linalg.norm(inter_point - light_position)) > 1e-6

    return need_shadow