import numpy as np
import time

def normalize(vector):
    return vector / np.linalg.norm(vector)

def find_intersection(scene, ray_origins, ray_directions):
    min_t = np.full(ray_origins.shape[0], np.inf, dtype=float)
    nearest_objects = np.full(ray_origins.shape[0], -1, dtype=int)

    for idx, entity in enumerate(scene.objects):
        t, mask_inter = entity.intersection(ray_origins, ray_directions)

        new_t = np.copy(min_t)
        new_t[mask_inter] = np.minimum(new_t[mask_inter], t[mask_inter])
        changes = (new_t != min_t)

        nearest_objects[changes] = idx
        min_t = new_t

    return min_t, nearest_objects

def find_color(scene, t, nearest_objects, camera_ray_origins, camera_ray_directions, img_height, img_width):
    inter_points = camera_ray_origins + np.expand_dims(t, 1) * camera_ray_directions
    colors = np.zeros((img_height * img_width, 3), dtype=np.uint8)

    start = time.time()
    for i in range(colors.shape[0]):
        if i % (colors.shape[0] * 0.01) == 0:
            print(f'Percentage of pixels: {i/colors.shape[0]}, Second passed: {time.time() - start}')
        object_idx = nearest_objects[i]

        if object_idx >= 0:
            p_object = scene.objects[object_idx]
            color = p_object.get_diffuse_specular_color(scene, inter_points[i], camera_ray_directions[i])
            color = np.clip(color, 0, 1) * 255
            colors[i] = color
        else:
            colors[i] = 0

    return colors.reshape(img_height, img_width, 3)

def calc_diffuse_specular_color(scene, inter_point, camera_ray_direction, normal, p_diff_color, p_spec_color, p_phong_coeff):
    diff_color = np.zeros(3, dtype=float)
    specular_color = np.zeros(3, dtype=float)

    for light in scene.lights[:]:
        light_ray_hits, ray_direction = is_soft_shadowed(light, inter_point, scene, normal) #1., normalize(inter_point - light.position)

        ####### Diffuse color #######
        curr_diff_color = abs(normal @ ray_direction) * p_diff_color * light.light_color

        # if need_shadow:
        curr_diff_color *= ((1 - light.shadow_intensity) + light.shadow_intensity * light_ray_hits)

        diff_color += curr_diff_color

        ####### Specular color #######
        R = 2 * (ray_direction @ normal) * normal - ray_direction
        curr_specular_color = p_spec_color * \
                              ((R @ - camera_ray_direction) ** p_phong_coeff) * \
                              light.light_color * light.specular_intensity

        # if need_shadow:
        curr_specular_color *= ((1 - light.shadow_intensity) + light.shadow_intensity * light_ray_hits)

        specular_color += curr_specular_color

    return diff_color + specular_color

def is_soft_shadowed(light, inter_point, scene, normal):
    ray_direction = normalize(inter_point - light.position)

    Vz = ray_direction
    Vx = normalize(np.cross(scene.camera.up_vector, Vz))
    Vy = normalize(np.cross(Vx, Vz))

    p_center = light.position
    p_radius = light.light_radius

    P_0 = p_center - (p_radius / 2) * Vx - (p_radius / 2) * Vy
    move_x = (Vx * p_radius) / scene.settings.soft_shadow_N
    move_y = (Vy * p_radius) / scene.settings.soft_shadow_N

    N_squared = scene.settings.soft_shadow_N * scene.settings.soft_shadow_N

    ray_origins = np.zeros((N_squared, 3), dtype=float)
    ray_directions = np.zeros((N_squared, 3), dtype=float)

    for i in range(scene.settings.soft_shadow_N):
        point = np.copy(P_0)
        for j in range(scene.settings.soft_shadow_N):
            ray_origins[i * scene.settings.soft_shadow_N + j] = inter_point + (normal * 1e-3)
            ray_directions[i * scene.settings.soft_shadow_N + j] = normalize(point - inter_point + (normal * 1e-3))

            point += move_x
        P_0 += move_y

    t, nearest_objects = find_intersection(scene, ray_origins, ray_directions)

    ray_hits = N_squared - np.sum(np.logical_and(t < np.linalg.norm(light.position - inter_point), (nearest_objects >= 0)))

    percent_hit = ray_hits / N_squared

    return percent_hit, ray_direction

# def is_shadowed(light_position, inter_point, scene, normal):
#     from modules.ray import Ray
#
#     find_intersection(scene, ray_origins, ray_directions)
#
#     ray = Ray(inter_point + (normal * 1e-3), light_position)
#     t, near_object = find_intersection(scene, ray)
#     need_shadow = (near_object is not None) and t < np.linalg.norm(light_position - inter_point)
#
#     # ray = Ray(light_position, inter_point)
#     # t, near_object = find_intersection(scene, ray, stop_at_first=False)
#     # need_shadow = abs(t - np.linalg.norm(inter_point - light_position)) > 1e-6
#
#     return need_shadow