import numpy as np
import time

def normalize(vector):
    norm = np.linalg.norm(vector, axis=-1)

    if vector.ndim > norm.ndim:
        norm = np.expand_dims(norm, -1)

    return vector/norm

def vector_dot(vec1, vec2):
    return np.sum(vec1 * vec2, axis=-1, keepdims=True)

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
    mask_inter = nearest_objects >= 0
    normals = np.empty((camera_ray_origins.shape[0], 3), dtype=float)
    diff_colors = np.empty((camera_ray_origins.shape[0], 3), dtype=float)
    spec_colors = np.empty((camera_ray_origins.shape[0], 3), dtype=float)
    phong_coeffs = np.empty((camera_ray_origins.shape[0]), dtype=float)

    for i in range(normals.shape[0]):
        object_idx = nearest_objects[i]
        if object_idx >= 0:
            normals[i] = scene.objects[object_idx].get_normal(inter_points[i])
            diff_colors[i] = scene.objects[object_idx].material.diffuse_color
            spec_colors[i] = scene.objects[object_idx].material.specular_color
            phong_coeffs[i] = scene.objects[object_idx].material.phong_specularity_coefficient

    colors = calc_diffuse_specular_color(scene,
                                         inter_points,
                                         camera_ray_directions,
                                         normals,
                                         diff_colors,
                                         spec_colors,
                                         phong_coeffs,
                                         mask_inter)

    return colors.reshape(img_height, img_width, 3)

def calc_diffuse_specular_color(scene, inter_points, camera_ray_directions, normals, p_diff_colors, p_spec_colors, p_phong_coeffs, mask_inter):
    colors = np.full_like(p_diff_colors, scene.settings.background_color, dtype=float)
    diff_colors = np.zeros_like(p_diff_colors, dtype=float)
    specular_colors = np.zeros_like(p_spec_colors, dtype=float)

    for light in scene.lights[:]:
        light_ray_hits = is_soft_shadowed(light, inter_points, scene, normals, mask_inter)#np.ones(normals.shape[0], dtype=float)
        ray_directions = np.empty_like(inter_points, float)
        ray_directions[mask_inter] = normalize(inter_points[mask_inter] - light.position)

        ####### Diffuse color #######
        curr_diff_color = np.zeros_like(diff_colors, dtype=float)
        curr_diff_color[mask_inter] = \
            np.abs(vector_dot(normals[mask_inter], ray_directions[mask_inter])) * \
            p_diff_colors[mask_inter] * light.light_color

        curr_diff_color[mask_inter] *= ((1 - light.shadow_intensity) + light.shadow_intensity * np.expand_dims(light_ray_hits[mask_inter], -1))

        diff_colors[mask_inter] += curr_diff_color[mask_inter]

        ####### Specular color #######
        curr_specular_color = np.zeros_like(specular_colors, dtype=float)
        R = 2 * vector_dot(ray_directions[mask_inter], normals[mask_inter]) * normals[mask_inter] - ray_directions[mask_inter]

        a = p_spec_colors[mask_inter]
        b = np.power(vector_dot(R,-camera_ray_directions[mask_inter]), np.expand_dims(p_phong_coeffs[mask_inter], -1))
        c = light.light_color * light.specular_intensity
        curr_specular_color[mask_inter] = a * b * c

        curr_specular_color[mask_inter] *= ((1 - light.shadow_intensity) + light.shadow_intensity * np.expand_dims(light_ray_hits[mask_inter], -1))

        specular_colors[mask_inter] += curr_specular_color[mask_inter]

    colors[mask_inter] = diff_colors[mask_inter] + specular_colors[mask_inter]
    colors = np.clip(colors, 0, 1) * 255.
    colors = colors.astype(np.uint8)

    return colors

def is_soft_shadowed(light, inter_points, scene, normals, mask_inter):
    ray_directions = normalize(inter_points[mask_inter] - light.position)
    ray_hits = np.zeros(inter_points.shape[0], int)

    Vz = ray_directions
    Vx = normalize(np.cross(scene.camera.up_vector, Vz))
    Vy = normalize(np.cross(Vx, Vz))

    p_center = light.position
    p_radius = light.light_radius

    P_0 = p_center - (p_radius / 2) * Vx - (p_radius / 2) * Vy
    move_x = (Vx * p_radius) / scene.settings.soft_shadow_N
    move_y = (Vy * p_radius) / scene.settings.soft_shadow_N

    N_squared = scene.settings.soft_shadow_N * scene.settings.soft_shadow_N

    ray_origins = np.zeros((ray_directions.shape[0], N_squared, 3), dtype=float)
    ray_directions = np.zeros((ray_directions.shape[0], N_squared, 3), dtype=float)

    for i in range(scene.settings.soft_shadow_N):
        point = np.copy(P_0)
        for j in range(scene.settings.soft_shadow_N):
            idx = i * scene.settings.soft_shadow_N + j
            ray_origins[:, idx] = inter_points[mask_inter] + (normals[mask_inter] * 1e-3)
            rand_point = point + np.random.uniform() * move_y + np.random.uniform() * move_x
            ray_directions[:, idx] = normalize(rand_point - inter_points[mask_inter] + (normals[mask_inter] * 1e-3))

            point += move_x
        P_0 += move_y

    t, nearest_objects = find_intersection(scene, ray_origins.reshape(-1, 3), ray_directions.reshape(-1, 3))
    t = t.reshape(ray_directions.shape[0], N_squared)
    nearest_objects = nearest_objects.reshape(ray_directions.shape[0], N_squared)

    ray_hits[mask_inter] = N_squared - \
                           np.sum(
                               np.logical_and(
                                   t < np.expand_dims(np.linalg.norm(light.position - inter_points[mask_inter], axis=-1), -1),
                                   (nearest_objects >= 0)
                               ), axis=-1
                           )

    percent_hit = ray_hits / N_squared

    return percent_hit