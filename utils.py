import numpy as np
import time

def normalize(vector):
    norm = np.linalg.norm(vector, axis=-1)

    if vector.ndim > norm.ndim:
        norm = np.expand_dims(norm, -1)

    return vector/norm

def vector_dot(vec1, vec2):
    return np.sum(vec1 * vec2, axis=-1, keepdims=True)

def render_img(scene, camera_ray_origins, camera_ray_directions):
    img = np.full((camera_ray_origins.shape[0], 3), scene.settings.background_color, dtype=float)

    t, nearest_objects = find_intersection(scene, camera_ray_origins, camera_ray_directions)
    mask_inter = nearest_objects >= 0
    img[mask_inter] = find_color(scene,
                                 t[mask_inter],
                                 nearest_objects[mask_inter],
                                 camera_ray_origins[mask_inter],
                                 camera_ray_directions[mask_inter])

    img = np.clip(img, 0, 1) * 255.
    img = img.astype(np.uint8)

    return img

def find_intersection(scene, ray_origins, ray_directions, find_all=False):
    if find_all:
        all_t = np.full((ray_origins.shape[0], len(scene.objects)), np.inf, dtype=float)
        all_objects = np.full((ray_origins.shape[0], len(scene.objects)), -1, dtype=int)

        for idx, entity in enumerate(scene.objects):
            t, mask_inter = entity.intersection(ray_origins, ray_directions)
            all_t[:, idx][mask_inter] = t[mask_inter]
            all_objects[:, idx][mask_inter] = idx

        sort_idx = np.argsort(all_t, 1)
        all_t = np.take_along_axis(all_t, sort_idx, axis=1)
        all_objects = np.take_along_axis(all_objects, sort_idx, axis=1)

        min_t = all_t[:, :2]
        nearest_objects = all_objects[:, :2]
    else:
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

def find_color(scene, t, nearest_objects, source_ray_origins, source_ray_directions, recursion_count=1):
    """
    :param source_ray_origins: All the rays starting points (originally the camera)
    :param source_ray_directions: All the rays directions (originally the intersection point)
    """
    inter_points = source_ray_origins + np.expand_dims(t, 1) * source_ray_directions
    normals = np.empty((source_ray_origins.shape[0], 3), dtype=float)
    diff_colors = np.empty((source_ray_origins.shape[0], 3), dtype=float)
    spec_colors = np.empty((source_ray_origins.shape[0], 3), dtype=float)
    phong_coeffs = np.empty((source_ray_origins.shape[0]), dtype=float)
    reflect_colors = np.empty((source_ray_origins.shape[0], 3), dtype=float)

    for i in range(normals.shape[0]):
        object_idx = nearest_objects[i]
        # if object_idx >= 0:
        normals[i] = scene.objects[object_idx].get_normal(inter_points[i])
        diff_colors[i] = scene.objects[object_idx].material.diffuse_color
        spec_colors[i] = scene.objects[object_idx].material.specular_color
        phong_coeffs[i] = scene.objects[object_idx].material.phong_specularity_coefficient
        reflect_colors[i] = scene.objects[object_idx].material.reflection_color

    diff_spec_colors = calc_diffuse_specular_color(scene,
                                         inter_points,
                                         source_ray_directions,
                                         normals,
                                         diff_colors,
                                         spec_colors,
                                         phong_coeffs)

    reflection_colors = np.full_like(diff_spec_colors, scene.settings.background_color, dtype=float)

    if recursion_count < scene.settings.max_recursion:
        # ref_directions = 2 * vector_dot(source_ray_directions, normals) * normals - source_ray_directions
        ref_directions = source_ray_directions - 2 * vector_dot(source_ray_directions, normals) * normals
        ref_t, ref_nearest_objects = find_intersection(scene, inter_points, ref_directions)
        mask_ref_inter = ref_nearest_objects >= 0

        if mask_ref_inter.any():
            reflection_colors[mask_ref_inter] = find_color(scene,
                                                           ref_t[mask_ref_inter],
                                                           ref_nearest_objects[mask_ref_inter],
                                                           inter_points[mask_ref_inter],
                                                           ref_directions[mask_ref_inter],
                                                           recursion_count + 1)
    reflection_colors *= reflect_colors

    colors = diff_spec_colors + reflection_colors

    return colors

def calc_diffuse_specular_color(scene, inter_points, source_ray_directions, normals, p_diff_colors, p_spec_colors, p_phong_coeffs):
    diff_colors = np.zeros_like(p_diff_colors, dtype=float)
    specular_colors = np.zeros_like(p_spec_colors, dtype=float)

    for light in scene.lights:
        light_ray_hits = is_soft_shadowed(light, inter_points, scene, normals)
        light_ray_directions = normalize(inter_points - light.position)

        ####### Diffuse color #######
        curr_diff_color = \
            np.abs(vector_dot(normals, light_ray_directions)) * \
            p_diff_colors * light.light_color

        curr_diff_color *= ((1 - light.shadow_intensity) + light.shadow_intensity * np.expand_dims(light_ray_hits, -1))

        diff_colors += curr_diff_color

        ####### Specular color #######
        R = 2 * vector_dot(light_ray_directions, normals) * normals- light_ray_directions

        s1 = p_spec_colors
        s2 = np.power(vector_dot(R, -source_ray_directions), np.expand_dims(p_phong_coeffs, -1))
        s3 = light.light_color * light.specular_intensity
        curr_specular_color = s1 * s2 * s3

        curr_specular_color *= ((1 - light.shadow_intensity) + light.shadow_intensity * np.expand_dims(light_ray_hits, -1))

        specular_colors += curr_specular_color

    return diff_colors + specular_colors

def is_soft_shadowed(light, inter_points, scene, normals):
    ray_directions = normalize(inter_points - light.position)
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
            ray_origins[:, idx] = inter_points + (normals * 1e-3)
            rand_point = point + np.random.uniform() * move_y + np.random.uniform() * move_x
            ray_directions[:, idx] = normalize(rand_point - inter_points + (normals * 1e-3))

            point += move_x
        P_0 += move_y

    t, nearest_objects = find_intersection(scene, ray_origins.reshape(-1, 3), ray_directions.reshape(-1, 3))
    t = t.reshape(ray_directions.shape[0], N_squared)
    nearest_objects = nearest_objects.reshape(ray_directions.shape[0], N_squared)

    ray_hits = N_squared - \
                           np.sum(
                               np.logical_and(
                                   t < np.expand_dims(np.linalg.norm(light.position - inter_points, axis=-1), -1),
                                   (nearest_objects >= 0)
                               ), axis=-1
                           )

    percent_hit = ray_hits / N_squared

    return percent_hit
