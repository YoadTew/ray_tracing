import argparse
import os
import time
import numpy as np
from PIL import Image

from modules.Scene import Scene
from modules.ray import Ray
from utils import normalize

def read_args():
    parser = argparse.ArgumentParser(description='Ray tracer running script')

    parser.add_argument('scene_path', help='path to scene file')
    parser.add_argument('img_path', help='path to save image')

    parser.add_argument('img_width', default=500, nargs='?', type=float, help='Image width (default: 500)')
    parser.add_argument('img_height', default=500, nargs='?', type=float, help='Image height (default: 500)')

    args = parser.parse_args()

    return args

def read_scene(scene_path):
    with open(scene_path, 'r') as scene_file:
        lines = scene_file.readlines()

        scene = Scene(lines)

        return scene

def save_img(img, img_path):
    PIL_img = Image.fromarray(img)
    PIL_img.save(img_path)

    print('Image saved!')

def calc_screen_vectors_old(scene):
    Vz = normalize(scene.camera.look_at - scene.camera.position)
    a, b, c = Vz

    Sx = -b
    Cx = np.sqrt(1 - Sx ** 2)
    Sy = -a / Cx
    Cy = c / Cx

    M = np.matrix([[Cy, 0, Sy], [-Sx * Sy, Cx, Sx * Cy], [-Cx * Sy, -Sx, Cx * Cy]])
    Vx = np.matmul(M, [1, 0, 0])
    Vy = np.matmul(M, [0, 1, 0])

    return Vx, Vy, Vz

def calc_screen_vectors(scene):
    Vz = normalize(scene.camera.look_at - scene.camera.position)
    Vx = normalize(np.cross(scene.camera.up_vector, Vz))
    Vy = normalize(np.cross(Vx, Vz))

    return Vx, Vy, Vz

def find_intersection(scene, ray):
    min_t = np.inf
    nearest_object = None

    for sphere in scene.spheres:
        t = sphere.intersection(ray)

        if t and t < min_t:
            min_t = t
            nearest_object = sphere

    return t, nearest_object

def main():
    args = read_args()
    scene = read_scene(args.scene_path)

    img = np.zeros((args.img_height, args.img_width, 3), dtype=np.uint8)

    # Calculate normalize screen vectors
    Vx, Vy, Vz = calc_screen_vectors(scene)

    # Calc screen values
    s_center = scene.camera.position + Vz * scene.camera.screen_distance
    s_width = scene.camera.screen_width
    s_height = (args.img_height / args.img_width) * s_width

    # Screen corner and moving pixel vectors
    P_0 = s_center - (s_width / 2) * Vx - (s_height / 2) * Vy
    move_x = (Vx * s_width) / args.img_width
    move_y = (Vy * s_height) / args.img_height

    start = time.time()
    for i in range(args.img_height):
        pixel = np.copy(P_0)  # Current pixel location
        for j in range(args.img_width):
            ray = Ray(scene.camera.position, pixel)
            t, nearest_object = find_intersection(scene, ray)

            if nearest_object:
                img[i, j] = nearest_object.material.diffuse_color * 255.
            else:
                img[i, j] = 0 #scene.settings.background_color * 255.

            pixel += move_x
        P_0 += move_y
    end = time.time()

    print(f'Render took {end - start} seconds')

    save_img(img, args.img_path)

if __name__ == '__main__':
    main()