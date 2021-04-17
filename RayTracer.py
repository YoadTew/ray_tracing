import argparse
import os
import time
import numpy as np
from PIL import Image

import modules.sphere
from modules.Scene import Scene
from utils import normalize, render_img

def read_args():
    parser = argparse.ArgumentParser(description='Ray tracer running script')

    parser.add_argument('scene_path', help='path to scene file')
    parser.add_argument('img_path', help='path to save image')

    parser.add_argument('img_width', default=500, nargs='?', type=int, help='Image width (default: 500)')
    parser.add_argument('img_height', default=500, nargs='?', type=int, help='Image height (default: 500)')

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

def calc_screen_vectors(scene):
    Vz = normalize(scene.camera.look_at - scene.camera.position)
    Vx = normalize(np.cross(scene.camera.up_vector, Vz))
    Vy = normalize(np.cross(Vx, Vz))

    return Vx, Vy, Vz

def main():
    args = read_args()
    scene = read_scene(args.scene_path)

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

    camera_ray_origins = np.zeros((args.img_height, args.img_width, 3), dtype=float)
    camera_ray_directions = np.zeros((args.img_height, args.img_width, 3), dtype=float)

    start = time.time()
    for i in range(args.img_height):
        pixel = np.copy(P_0)  # Current pixel location
        for j in range(args.img_width):
            camera_ray_origins[i, j] = scene.camera.position
            camera_ray_directions[i, j] = normalize(pixel - scene.camera.position)

            pixel += move_x
        P_0 += move_y

    camera_ray_origins = camera_ray_origins.reshape(-1, 3)
    camera_ray_directions = camera_ray_directions.reshape(-1, 3)

    img = render_img(scene, camera_ray_origins, camera_ray_directions)
    img = img.reshape(args.img_height, args.img_width, 3)

    print(f'Render took {time.time() - start} seconds')

    save_img(img, args.img_path)

if __name__ == '__main__':
    np.random.seed(0)

    main()