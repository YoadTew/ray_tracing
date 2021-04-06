from modules.camera import Camera
from modules.settings import Settings

def read_scene(scene_path):
    with open(scene_path, 'r') as scene_file:
        lines = scene_file.readlines()

        for line in lines:
            if line[0] != '#' and line != '\n':
                split_line = line.split()

                if split_line[0] == 'cam':
                    obj = Camera(split_line[1:])
                elif split_line[0] == 'set':
                    obj = Settings(split_line[1:])
                elif split_line[0] == 'mtl':
                    pass
                elif split_line[0] == 'pln':
                    pass
                elif split_line[0] == 'sph':
                    pass
                elif split_line[0] == 'box':
                    pass
                elif split_line[0] == 'lgt':
                    pass

read_scene('scenes/Room1.txt')
