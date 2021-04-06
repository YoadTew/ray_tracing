from modules.Scene import Scene

def read_scene(scene_path):
    with open(scene_path, 'r') as scene_file:
        lines = scene_file.readlines()

        scene = Scene(lines)

        return scene

scene = read_scene('scenes/Room1.txt')

print('Read Scene Done!')
