from modules.camera import Camera
from modules.settings import Settings
from modules.material import Material

class Scene:
    def __init__(self, lines):
        self.camera = None
        self.settings = None
        self.materials = []
        self.spheres = []
        self.planes = []
        self.boxes = []
        self.lights = []

        for line in lines:
            if line[0] != '#' and line != '\n':
                split_line = line.split()

                if split_line[0] == 'cam':
                    self.camera = Camera(split_line[1:])
                elif split_line[0] == 'set':
                    self.settings = Settings(split_line[1:])
                elif split_line[0] == 'mtl':
                    material = Material(split_line[1:], len(self.materials) + 1)
                    self.materials.append(material)
                elif split_line[0] == 'pln':
                    pass
                elif split_line[0] == 'sph':
                    pass
                elif split_line[0] == 'box':
                    pass
                elif split_line[0] == 'lgt':
                    pass