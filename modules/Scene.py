from modules.camera import Camera
from modules.settings import Settings
from modules.material import Material
from modules.plane import Plane
from modules.sphere import Sphere
from modules.box import Box
from modules.light import Light

class Scene:
    def __init__(self, lines):
        self.camera = None
        self.settings = None
        self.materials = []
        self.planes = []
        self.spheres = []
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
                    self.planes.append(Plane(split_line[1:]))
                elif split_line[0] == 'sph':
                    self.spheres.append(Sphere(split_line[1:]))
                elif split_line[0] == 'box':
                    self.boxes.append(Box(split_line[1:]))
                elif split_line[0] == 'lgt':
                    self.lights.append(Light(split_line[1:]))
