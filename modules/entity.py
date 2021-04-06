import numpy as np

class Entity:
    def __init__(self, params, materials):
        self.material_index = int(params[4])
        self.material = materials[self.material_index - 1]