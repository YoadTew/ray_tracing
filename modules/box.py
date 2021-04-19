import numpy as np
from modules.entity import Entity
from modules.plane import Plane

class Box(Entity):
    def __init__(self, params, materials):
        super().__init__(params, materials)
        self.center = np.array(params[0:3], dtype=float)
        self.scale = float(params[3])

        offset_x = np.array([1, 0, 0]) * (self.scale / 2)
        offset_y = np.array([0, 1, 0]) * (self.scale / 2)
        offset_z = np.array([0, 0, 1]) * (self.scale / 2)

        nx = np.array([1, 0, 0])
        ny = np.array([0, 1, 0])
        nz = np.array([0, 0, 1])

        c_x1 = (self.center - offset_x) @ -nx
        c_x2 = (self.center + offset_x) @ nx
        c_y1 = (self.center - offset_y) @ -ny
        c_y2 = (self.center + offset_y) @ ny
        c_z1 = (self.center - offset_z) @ -nz
        c_z2 = (self.center + offset_z) @ nz

        self.planes = [
            Plane([*-nx, c_x1, self.material_index], materials),
            Plane([*nx, c_x2, self.material_index], materials),
            Plane([*-ny, c_y1, self.material_index], materials),
            Plane([*ny, c_y2, self.material_index], materials),
            Plane([*-nz, c_z1, self.material_index], materials),
            Plane([*nz, c_z2, self.material_index], materials)]

        # self.planes = [
        #     Plane([*-nx, -self.center[0] + (self.scale / 2), self.material_index], materials),
        #     Plane([*nx, self.center[0] + (self.scale / 2), self.material_index], materials),
        #     Plane([*-ny, -self.center[1] + (self.scale / 2), self.material_index], materials),
        #     Plane([*ny, self.center[1] + (self.scale / 2), self.material_index], materials),
        #     Plane([*-nz, -self.center[2] + (self.scale / 2), self.material_index], materials),
        #     Plane([*nz, self.center[2] + (self.scale / 2), self.material_index], materials)]

    def intersection(self, ray_origins, ray_directions):
        min_t = np.full(ray_origins.shape[0], np.inf, dtype=float)
        nearest_objects = np.full(ray_origins.shape[0], -1, dtype=int)

        for idx, plane in enumerate(self.planes):
            t, _ = plane.intersection(ray_origins, ray_directions)

            inter_points = ray_origins + np.expand_dims(t, -1) * ray_directions
            mask_box = np.logical_and(np.all(inter_points <= (self.center + (self.scale / 2)), axis=-1),
                                      np.all(inter_points >= (self.center - (self.scale / 2)), axis=-1))

            new_t = np.copy(min_t)
            new_t[mask_box] = np.minimum(new_t[mask_box], t[mask_box])
            changes = (new_t != min_t)
            nearest_objects[changes] = idx
            min_t = new_t

        mask_inter = (nearest_objects >= 0)

        return min_t, mask_inter

    def intersection_old(self, ray_origins, ray_directions):
        min_t = np.full(ray_origins.shape[0], np.inf, dtype=float)
        nearest_objects = np.full(ray_origins.shape[0], -1, dtype=int)

        for idx, plane in enumerate(self.planes):
            t, mask_inter = plane.intersection(ray_origins, ray_directions)

            new_t = np.copy(min_t)
            new_t[mask_inter] = np.minimum(new_t[mask_inter], t[mask_inter])
            changes = (new_t != min_t)

            nearest_objects[changes] = idx
            min_t = new_t

        mask_inter = (nearest_objects >= 0)

        inter_points = ray_origins + np.expand_dims(min_t, -1) * ray_directions

        mask_edges = np.logical_and(np.all(inter_points < (self.center + (self.scale)), axis=-1),
                                    np.all(inter_points > (self.center - (self.scale)), axis=-1))
        mask_inter = np.logical_and(mask_edges, mask_inter)

        return min_t, mask_inter

    def get_normal(self, point):
        for plane in self.planes:
            if np.abs(point @ plane.normal - plane.offset) < 1e-5:
                return plane.normal

        print('here')