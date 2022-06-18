import numpy as np
from matplotlib import pyplot as plt
from utils_general import row_wise_norm2, SMALL_NUMBER, SIDES, rotation_45
from abc import ABCMeta, abstractmethod
from itertools import product
import scipy.interpolate as sc
from utils_hrtf import hrtf_generator, hrtf_bank
import matplotlib.colors as colors
from collections import Iterable
from time import time


class GreenFunction(metaclass=ABCMeta):
    def __init__(self, frequencies, c=343):
        self.frequencies = np.asarray(frequencies) if isinstance(frequencies, Iterable) else np.asarray([frequencies])
        self.c = c
        self.k = 2 * np.pi * self.frequencies / c
        self.form = None

    @abstractmethod
    def eval(self, x, x0, side):
        pass


class HRTF(GreenFunction):
    def __init__(self, frequencies, theta=0, reference=None, c=343):
        super().__init__(frequencies, c)
        self.theta = theta
        self.form = 'HRTF'
        self.reference = reference
        self.domain = np.linspace(0, 1, 1025) * 22050
        t0 = time()
        self.hrtf_bank = {side: [sc.interp1d(self.domain, hrtf)(self.frequencies) for hrtf in hrtf_bank[side][3]]
                          for side in SIDES}
        t1 = time()
        print('Time A', t1 - t0)

    def adjust_delay(self, signal, new_distance, old_distance):
        delay_distance = new_distance - old_distance
        delay_time = delay_distance / self.c
        complex_exponential = np.e ** (-2 * np.pi * 1j * delay_time * self.frequencies)
        delayed_signal = signal * complex_exponential
        return delayed_signal

    def adjust_amplitude(self, signal, new_distance, old_distance):
        attenuated_signal = signal * abs(old_distance / new_distance)
        return attenuated_signal

    def adjust(self, signal, new_distance, old_distance):
        signal_delayed = self.adjust_delay(signal, new_distance, old_distance)
        signal_del_att = self.adjust_amplitude(signal_delayed, new_distance, old_distance)
        return signal_del_att

    def hrtf_generator(self, side, distance, index):
        #sexagesimal_angle = radian_angle * 180 / np.pi
        #index = int((sexagesimal_angle + 90) % 360)  # index = 0 <-> sexagesimal_angle = -90 in the standard reference sys.
        return self.adjust(self.hrtf_bank[side][index], distance, 3)

    def hrtf(self, x, x0, side):
        dif = x0 - x  # Be aware of the sign
        distances = row_wise_norm2(dif)
        # % modular function always return a positive number
        angles = (np.arctan2(dif[:, 1], dif[:, 0]) - self.theta) % (2 * np.pi)
        if self.reference is not None:
            dif_ref = self.reference - x
            angles = (angles - (np.arctan2(dif_ref[:, 1], dif_ref[:, 0]) - 1 / 2 * np.pi) % (2 * np.pi)) % (2 * np.pi)
        indices = ((90 + angles * 180 / np.pi) % 360).astype(np.int) # index = 0 <-> sexagesimal_angle = -90 in the standard reference sys.
        # array = np.asarray([sc.interp1d(self.domain, hrtf_generator(side, distance, angle))(self.frequencies)
        #                     for distance, angle in zip(distances, angles)]).T
        array = np.asarray([self.hrtf_generator(side, distance, index) for distance, index in zip(distances, indices)]).T
        return array

    def eval(self, x, x0, side):
        return self.hrtf(x, x0, side)

    def eval(self, x, x0, side):
        dif = x0 - x  # Be aware of the sign
        distances = row_wise_norm2(dif)
        angles = (np.arctan2(dif[:, 1], dif[:, 0]) - self.theta + 4 * np.pi) % (2 * np.pi)
        if self.reference is not None:
            dif_ref = self.reference - x
            angles -= (np.arctan2(dif_ref[:, 1], dif_ref[:, 0]) + np.pi * 7 / 2) % (2 * np.pi)
            angles = (angles + 4 * np.pi) % (2 * np.pi)
        array = np.asarray([sc.interp1d(self.domain, hrtf_generator(side, distance, angle))(self.frequencies)
                            for distance, angle in zip(distances, angles)]).T
        return array

# fig, ax = plt.subplots()
# ax.set_aspect('equal')
# vmax = np.max(angles)
# vmin = np.min(angles)
# m = int(np.sqrt(len(angles)))
# print('min', np.min(angles))
# print('max', np.max(angles))
# if vmin < np.pi < vmax:
#     divnorm = colors.TwoSlopeNorm(vmin=vmin, vmax=vmax, vcenter=np.pi)
#     im = ax.imshow(rotation_45(angles.reshape(m, m)),
#                    cmap=plt.cm.hsv,
#                    norm=divnorm,
#                    extent=np.asarray([0, 5, 0, 5]),
#                    interpolation='spline16')
# else:
#     im = ax.imshow(rotation_45(angles.reshape(m, m)),
#                    cmap=plt.cm.hsv,
#                    extent=np.asarray([0, 5, 0, 5]),
#                    interpolation='spline16')
# vec = x0 - np.asarray([2.5, 2.5])
# plt.savefig(fname=f'pos_{vec[0]};{vec[1]}'.replace('.', ','), dpi=300)
# plt.show()


class Monopole(GreenFunction):
    def __init__(self, frequencies, c=343):
        super().__init__(frequencies, c)
        self.form = 'monopole'

    def monopole3D(self, x, x0):
        r = row_wise_norm2(x - x0)
        return np.e ** (-1j * np.outer(self.k, r)) / r

    def monopole2D(self, x, x0):
        r = row_wise_norm2(x - x0)

    def eval(self, x, x0, side=None):
        return self.monopole3D(x, x0)


class MonopoleScattered(Monopole):
    def __init__(self, frequencies, positions, capacities=None, c=343):
        super().__init__(frequencies, c)
        self.form = 'scatter 3D'
        self.pos = positions
        self.n_s = len(positions)
        self.n_k = len(self.frequencies)
        self.cap = capacities if capacities is not None else np.ones(self.n_s)

    def eval(self, x, x0, side=None):
        # solve Foldy's equations
        g = np.asarray([-1j * self.k * self.cap[i] for i in range(self.n_s)])
        A = np.moveaxis(np.asarray([[1 / g[i] if i == j else -self.monopole3D(self.pos[i], self.pos[j])[:, 0]
                                     for j in range(self.n_s)] for i in range(self.n_s)]), source=2, destination=0)
        b = self.monopole3D(self.pos, x0)
        z = np.asarray([np.linalg.solve(A[k], b[k]) for k in range(self.n_k)])
        u_inc = self.monopole3D(x, x0)
        u_sca = np.sum([[(z[k][i] * self.monopole3D(x, self.pos[i])[k])
                         for k in range(self.n_k)] for i in range(self.n_s)], axis=0)
        return u_inc + u_sca


# TODO arreglar multiherencia
class HRTFScattered(HRTF, Monopole):
    def __init__(self, frequencies, positions, capacities=None, theta=0, reference=None, c=343):
        self.frequencies = np.asarray(frequencies) if isinstance(frequencies, Iterable) else np.asarray([frequencies])
        self.k = 2 * np.pi * self.frequencies / c
        # self.frequencies = frequencies
        self.theta = theta
        self.form = 'HRTF'
        self.reference = reference
        self.domain = np.linspace(0, 1, 1025) * 22050
        self.pos = positions
        self.n_s = len(positions)
        self.n_k = len(self.frequencies)
        self.cap = capacities if capacities is not None else np.ones(self.n_s)

    def eval(self, x, x0, side=None):
        # solve Foldy's equations
        g = np.asarray([-1j * self.k * self.cap[i] for i in range(self.n_s)])
        A = np.moveaxis(np.asarray([[1 / g[i] if i == j else -self.monopole3D(self.pos[i], self.pos[j])[:, 0]
                                     for j in range(self.n_s)] for i in range(self.n_s)]), source=2, destination=0)
        b = self.monopole3D(self.pos, x0)
        z = np.asarray([np.linalg.solve(A[k], b[k]) for k in range(self.n_k)])
        u_inc = self.hrtf(x, x0, side)
        u_sca = np.sum([[(z[k][i] * self.hrtf(x, self.pos[i], side)[k])
                         for k in range(self.n_k)] for i in range(self.n_s)], axis=0)
        return u_inc + u_sca


class GFSquareImage2D(Monopole):
    def __init__(self, frequencies, diameter, beta=1, n_order=5, c=343):
        super().__init__(frequencies, c)
        self.form = 'square room image'
        self.beta = beta
        self.diam = diameter
        self.center = diameter / 2 * np.ones(2)
        self.n_order = n_order
        self.walls_direction = diameter / 2 * np.asarray(((0, 1), (1, 0), (0, -1), (-1, 0)))
        self.walls_inversion = np.asarray(((1, -1), (-1, 1), (1, -1), (-1, 1)))
        self.walls_coordinate = np.asarray((1, 0, 1, 0))
        self.walls_sign = np.asarray((1, 1, -1, -1))
        self.image_positions = None

    def eval(self, x, x0, side=None):
        # print('x0', x0)
        self.image_positions = []
        # We shift the origin of coordinates to the center of the room.
        self.image_positions_definition(x0 - self.center, self.n_order)
        # self.image_positions = np.asarray(self.image_positions)
        # plt.scatter(self.image_positions[:, 0], self.image_positions[:, 1])
        # plt.plot()
        # plt.show()
        # print('Image positions', self.image_positions)
        return np.sum([self.monopole3D(x, xi) * self.beta ** exp_beta for xi, exp_beta in self.image_positions], axis=0)

    def image_positions_definition(self, incoming_dir, n_order):
        self.image_positions.append((incoming_dir + self.center, self.n_order - n_order))
        if n_order > 0:
            for wall_dir, wall_inv, wall_coo, wall_sig in zip(self.walls_direction, self.walls_inversion,
                                                              self.walls_coordinate, self.walls_sign):
                if (incoming_dir - wall_dir)[wall_coo] * wall_sig <= - SMALL_NUMBER:
                    new_dir = incoming_dir * wall_inv + 2 * wall_dir
                    self.image_positions_definition(new_dir, n_order - 1)


class GFSquareImage3D(Monopole):
    def __init__(self, frequencies, diameter, height, beta=0, n_order=5, c=343):
        super().__init__(frequencies, c)
        self.form = 'square room image'
        self.beta = beta
        self.diam = diameter
        self.height = height
        self.n_order = n_order
        self.center = 1 / 2 * np.asarray((diameter, diameter, height))
        self.walls_direction = 1 / 2 * np.asarray(((diameter, 0, 0), (-diameter, 0, 0),  # right, left
                                                   (0, diameter, 0), (0, -diameter, 0),  # in, out
                                                   (0, 0, height), (0, 0, -height)))  # up, down
        self.walls_inversion = np.asarray(((-1, 1, 1), (-1, 1, 1),  # right, left
                                           (1, -1, 1), (1, -1, 1),  # in, out
                                           (1, 1, -1), (1, 1, -1)))  # up, down
        self.walls_coordinate = np.asarray((0, 0, 1, 1, 2, 2))
        self.walls_sign = np.asarray((1, -1, 1, -1, 1, -1))
        self.image_positions = None

    @staticmethod
    def parse(z):
        if z.ndim == 1 and z.shape[0] == 2:
            z = np.append(z, 1.7)
        if z.ndim == 2 and z.shape[1] == 2:
            z = np.c_[z, 1.7 * np.ones(len(z))]
        return z

    def eval(self, x, x0, side=None):
        x = self.parse(x)
        x0 = self.parse(x0)

        self.image_positions = []
        # We shift the origin of coordinates to the center of the room.
        self.image_positions_definition(x0 - self.center, self.n_order)

        return np.sum([self.monopole3D(x, xi) * self.beta ** exp_beta for xi, exp_beta in self.image_positions], axis=0)

    def image_positions_definition(self, incoming_dir, n_order):
        self.image_positions.append((incoming_dir + self.center, self.n_order - n_order))
        if n_order > 0:
            for wall_dir, wall_inv, wall_coo, wall_sig in zip(self.walls_direction, self.walls_inversion,
                                                              self.walls_coordinate, self.walls_sign):
                if (incoming_dir - wall_dir)[wall_coo] * wall_sig <= - SMALL_NUMBER:
                    new_dir = incoming_dir * wall_inv + 2 * wall_dir
                    self.image_positions_definition(new_dir, n_order - 1)

# TODO arreglar multiherencia
class HRTFSquareImage2D(GFSquareImage2D, HRTF):
        def __init__(self, frequencies, diameter, theta=0, reference=None, beta=0, n_order=5, c=343):
            self.frequencies = np.asarray(frequencies) if isinstance(frequencies, Iterable) else np.asarray(
                [frequencies])
            self.k = 2 * np.pi * self.frequencies / c

            self.frequencies = frequencies
            self.theta = theta
            self.form = 'HRTF'
            self.reference = reference
            self.domain = np.linspace(0, 1, 1025) * 22050

            self.beta = beta
            self.diam = diameter
            self.center = diameter / 2 * np.ones(2)
            self.n_order = n_order
            self.walls_direction = diameter / 2 * np.asarray(((0, 1), (1, 0), (0, -1), (-1, 0)))
            self.walls_inversion = np.asarray(((1, -1), (-1, 1), (1, -1), (-1, 1)))
            self.walls_coordinate = np.asarray((1, 0, 1, 0))
            self.walls_sign = np.asarray((1, 1, -1, -1))
            self.image_positions = None

        def eval(self, x, x0, side=None):
            self.image_positions = []
            self.image_positions_definition(x0 - self.center, self.n_order)
            return np.sum([self.hrtf(x, xi, side) * self.beta ** exp_beta for xi, exp_beta in self.image_positions],
                          axis=0)


class GFSquareModal(GreenFunction):
    def __init__(self, frequencies, diameter, height=2.5, beta=0, n_modes=10, c=343):
        super().__init__(frequencies, c)
        self.form = 'square room modal'
        self.beta = beta
        self.diam = diameter
        self.height = height
        self.volume = diameter ** 2 * height
        self.n_modes = n_modes

        aux = 2 * 1j * beta * self.k * diameter
        self.q = np.asarray([1 / (np.pi * 1j) * np.sqrt(aux)] +
                            [n - aux / (np.pi ** 2 * n) for n in range(1, n_modes)])

        aux_height = 2 * 1j * beta * self.k * height
        self.q_height = np.asarray([1 / (np.pi * 1j) * np.sqrt(aux_height)] +
                                   [n - aux_height / (np.pi ** 2 * n) for n in range(1, n_modes)])

        self.k2 = np.asarray([np.pi ** 2 * (2 * (self.q[n] / diameter) ** 2 + (self.q_height[n] / height) ** 2)
                              for n in range(n_modes)])
        self.A = np.asarray([self.montecarlo_sum(n) for n in range(n_modes)])

    def psi(self, n, x):
        return np.nanprod([self.psi_coord(n, x[:, 0], self.q, self.diam),
                           self.psi_coord(n, x[:, 1], self.q, self.diam),
                           self.psi_coord(n, x[:, 2], self.q_height, self.height)], axis=0)

    def psi_coord(self, n, x, q, L):
        return np.cos(q[n] * np.pi * x / L + 1j * self.beta * self.k * L / (np.pi * q[n]))

    def psi_norm(self, n, x):
        return self.psi(n, x) / self.A[n]

    def montecarlo_sum(self, n):
        C = 100000
        n_points = int(self.volume * C)
        eval_points = np.random.random((n_points, 3)) * np.asarray((self.diam, self.diam, self.height))
        return np.nansum(np.square(np.abs(self.psi(n, eval_points))), axis=0) * self.volume / n_points

    def eval(self, x, x0, side=None, d_type=None, d_ref=None):
        x = self.parse(x)
        x0 = self.parse(x0)
        return np.expand_dims(np.sum([self.psi_norm(n, x) * self.psi_norm(n, x0) / (self.k2[n] - self.k ** 2)
                                      for n in range(self.n_modes)], axis=0) / self.volume, axis=0)

    @staticmethod
    def parse(z):
        if z.ndim == 1:
            z = np.expand_dims(z, axis=0)
        if z.shape[1] == 2:
            z = np.c_[z, 1.7 * np.ones(len(z))]
        return z


GREEN_MAP = {'monopole': Monopole,
             'square modal': GFSquareModal,
             'square image': GFSquareImage2D
             }
