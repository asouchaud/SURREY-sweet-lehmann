import numpy as np
import matplotlib.pyplot as plt
import warnings
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from abc import ABCMeta, abstractmethod
from collections.abc import Iterable
from utils_general import BIG_NUMBER, SIDES
from green_function import Monopole


class Loudspeakers(metaclass=ABCMeta):
    def __init__(self, n_loud, diameter, frequencies, regions_interest, green_function):
        self.diam = diameter
        self.n_loud = n_loud
        self.pos = None
        self.ang = None
        self.GF = green_function
        self.G = dict()
        self.initialize(regions_interest, frequencies)

    def __str__(self):
        sentence = [f'Position {i}: {self.pos[i]}, '
                    f'Angle {i}: {self.ang[i]})' for i in range(self.n_loud)]
        return '\n'.join(sentence)

    def construct_G(self, points, s=None):
        return np.moveaxis(np.asarray([self.GF.eval(points, self.pos[i], s) for i in range(self.n_loud)]), 0, -1)

    def initialize(self, regions_interest, frq):
        # If the Green function is not specified we assume that is a monopole.
        if self.GF is None:
            self.GF = Monopole(frq)

        # Check if the loudspeakers form correspond with the definition of the loudspeakers Green function.
        if self.GF.form == 'square room image' and self.form != 'square':
            raise TypeError(f'Loudspeakers must be of form "square" in order to support'
                            f' a Green function of form "square room image".')

        if self.pos is None and self.ang is None:
            self.construct()

        if not isinstance(regions_interest, Iterable):
            regions_interest = [regions_interest]

        for r in regions_interest:
            if self.GF.form == 'HRTF':
                self.G[r.name] = {s: self.construct_G(r.points, s) for s in SIDES}
            else:
                self.G[r.name] = self.construct_G(r.points)

    def plot_design(self, colors=None, ratio=1):
        # normalized coordinates of loudspeaker symbol (see IEC 60617-9)
        codes, coordinates = zip(*(
            (Path.MOVETO, [-0.62, 0.21]),
            (Path.LINETO, [-0.31, 0.21]),
            (Path.LINETO, [0, 0.5]),
            (Path.LINETO, [0, -0.5]),
            (Path.LINETO, [-0.31, -0.21]),
            (Path.LINETO, [-0.62, -0.21]),
            (Path.CLOSEPOLY, [0, 0]),
            (Path.MOVETO, [-0.31, 0.21]),
            (Path.LINETO, [-0.31, -0.21]),
        ))

        path_loudspeakers = Path(np.asarray(coordinates), codes)
        coordinates = np.asarray(coordinates) * self.size * ratio

        if colors is None:
            colors = [0.5 for i in range(self.n_loud)]

        ax = plt.gca()
        for i, (p, theta) in enumerate(zip(self.pos, self.ang)):
            # rotate and translate coordinates
            R = np.asarray([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            transformed_coordinates = np.inner(coordinates, R) + p
            patch = PathPatch(Path(transformed_coordinates[:, :2], codes),
                              edgecolor='0',
                              facecolor=np.tile(1 - colors[i], 3))
            ax.add_patch(patch)
        return path_loudspeakers

    def plot(self, show=False):
        loudspeaker_path = self.plot_design()
        plot = plt.scatter(self.pos[:, 0], self.pos[:, 1],
                           s=500 / self.n_loud,
                           c='y')
        plt.xlim(np.asarray((-0.1, 1.1)) * self.diam)
        plt.ylim(np.asarray((-0.1, 1.1)) * self.diam)

        fake_plot = plt.scatter([BIG_NUMBER], [BIG_NUMBER],
                                marker=loudspeaker_path,
                                edgecolor='black',
                                facecolor='gray')
        if show:
            plt.title('Loudspeakers')
            plt.show()

        return fake_plot

    @abstractmethod
    def construct(self):
        pass

    @property
    @abstractmethod
    def perimeter(self):
        pass

    @property
    def size(self):
        return min(self.perimeter / self.n_loud, 0.1 * self.perimeter)


class SquareLoudspeakers(Loudspeakers):
    """ La geometria de los parlantes corresponde a un cuadrado de lado L. """

    def __init__(self, n_loud, diameter, frequencies, regions_interest=(), green_function=None):
        self.form = 'square'
        super().__init__(n_loud, diameter, frequencies, regions_interest, green_function)

    def check(self):
        if self.n_loud % 4 != 0:
            warnings.warn('The number of square form loudspeakers must be a multiple of 4, it will be truncated.')
            self.n_loud = 4 * (self.n_loud // 4)

    def construct(self):
        self.check()
        k = int(self.n_loud / 4)
        self.pos = np.asarray([[i * self.diam / k, 0] for i in range(k)] +  # down
                              [[self.diam, i * self.diam / k] for i in range(k)] +  # right
                              [[i * self.diam / k, self.diam] for i in range(1, k + 1)] +  # up
                              [[0, i * self.diam / k] for i in range(1, k + 1)])  # left

        self.ang = np.asarray([np.pi / 4] + [np.pi / 2 for i in range(1, k)] +  # down
                              [3 * np.pi / 4] + [np.pi for i in range(1, k)] +  # right
                              [3 * np.pi / 2 for i in range(k - 1)] + [5 * np.pi / 4] +  # up
                              [0 for i in range(k - 1)] + [7 * np.pi / 4])  # left

    @property
    def perimeter(self):
        return 4 * self.diam


class LineLoudspeakers(Loudspeakers):
    """ La geometria de los parlantes correponde a un segmento de largo L. """

    def __init__(self, n_loud, diameter, frequencies, regions_interest=(), green_function=None):
        self.form = 'line'
        super().__init__(n_loud, diameter, frequencies, regions_interest, green_function)

    def construct(self):
        self.pos = np.asarray([[i * self.diam / (self.n_loud - 1), self.diam] for i in range(self.n_loud)])
        self.ang = np.asarray([3 * np.pi / 2 for i in range(self.n_loud)])

    @property
    def perimeter(self):
        return self.diam


class OctagonLoudspeakers(Loudspeakers):
    """ La geometria de los parlantes correponde a un octagono inscrito en un cuadrado de lado L. """

    def __init__(self, n_loud, diameter, frequencies, regions_interest=(), green_function=None):
        self.form = 'octagon'
        super().__init__(n_loud, diameter, frequencies, regions_interest, green_function)

    def check(self):
        if self.n_loud % 8 != 0:
            warnings.warn(f'The number of octagon form loudspeakers must be a multiple of 8, it will be truncated.')
            self.n_loud = 8 * (self.n_loud // 8)

    def construct(self):
        self.check()
        k = int(self.n_loud / 8)
        x = self.diam / (2 + np.sqrt(2))
        y = self.diam * np.sqrt(2) / (2 + np.sqrt(2))
        self.pos = np.asarray([[x + i * y / k, 0] for i in range(k + 1)] +  # down
                              [[x + i * y / k, self.diam] for i in range(k + 1)] +  # up
                              [[0, x + i * y / k] for i in range(k + 1)] +  # left
                              [[self.diam, x + i * y / k] for i in range(k + 1)] +  # right
                              [[i * x / k, x - i * x / k] for i in range(1, k)] +  # down-left
                              [[x + y + i * x / k, i * x / k] for i in range(1, k)] +  # down-right
                              [[i * x / k, x + y + i * x / k] for i in range(1, k)] +  # up-left
                              [[x + y + i * x / k, 2 * x + y - i * x / k] for i in range(1, k)])  # up-right
        self.ang = np.asarray([3 * np.pi / 8] + [np.pi / 2 for i in range(1, k)] + [5 * np.pi / 8] +  # down
                              [13 * np.pi / 8] + [3 * np.pi / 2 for i in range(1, k)] + [11 * np.pi / 8] +  # up
                              [np.pi / 8] + [0 for i in range(1, k)] + [15 * np.pi / 8] +  # left
                              [7 * np.pi / 8] + [np.pi for i in range(1, k)] + [9 * np.pi / 8] +  # right
                              [np.pi / 4 for i in range(1, k)] + [3 * np.pi / 4 for i in range(1, k)] +  # d-l, d-r
                              [7 * np.pi / 4 for i in range(1, k)] + [5 * np.pi / 4 for i in range(1, k)])  # u-l, u-r

    @property
    def perimeter(self):
        return 8 * np.sqrt(2) / (2 + np.sqrt(2)) * self.diam


class CircleLoudspeakers(Loudspeakers):
    """ La geometria de los parlantes correponde a un circulo inscrito en un cuadrado de lado L. """

    def __init__(self, n_loud, diameter, frequencies, regions_interest=(), green_function=None):
        self.form = 'circle'
        super().__init__(n_loud, diameter, frequencies, regions_interest, green_function)

    def construct(self):
        self.pos = np.asarray([[self.diam / 2 * (1 + np.cos(2 * np.pi * i / self.n_loud)),
                                self.diam / 2 * (1 + np.sin(2 * np.pi * i / self.n_loud))] for i in range(self.n_loud)])
        self.ang = np.asarray([np.pi + 2 * np.pi * i / self.n_loud for i in range(self.n_loud)])

    @property
    def perimeter(self):
        return np.pi * self.diam


class DoubleCircleLoudspeakers(Loudspeakers):
    def __init__(self, n_loud, diameter, frequencies=(), regions_interest=(), green_function=None):
        self.form = 'circle'
        super().__init__(n_loud, diameter, frequencies, regions_interest, green_function)

    def check(self):
        if self.n_loud % 2 != 0:
            warnings.warn(f'The number of 2-circle form loudspeakers must be a multiple of 2, it will be truncated.')
            self.n_loud = 2 * (self.n_loud // 2)

    def construct(self, epsilon=0.000005):
        self.check()
        n_row = int(self.n_loud / 2)
        r1 = self.diam / 2
        r2 = r1 * (1 + epsilon)
        self.pos = np.asarray([(r1 * (1 + np.cos(2 * np.pi * i / n_row)),
                                r1 * (1 + np.sin(2 * np.pi * i / n_row))) for i in range(n_row)] +
                              [(r1 + r2 * np.cos(2 * np.pi * i / n_row),
                                r1 + r2 * np.sin(2 * np.pi * i / n_row)) for i in range(n_row)])
        self.ang = np.asarray([np.pi + 2 * np.pi * i / n_row for i in range(n_row)] +
                              [np.pi + 2 * np.pi * i / n_row for i in range(n_row)])

    @property
    def perimeter(self):
        return np.pi * self.diam

    @property
    def size(self):
        return min(self.perimeter / (self.n_loud / 2), 0.1 * self.perimeter)
