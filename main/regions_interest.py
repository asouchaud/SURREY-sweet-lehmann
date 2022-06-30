import numpy as np
import matplotlib.pyplot as plt
from abc import ABCMeta, abstractmethod
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from utils_general import COLOR, row_wise_norm2, BIG_NUMBER


class RegionOfInterest(metaclass=ABCMeta):
    def __init__(self, n_points, diameter, ratio=0.99, shift=np.asarray((0, 0)), mu=1, region_type='sound',
                 ratio_hole=0, shift_hole=np.asarray((0, 0))):
        self.n_points = n_points
        self.n_mesh = n_points
        self.diam = diameter
        self.ratio = ratio
        self.shift = shift
        self.ratio_hole = ratio_hole * ratio
        self.shift_hole = shift_hole + shift
        self.mu = mu

        self.type = region_type
        self.name = f'{region_type} zone'
        self.color = COLOR[region_type]
        self.form = None
        self.edges = None

        self.points = None
        self.idx_form = None
        self.idx_hole = None
        self.idx_sweet = None
        self.extent = None

    def __str__(self):
        return f'Points of {self.name}: {self.points}]'

    def start(self):
        if self.type == 'complete' or self.form == 'rectangle':
            return *self.shift, *(self.shift + self.diam)
        return *(self.shift + self.shift_correction), *(self.shift + self.diam - self.shift_correction)

    def initialize(self):
        self.normalize()
        self.n_mesh = self.n_m2 ** 2

        # First, we build the interest region as if it's geometry were a square with lenght side 'diam'.
        start_x, start_y, end_x, end_y = self.start()
        mesh_x, mesh_y = np.mgrid[start_x:end_x:self.n_m2 * 1j, start_y:end_y:self.n_m2 * 1j]
        self.points = np.vstack([mesh_x.ravel(), mesh_y.ravel()]).T
        self.extent = np.asarray([start_x, end_x, start_y, end_y])

        # Second, we take off some points to get the desired shape.
        self.edges = self.construct_edges()
        self.idx_form = self.construct_form()
        if self.type != 'complete':
            self.idx_hole = self.construct_hole()
            self.points = self.points[self.idx_form * np.invert(self.idx_hole)]
            self.n_points = len(self.points)
            self.idx_sweet = np.ones(self.n_points, dtype=bool)

    @abstractmethod
    def construct_form(self):
        pass

    @abstractmethod
    def construct_hole(self):
        pass

    @abstractmethod
    def construct_edges(self):
        pass

    @abstractmethod
    def normalize(self):
        pass

    def plot_design(self):
        edges = self.edges
        codes, coords = zip(*((Path.MOVETO, edges[0]), *((Path.LINETO, e) for e in edges), (Path.CLOSEPOLY, edges[0])))
        ax = plt.gca()
        patch = PathPatch(Path(coords, codes),
                          edgecolor=self.color,
                          fill=False)
        ax.add_patch(patch)

    def plot(self, show=False):
        plot = plt.scatter(self.points[:, 0], self.points[:, 1],
                           s=50 * self.area / self.n_points,
                           c=self.color.reshape(1, -1))

        plot_fake = plt.scatter([BIG_NUMBER], [BIG_NUMBER],
                           s=10,
                           c=self.color.reshape(1, -1))
        self.plot_design()
        plt.xlim(np.asarray((-0.1, 1.1)) * self.diam + self.shift[0])
        plt.ylim(np.asarray((-0.1, 1.1)) * self.diam + self.shift[1])

        if show:
            plt.title(f'{self.name.capitalize()}')
            plt.show()
        return plot_fake

    def points_ear(self, side, angle=0, d_type='linear', d_ref=None):
        theta = angle if side == 'right' else angle + np.pi
        if d_type == 'linear':
            displacement = 0.076 * np.asarray([np.cos(theta), np.sin(theta)])  # KEMAR head breadth: 15.2 cm
        elif d_type == 'reference':
            dif = d_ref - self.points
            thetas_reference = theta + np.arctan2(dif[:, 1], dif[:, 0]) - np.pi / 2
            displacement = 0.076 * np.column_stack([np.cos(thetas_reference), np.sin(thetas_reference)])
        else:
            displacement = 0
        return self.points + displacement

    @property
    def n_p2(self):
        return int(np.sqrt(self.n_points))

    @property
    def n_m2(self):
        return int(np.sqrt(self.n_mesh))

    @property
    def mask(self):
        return self.idx_form.reshape((self.n_m2, self.n_m2))

    @property
    def min_dist(self):
        return self.diam * self.ratio / (self.n_m2 - 1)

    @property
    def shift_correction(self):
        return self.diam * (1 - self.ratio) / 2

    @property
    @abstractmethod
    def area(self):
        pass


class SquareRegion(RegionOfInterest):
    def __init__(self, n_points, diameter, ratio=0.99, shift=np.asarray((0, 0)), mu=1, region_type='sound',
                 ratio_hole=0, shift_hole=np.asarray((0, 0))):
        super().__init__(n_points, diameter, ratio, shift, mu, region_type, ratio_hole, shift_hole)
        self.form = 'square'

    def construct_form(self, points=None):
        points, n_points = (self.points, self.n_mesh) if points is None else (points, len(points))
        center = self.diam / 2 + self.shift
        radius = self.diam * self.ratio / 2
        idx_form = np.max(np.abs(points - center), axis=1) <= radius
        return idx_form

    def construct_hole(self, points=None):
        points = self.points if points is None else points
        lim = self.diam * self.ratio_hole / 2
        px = points[:, 0] - (self.shift_hole[0] + self.diam / 2)
        py = points[:, 1] - (self.shift_hole[1] + self.diam / 2)
        idx_hole = (-lim < px) & (px < lim) & (-lim < py) & (py < lim)
        return idx_hole

    def construct_edges(self):
        return np.asarray([(0, 0), (0, 1),
                           (1, 1), (1, 0)]) * self.diam * self.ratio + self.shift + self.shift_correction

    def normalize(self):
        pass

    @property
    def area(self):
        return self.diam ** 2 * (self.ratio - self.ratio_hole) ** 2


class RectangleRegion(RegionOfInterest):
    def __init__(self, n_points, diameter, ratio=0.99, shift=np.asarray((0, 0)), mu=1, region_type='sound',
                 ratio_hole=0, shift_hole=np.asarray((0, 0))):
        super().__init__(n_points, diameter, ratio, shift, mu, region_type, ratio_hole, shift_hole)
        self.form = 'rectangle'

    def construct_form(self, points=None):
        points = self.points if points is None else points
        idx_form = points[:, 1] - self.shift[1] <= self.diam * self.ratio
        return idx_form

    def construct_hole(self, points=None):
        points, n_points = (self.points, self.n_mesh) if points is None else (points, len(points))
        idx_hole = np.zeros(n_points, dtype=bool)
        return idx_hole

    def construct_edges(self):
        return np.asarray([(0, 0), (0, self.ratio), (1, self.ratio), (1, 0)]) * self.diam + self.shift

    def normalize(self):
        self.n_mesh = int(np.sqrt(self.n_mesh / self.ratio)) ** 2

    @property
    def area(self):
        return (self.ratio * self.diam) * self.diam


class OctagonRegion(RegionOfInterest):
    def __init__(self, n_points, diameter, ratio=0.99, shift=np.asarray((0, 0)), mu=1, region_type='sound',
                 ratio_hole=0, shift_hole=np.asarray((0, 0))):
        super().__init__(n_points, diameter, ratio, shift, mu, region_type, ratio_hole, shift_hole)
        self.form = 'octagon'

    def construct_form(self, points=None):
        points = self.points if points is None else points
        gap = self.diam * (1 - self.ratio)
        inf = self.diam * self.ratio / (2 + np.sqrt(2)) + gap
        sup = self.diam * self.ratio * (2 - 1 / (2 + np.sqrt(2))) + gap
        px = points[:, 0] - self.shift[0]
        py = points[:, 1] - self.shift[1]
        idx_form = ((inf <= px + py) & (px + py <= sup)
                    & (inf <= px - py + self.diam) & (px - py + self.diam <= sup))
        return idx_form

    def construct_hole(self, points=None):
        points = self.points if points is None else points
        gap = self.diam * (1 - self.ratio_hole)
        inf = self.diam * self.ratio_hole / (2 + np.sqrt(2)) + gap
        sup = self.diam * self.ratio_hole * (2 - 1 / (2 + np.sqrt(2))) + gap
        start = self.shift_hole + self.diam * (1 - self.ratio_hole) / 2
        end = self.shift_hole + self.diam * (1 + self.ratio_hole) / 2
        px = points[:, 0] - self.shift_hole[0]
        py = points[:, 1] - self.shift_hole[1]
        idx_hole = (inf < px + py) & (px + py < sup) & (inf < px - py + self.diam) & (px - py + self.diam < sup) & \
                   (start[0] < px) & (px < end[0]) & (start[1] < py) & (py < end[1])

        return idx_hole

    def construct_edges(self):
        c1 = 1 / (2 + np.sqrt(2))
        c2 = (1 + np.sqrt(2)) / (2 + np.sqrt(2))
        return np.asarray([(c1, 0), (c2, 0), (1, c1), (1, c2),
                           (c2, 1), (c1, 1), (0, c2), (0, c1)]) * self.diam * self.ratio + \
               self.shift + self.shift_correction

    def normalize(self):
        self.n_mesh = int(self.n_mesh * (1 + 2 / np.sqrt(2)) ** 2 / ((1 + 2 / np.sqrt(2)) ** 2 - 1))

    @property
    def area(self):
        return (1 - 2 / (2 + np.sqrt(2)) ** 2) * self.diam ** 2 * (self.ratio - self.ratio_hole) ** 2


class CircleRegion(RegionOfInterest):
    def __init__(self, n_points, diameter, ratio=0.99, shift=np.asarray((0, 0)), mu=1, region_type='sound',
                 ratio_hole=0, shift_hole=np.asarray((0, 0))):
        super().__init__(n_points, diameter, ratio, shift, mu, region_type, ratio_hole, shift_hole)
        self.form = 'circle'

    def construct_form(self, points=None):
        points = self.points if points is None else points
        center = self.diam / 2 + self.shift
        radius = self.diam * self.ratio / 2
        idx_form = row_wise_norm2(points - center) <= radius
        return idx_form

    def construct_hole(self, points=None):
        points = self.points if points is None else points
        center = self.diam / 2 + self.shift_hole
        radius = self.diam * self.ratio_hole / 2
        idx_hole = row_wise_norm2(points - center) < radius
        return idx_hole

    def construct_edges(self, n=1000):
        return self.diam * self.ratio / 2 * np.asarray([[(1 + np.cos(2 * np.pi * i / n)),
                                                         (1 + np.sin(2 * np.pi * i / n))] for i in range(n)]) \
               + self.shift + self.shift_correction

    def normalize(self):
        self.n_mesh = int(np.sqrt(self.n_mesh * 4 / np.pi)) ** 2

    @property
    def area(self):
        return np.pi / 4 * (self.diam ** 2 * (self.ratio ** 2 - self.ratio_hole ** 2))


FORM = {'square': SquareRegion,
        'line': RectangleRegion,
        'circle': CircleRegion,
        'octagon': OctagonRegion}
