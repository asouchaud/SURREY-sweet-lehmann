import numpy as np
import matplotlib.pyplot as plt
from collections.abc import Iterable
from utils_general import decibels, square, row_wise_norm2, BIG_NUMBER, SIDES
from utils_psychoacousticmodels import eta, gamma, F, C_s, C_a, absolute_threshold_hearing
from green_function import Monopole


class Source:
    def __init__(self, frequencies, positions, intensities, regions_interest=(), green_function=None):
        self.frq = frequencies
        self.pos = positions
        self.int = intensities
        self.n_sources, self.n_freq = None, None
        self.GF = green_function
        self.parse()
        self.u0 = dict()
        self.T_weights = dict()

    def parse(self):
        # First, we expand the the input into a regular form.
        if type(self.frq) in {int, float, np.float64}:
            self.frq = np.asarray([self.frq])
        if type(self.int) in {int, float, np.float64}:
            self.int = np.asarray([[self.int]])
        elif np.asarray(self.int).ndim == 1:
            self.int = np.expand_dims(self.int, axis=0)
        if np.asarray(self.pos).ndim == 1:
            self.pos = np.expand_dims(self.pos, axis=0)
        self.n_sources, self.n_freq = self.int.shape

        # Second, we check if there is a mismatch on the dimensions of the elements of the input.
        if self.n_sources != len(self.pos):
            raise TypeError('The number of rows of "intensities" must be equal to the number of rows of "positions"'
                            ', since in both cases it denotes the number of point sources of u0.')
        if self.n_freq != len(self.frq):
            raise TypeError('The number of columns of "intensities" must be equal to the lenght "frequencies"'
                            ', since in both cases it denotes the number of frequencies of u0.')

    def construct_u0_silence(self, n_points):
        return np.zeros((self.n_freq, n_points))

    def construct_u0_sound(self, points, s=None):
        return np.sum([self.GF.eval(points, self.pos[i], s).T * self.int[i] for i in range(self.n_sources)], axis=0).T

    def construct_u0(self, r):
        if r.type == 'silence':
            u0 = self.construct_u0_silence(r.n_points)
            if self.GF.form == 'HRTF':
                self.u0[r.name] = {s: u0 for s in SIDES}
            else:
                self.u0[r.name] = u0
        else:
            if self.GF.form == 'HRTF':
                self.u0[r.name] = {s: self.construct_u0_sound(r.points, s) for s in SIDES}
                if r.type == 'complete':
                    for s in SIDES:
                        self.u0[r.name][s][:, r.idx_hole] = 0
            else:
                self.u0[r.name] = self.construct_u0_sound(r.points)
                if r.type == 'complete':
                    self.u0[r.name][:, r.idx_hole] = 0

    def construct_T_weights_silence(self, n_points):
        return np.asarray([1 / absolute_threshold_hearing(f, margin=20) ** 2 * np.ones(n_points) for f in self.frq])

    def construct_T_weights_sound(self, A, u0):
        return C_s * np.asarray(
            [np.sum(A[k] / (np.outer(square(u0[k]), A[k]) + C_a), axis=1) for k in range(self.n_freq)])

    def construct_T_weights(self, r):
        if r.type == 'silence':
            T_weights = self.construct_T_weights_silence(r.n_points)
            if self.GF.form == 'HRTF':
                self.T_weights[r.name] = {side: T_weights for side in SIDES}
            else:
                self.T_weights[r.name] = T_weights
        else:
            A = np.asarray([(eta(f) * gamma(f, F)) ** 2 for f in self.frq])
            if self.GF.form == 'HRTF':
                self.T_weights[r.name] = {s: self.construct_T_weights_sound(A, self.u0[r.name][s])
                                          for s in SIDES}
                if r.type == 'complete':
                    for k, f in enumerate(self.frq):
                        for s in SIDES:
                            self.T_weights[r.name][s][k, r.idx_hole] = 1 / absolute_threshold_hearing(f, margin=20) ** 2
            else:
                self.T_weights[r.name] = self.construct_T_weights_sound(A, self.u0[r.name])
                if r.type == 'complete':
                    for k, f in enumerate(self.frq):
                        self.T_weights[r.name][k, r.idx_hole] = 1 / absolute_threshold_hearing(f, margin=20) ** 2

    def initialize(self, regions_interest):
        # If the Green function is not specified we assume that is a monopole.
        if self.GF is None:
            self.GF = Monopole(self.frq)

        if not isinstance(regions_interest, Iterable):
            regions_interest = [regions_interest]

        for r in regions_interest:
            self.construct_u0(r)
            self.construct_T_weights(r)

    def __str__(self):
        sentence = [f'Position {i}: {self.pos[i]}, '
                    f'Frequency intensities {i}: {zip(self.frq, self.int[i])})' for i in range(self.n_sources)]
        return '\n'.join(sentence)

    def plot(self, show=False):
        plot = plt.scatter(self.pos[:, 0], self.pos[:, 1],
                           s=decibels(row_wise_norm2(self.int)) / 2,
                           facecolors='none',
                           edgecolors='black')

        plot_fake = plt.scatter([BIG_NUMBER], [BIG_NUMBER],
                                s=10,
                                facecolors='none',
                                edgecolors='black')
        if show:
            plt.title('Sources position')
            plt.show()
        return plot_fake
