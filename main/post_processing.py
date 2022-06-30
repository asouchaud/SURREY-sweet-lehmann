import numpy as np
import matplotlib.pyplot as plt
from regions_interest import FORM, RegionOfInterest
from loudspeakers import Loudspeakers
from source import Source
from utils_general import decibels, square, SMALL_NUMBER, g3dv, normalized, rotation_45, rotation_inverse_45, SIDES
from utils_psychoacousticmodels import knobel_model, sherlock_model
from green_function import Monopole
from collections.abc import Iterable
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as ticker
import matplotlib.colors as colors
from copy import copy

FIG_SIZE = (5, 5)


class PostProcessing:
    def __init__(self, solution, source, loudspeakers, regions_interest, green_function=None, n_points=1000 ** 2,
                 r=1.5):
        self._solution = solution
        self._source = source
        self._loudspeakers = loudspeakers
        self._GF = green_function
        self._regions_interest = regions_interest
        self._regions_mask = dict()
        self._u = None
        self._region = None
        self._parse()
        self._initialize(n_points, r)

    @property
    def solution(self):
        return self._solution

    @solution.setter
    def solution(self, value):
        self._solution = value
        self._construct_u()

    def _construct_u(self):
        if self._GF.form == 'HRTF':
            self._u = {s: np.asarray([self._loudspeakers.G[self._region.name][s][k] @ self._solution[k]
                                      for k in range(self._source.n_freq)]) for s in SIDES}
        else:
            self._u = np.asarray([self._loudspeakers.G[self._region.name][k] @ self._solution[k]
                                  for k in range(self._source.n_freq)])

    def _parse(self):
        if self._solution is not None:
            if not isinstance(self._solution, np.ndarray) \
                    or self._solution.shape != (self._source.n_freq, self._loudspeakers.n_loud):
                raise TypeError(
                    '"solution" must be a numpy array object with dimensions (n_frequencies, n_loudspeakers).')

        if not isinstance(self._loudspeakers, Loudspeakers):
            raise TypeError('"loudspeakers" must be a Loudspeakers object.')

        if not isinstance(self._source, Source):
            raise TypeError('"source" must be a Source object.')

        if not isinstance(self._regions_interest, Iterable):
            self._regions_interest = [self._regions_interest]

        if any(not isinstance(x, RegionOfInterest) for x in self._regions_interest):
            raise TypeError('"regions_interest" must be an RegionOfInterest object, or a list of them.')

        if self._GF is None:
            self._GF = Monopole(self._source.frq)

    def _initialize_region(self, n_points, r):
        # First, we define the complete region.
        self._region = FORM[self._loudspeakers.form](n_points=n_points,
                                                     diameter=r * self._loudspeakers.diam,
                                                     ratio=0.999 / r,
                                                     shift=(1 - r) / 2 * np.ones(2) * self._loudspeakers.diam,
                                                     region_type='complete')
        self._region.initialize()

        # Second, we define the regions of interest mask.
        for region in self._regions_interest:
            self._regions_mask[region.name] = region.construct_form(self._region.points)

        # Third, we set the points where u0 is 0 (recycling the atribute idx_hole).
        self._region.idx_hole = (np.zeros(self._region.n_mesh) + sum((self._regions_mask[region.name]
                                                                      for region in self._regions_interest if
                                                                      region.type == 'silence'))).astype(bool)

    def _initialize_waves(self):
        # Fourth, we initialize u0, T weights (source) and G (loudspeakers) and "u" over the regions of interest.
        self._source.initialize(self._region)
        self._loudspeakers.initialize(self._region, self._source.frq)
        if self._solution is not None:
            self._construct_u()

    def _initialize(self, n_points, r):
        self._initialize_region(n_points, r)
        self._initialize_waves()

    def u0(self, mask=True, regions=True, dB=False, name=None):
        if self._GF.form == 'HRTF':
            u0s = [self.reshape_3d(self._source.u0[self._region.name][s]) for s in SIDES]
        else:
            u0s = [self.reshape_3d(self._source.u0[self._region.name])]
        max = np.nanpercentile(np.real(u0s), 99)
        for u0 in u0s:
            if dB:
                u0 = decibels(np.abs(u0))
            else:
                u0 = np.real(u0)
            if mask:
                u0 = np.ma.masked_array(u0, mask=np.invert(self._region.mask))
            for k in range(self._source.n_freq):
                fig, ax = plt.subplots(figsize=FIG_SIZE)
                ax.set_aspect('equal')
                if regions:
                    self.include_regions_contours()
                im = plt.imshow(rotation_45(u0[k]),
                                cmap=plt.cm.RdBu_r,
                                vmin=-max,
                                vmax=max,
                                extent=self._region.extent,
                                interpolation='spline16')
                self.adjust_layout(im)
                # plt.savefig(fname=f'{name}_u0_{self._source.frq[k]}', dpi=300)
                plt.show()
        return u0s

    def u(self, mask=True, regions=True, loudspeakers=True, dB=False, name=None):
        if self._GF.form == 'HRTF':
            u0s = [copy(self._source.u0[self._region.name][s]) for s in SIDES]
            us = [self.reshape_3d(self._u[s]) for s in SIDES]
        else:
            u0s = [copy(self._source.u0[self._region.name])]
            us = [self.reshape_3d(self._u)]
        max = np.nanpercentile(np.real(u0s), 99)

        for i, (u, u0) in enumerate(zip(us, u0s)):
            if dB:
                u = decibels(np.abs(u))
                u0 = decibels(np.abs(u0))
            else:
                u = np.real(u)
                u0 = np.real(u0)
            if mask:
                u = np.ma.masked_array(u, mask=np.invert(self._region.mask))
                u0 *= self._region.idx_form
            for k in range(self._source.n_freq):
                fig, ax = plt.subplots(figsize=FIG_SIZE)
                ax.set_aspect('equal')
                if loudspeakers:
                    self.include_loudspeakers(k)
                if regions:
                    self.include_regions_contours()
                im = plt.imshow(rotation_45(u[k]),
                                cmap=plt.cm.RdBu_r,
                                vmin=-max,
                                vmax=max,
                                extent=self._region.extent,
                                interpolation='spline16')
                self.adjust_layout(im)
                if name is None:
                    name = f'u_{self._source.frq[k]}_{i}'
                # plt.savefig(name + f'_{i}', dpi=300)
                plt.show()
        return us

    def u0_u(self, mask=True, regions=True, loudspeakers=True, dB=False):
        if self._GF.form == 'HRTF':
            u0s = np.asarray([self.reshape_3d(np.real(self._source.u0[self._region.name][s])) for s in SIDES])
            us = np.asarray([self.reshape_3d(np.real(self._u[s])) for s in SIDES])
        else:
            u0s = np.asarray([self.reshape_3d(np.real(self._source.u0[self._region.name]))])
            us = np.asarray([self.reshape_3d(np.real(self._u))])

        difs = u0s - us
        for u0, dif in zip(u0s, difs):
            if dB:
                dif = decibels(np.real(dif))
                u0 = decibels(np.real(u0))
            if mask:
                dif *= self._region.mask
                u0 *= self._region.idx_form
            for k in range(self._source.n_freq):
                fig, ax = plt.subplots(figsize=FIG_SIZE)
                ax.set_aspect('equal')
                if loudspeakers:
                    self.include_loudspeakers(k)
                if regions:
                    self.include_regions_contours()
                max = np.nanpercentile(u0[k], 99)
                im = plt.imshow(rotation_45(dif[k]),
                                cmap=plt.cm.RdBu_r,
                                vmin=-max,
                                vmax=max,
                                extent=self._region.extent,
                                interpolation='spline16')
                self.adjust_layout(im)
                # plt.savefig(fname=f'u0-u_{self._source.frq[k]}', dpi=300)
                plt.show()
        return difs

    def u_level(self, mask=True, show=True, regions=True, loudspeakers=True, constraint_type='knobel', discomfort=True,
                dB=True):
        constraint_types = {'knobel': knobel_model(),
                            'sherlock': sherlock_model()}
        threshold = constraint_types[constraint_type](self._source.frq) + SMALL_NUMBER
        if self._GF.form == 'HRTF':
            u_abs = np.nanmax([self.reshape_3d(np.abs(self._u[s])) for s in SIDES], axis=0)
        else:
            u_abs = self.reshape_3d(np.abs(self._u))
        if dB:
            threshold = decibels(threshold)
            u_abs = decibels(u_abs)
        normalization = self._region.n_mesh
        u_level = np.nansum(u_abs, axis=0)
        if mask:
            u_abs *= self._region.mask
            u_level *= self._region.mask
            normalization = np.sum(self._region.mask)
            if dB:
                u_abs += threshold * np.invert(self._region.mask)
        level = {'global': (np.nansum(u_level) / normalization, np.nanmax(u_level))}
        if regions:
            for region in self._regions_interest:
                u_level_region = u_level * self.reshape_2d(self._regions_mask[region.name])
                average = np.nansum(u_level_region) / np.sum(self._regions_mask[region.name])
                maximum = np.nanmax(u_level_region)
                level[region.name] = (average, maximum)
        if show:
            for name, (avg, vmax) in level.items():
                print(f'Optimal "u" level ({name}): - Average: {avg},\n{" " * (22 + len(name))}- Maximum: {vmax}.')
            for k in range(self._source.n_freq):
                fig, ax = plt.subplots(figsize=FIG_SIZE)
                ax.set_aspect('equal')
                if loudspeakers:
                    self.include_loudspeakers()
                if regions:
                    self.include_regions_contours()
                if discomfort:
                    ax.contour(rotation_inverse_45(u_abs[k]),
                               levels=[threshold[k]],
                               extent=self._region.extent)
                vmin = np.nanpercentile(np.ravel(u_abs[k]), 1)
                vmax = np.nanpercentile(np.ravel(u_abs[k]), 99)
                divnorm = colors.TwoSlopeNorm(vmin=vmin, vmax=vmax, vcenter=threshold[k])
                im = plt.imshow(rotation_45(u_abs[k]),
                                cmap=plt.cm.RdBu_r,
                                norm=divnorm,
                                extent=self._region.extent,
                                interpolation='spline16')
                self.adjust_layout(im)
                # plt.savefig(fname=f'u_level', dpi=300)
                plt.show()
        return level

    def sweet_spot(self, mask=True, show=True, regions=True, loudspeakers=True, phase=True):
        if self._GF.form == 'HRTF':
            u0 = {s: self._source.u0[self._region.name][s] if phase is True else np.abs(
                self._source.u0[self._region.name][s]) for s in SIDES}
            u = {s: self._u[s] if phase is True else np.abs(self._u[s]) for s in SIDES}
            T_w = self._source.T_weights[self._region.name]
            psy_error = np.nansum(np.maximum(*[square(u[s] - u0[s]) * T_w[s] for s in SIDES]), axis=0) - 1
        else:
            u0 = self._source.u0[self._region.name] if phase is True else np.abs(self._source.u0[self._region.name])
            u = self._u if phase is True else np.abs(self._u)
            T_w = self._source.T_weights[self._region.name]
            psy_error = np.nansum(square(u - u0) * T_w, axis=0) - 1

        sweet_spot = self.reshape_2d(np.asarray(psy_error <= SMALL_NUMBER, dtype=float))
        normalization = np.sum(self._region.n_mesh)
        if mask:
            sweet_spot *= self._region.mask
            normalization = np.sum(self._region.mask)
        sweet_spot_proportions = {'global': np.nansum(sweet_spot) / normalization}
        if regions:
            for region in self._regions_interest:
                sweet_spot_region = np.nansum(sweet_spot * self.reshape_2d(self._regions_mask[region.name])) / \
                                    np.sum(self._regions_mask[region.name])
                sweet_spot_proportions[region.name] = sweet_spot_region
        if show:
            for name, proportion in sweet_spot_proportions.items():
                print(f'Sweet Spot proportion ({name}): {proportion}')
            fig, ax = plt.subplots(figsize=FIG_SIZE)
            ax.set_aspect('equal')
            if loudspeakers:
                self.include_loudspeakers()
            if regions:
                self.include_regions_contours()
            im = plt.imshow(rotation_45(sweet_spot),
                            cmap=plt.cm.binary,
                            vmin=0,
                            vmax=1,
                            extent=self._region.extent,
                            interpolation='spline16')
            self.adjust_layout(im, correction=True, n_max=2)
            # plt.savefig(fname=f'sweet_spot_phase={phase}', dpi=300)
            plt.show()
        return sweet_spot_proportions

    def discomfort_spot(self, mask=True, show=True, regions=True, loudspeakers=True, constraint_type='knobel'):
        constraint_types = {'knobel': knobel_model(),
                            'sherlock': sherlock_model()}
        threshold = constraint_types[constraint_type](self._source.frq)
        if self._GF.form == 'HRTF':
            discomfort = np.maximum(*[np.nansum(np.abs(self._u[s]).T / threshold, axis=1) for s in SIDES])
        else:
            discomfort = np.nansum(np.abs(self._u).T / threshold, axis=1)
        discomfort_spot = self.reshape_2d(discomfort >= 1)
        normalization = self._region.n_mesh
        if mask:
            discomfort_spot *= self._region.mask
            normalization = np.sum(self._region.mask)
        discomfort_spot_proportions = {'global': np.nansum(discomfort_spot) / normalization}
        if regions:
            for region in self._regions_interest:
                discomfort_region = np.nansum(discomfort_spot * self.reshape_2d(self._regions_mask[region.name])) / \
                                    np.sum(self._regions_mask[region.name])
                discomfort_spot_proportions[region.name] = discomfort_region
        if show:
            for name, proportion in discomfort_spot_proportions.items():
                print(f'Discomfort Spot proportion ({name}): {proportion}')
            fig, ax = plt.subplots(figsize=FIG_SIZE)
            ax.set_aspect('equal')
            if loudspeakers:
                self.include_loudspeakers()
            if regions:
                self.include_regions_contours()
            im = plt.imshow(rotation_45(discomfort_spot),
                            cmap=plt.cm.binary,
                            vmin=0,
                            vmax=1,
                            extent=self._region.extent,
                            interpolation='spline16')
            self.adjust_layout(im, correction=True, n_max=2)
            # plt.savefig(fname=f'discomfort_spot', dpi=300)
            plt.show()
        return discomfort_spot_proportions

    def physical_error(self, mask=True, show=True, regions=True, loudspeakers=True, dB=False):
        if self._GF.form == 'HRTF':
            u0 = {s: self.reshape_3d(np.real(self._source.u0[self._region.name][s])) for s in SIDES}
            u = {s: self.reshape_3d(np.real(self._u[s])) for s in SIDES}
            dif = np.maximum(*[np.nansum(np.square(np.abs(u0[s] - u[s])), axis=0) for s in SIDES])
        else:
            u0 = self.reshape_3d(np.real(self._source.u0[self._region.name]))
            u = self.reshape_3d(np.real(self._u))
            dif = np.nansum(np.square(np.abs(u0 - u)), axis=0)
        normalization = self._region.n_mesh
        if mask:
            dif *= self._region.mask
            u0 *= self._region.mask
            normalization = np.sum(self._region.mask)
        errors = {'global': (np.nansum(dif) / normalization, np.nanmax(dif))}
        if regions:
            for region in self._regions_interest:
                dif_r = dif * self.reshape_2d(self._regions_mask[region.name])
                errors[region.name] = (np.nansum(dif_r) / np.sum(self._regions_mask[region.name]), np.nanmax(dif_r))
        if show:
            if dB:
                dif = np.nan_to_num(decibels(dif), nan=0, neginf=0)
            for name, (avg, max) in errors.items():
                print(f'Physical error ({name}): - ||.||^2 in time, Average in space: {avg},\n'
                      f'{" " * (19 + len(name))}- ||.||^2 in time, Maximum in space: {max}.')
            fig, ax = plt.subplots(figsize=FIG_SIZE)
            ax.set_aspect('equal')
            if loudspeakers:
                self.include_loudspeakers()
            if regions:
                self.include_regions_contours()
            max = np.nanpercentile(dif.ravel(), 99)
            im = plt.imshow(rotation_45(dif),
                            cmap=plt.cm.RdBu_r,
                            vmin=-max,
                            vmax=max,
                            extent=self._region.extent,
                            interpolation='spline16')
            self.adjust_layout(im, correction=True, n_max=1 + len(str(max)))
            # plt.savefig(fname=f'physical_error', dpi=300)
            plt.show()
        return errors

    def psychoacoustic_error(self, mask=True, show=True, regions=True, loudspeakers=True, sweet=True, phase=True):
        if self._GF.form == 'HRTF':
            u0 = {s: self._source.u0[self._region.name][s] if phase is True else np.abs(
                self._source.u0[self._region.name][s]) for s in SIDES}
            u = {s: self._u[s] if phase is True else np.abs(self._u[s]) for s in SIDES}
            T_w = {s: self._source.T_weights[self._region.name][s] for s in SIDES}
            psy_error = self.reshape_2d(np.nan_to_num(
                np.log(np.nansum(np.maximum(*[square(u[s] - u0[s]) * T_w[s] for s in SIDES]), axis=0)), nan=0,
                neginf=0))
        else:
            u0 = self._source.u0[self._region.name] if phase is True else np.abs(self._source.u0[self._region.name])
            u = self._u if phase is True else np.abs(self._u)
            T_w = self._source.T_weights[self._region.name]

            psy_error = self.reshape_2d(np.nan_to_num(np.log(np.nansum(square(u - u0) * T_w, axis=0)),
                                                      nan=0, neginf=0))

        normalization = self._region.n_mesh
        if mask:
            psy_error *= self._region.mask
            normalization = np.sum(self._region.mask)
        errors = {'global': (np.nansum(psy_error) / normalization, np.nanmax(psy_error.ravel()))}
        if regions:
            for region in self._regions_interest:
                psy_error_region = psy_error * self.reshape_2d(self._regions_mask[region.name])
                average = np.nansum(psy_error_region) / np.sum(self._regions_mask[region.name])
                maximum = np.nanmax(psy_error_region.ravel())
                errors[region.name] = (average, maximum)
        if show:
            for name, (avg, max) in errors.items():
                print(f'Phycho-acoustic error ({name}): - Average: {avg},\n{" " * (26 + len(name))}- Maximum: {max}.')
            fig, ax = plt.subplots(figsize=FIG_SIZE)
            ax.set_aspect('equal')
            if loudspeakers:
                self.include_loudspeakers()
            if regions:
                self.include_regions_contours()
            if sweet:
                ax.contour(rotation_inverse_45(psy_error),
                           levels=[0],
                           extent=self._region.extent)
            max = np.nanpercentile(np.ravel(psy_error), 99.9)
            min = np.nanpercentile(np.ravel(psy_error), 0.1)

            divnorm = colors.TwoSlopeNorm(vmin=min, vmax=max, vcenter=0)
            im = ax.imshow(rotation_45(psy_error),
                           cmap=plt.cm.RdBu_r,
                           norm=divnorm,
                           extent=self._region.extent,
                           interpolation='spline16')
            self.adjust_layout(im, correction=True, n_max=np.max((len(str(max)), len(str(min)))))
            plt.tight_layout()
            # plt.savefig(fname=f'psycho_acoustic_error', dpi=300)
            plt.show()
        return errors

    def compute_IPD(self, array):
        return np.angle(array['left'] * np.conj(array['right']))
        
    def compute_ITD(self, array):
        return (self.compute_IPD(array).T / (-2 * np.pi * self._source.frq)).T

    def compute_ILD(self, array):
        return 20 * np.log(np.abs(array['left']) / np.abs(array['right']))

    def interaural_difference(self, mask=True, regions=True, loudspeakers=True, object_name='u', interaural_name='ITD'):
        if self._GF.form != 'HRTF':
            raise TypeError('Green function must be of the form HRTF to evaluate interaural differences.')
        object_map = {'u': self._u,
                      'u0': self._source.u0[self._region.name]}
        interaural_map = {'IPD': self.compute_IPD,
                          'ITD': self.compute_ITD,
                          'ILD': self.compute_ILD}
        interaural_differences = self.reshape_3d(interaural_map[interaural_name](object_map[object_name]))

        for interaural_difference in interaural_differences:
            if mask:
                interaural_difference *= self._region.mask
            fig, ax = plt.subplots(figsize=FIG_SIZE)
            ax.set_aspect('equal')
            if loudspeakers:
                self.include_loudspeakers()
            if regions:
                self.include_regions_contours()
            im = plt.imshow(rotation_45(interaural_difference),
                            cmap=plt.cm.RdBu_r,
                            extent=self._region.extent,
                            interpolation='spline16')
            self.adjust_layout(im)

    def intensity_direction_error(self, mask=True, show=True, regions=True, loudspeakers=True):
        if not isinstance(self._GF, Monopole):
            raise TypeError('IDE only availiable for Monopole Green functions.')
        u0 = self._source.u0[self._region.name]
        keys = ['global'] + [region.name for region in self._regions_interest] if regions else ['global']
        errors = dict((key, np.empty((self._source.n_freq, 2))) for key in keys)
        for k in range(self._source.n_freq):
            u0_vel = np.sum(
                [self._source.int[i][k] * g3dv(self._region.points, self._source.pos[i], self._source.frq[k])
                 for i in range(self._source.n_sources)], axis=0)
            u_vel = np.sum(
                [self._solution[k][i] * g3dv(self._region.points, self._loudspeakers.pos[i], self._source.frq[k])
                 for i in range(self._loudspeakers.n_loud)], axis=0)
            u0_id = normalized(1 / 2 * np.real(u0[k] * np.conj(u0_vel))).T
            u_id = normalized(1 / 2 * np.real(self._u[k] * np.conj(u_vel))).T
            id_error = self.reshape_2d(np.arccos(np.einsum('ij,ij->i', u0_id, u_id).round(10)) / np.pi)
            normalization = self._region.n_mesh
            if mask:
                id_error *= self._region.mask
                normalization = np.sum(self._region.mask)
            errors['global'][k] = (np.nansum(id_error) / normalization, np.nanmax(id_error))
            if regions:
                for region in self._regions_interest:
                    id_region = id_error * self.reshape_2d(self._regions_mask[region.name])
                    errors[region.name][k] = (np.nansum(id_region) / np.sum(self._regions_mask[region.name]),
                                              np.nanmax(id_region))
            if show:
                for name, array in errors.items():
                    print(f'Intensity direction error ({self._source.frq[k]} Hz, {name}): '
                          f'- Average: {array[k][0]},\n'
                          f'{" " * (35 + len(str(self._source.frq[k])) + len(name))}'
                          f'- Maximum: {array[k][1]}.')
                fig, ax = plt.subplots(figsize=FIG_SIZE)
                ax.set_aspect('equal')
                if loudspeakers:
                    self.include_loudspeakers(k)
                if regions:
                    self.include_regions_contours()
                im = plt.imshow(rotation_45(id_error),
                                cmap=plt.cm.bone_r,
                                vmax=1,
                                vmin=0,
                                extent=self._region.extent,
                                interpolation='spline16')
                # plt.title(f'Intensity Direction Error ({self._source.frq[k]} Hz)')
                self.adjust_layout(im, correction=True, n_max=2)
                # plt.savefig(fname=f'intensity_direction_error_{self._source.frq[k]}', dpi=300)
                plt.show()
        return errors

    def external_plot(self, array, array_reference=None, plot_type=None, vmax=None, vmin=None, vcenter=None, mask=True,
                      regions=True, loudspeakers=True, name=None, position_reference=None, halfspace=False):
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        ax.set_aspect('equal')
        # if max:
        #     array = np.nan_to_num(array, nan=max)
        # if mask:
        #     array *= self._region.mask
        nans_array = np.isnan(array.ravel())
        if plot_type == 'vector':
            radians_array = ((array + 90) / 180 * np.pi)
            cartesi_error = np.abs((array - array_reference + 180) % 360 - 180)
            start_x, end_x, start_y, end_y = self._region.extent
            n_side = 14
            mesh_x, mesh_y = np.mgrid[start_x:end_x:n_side * 1j, start_y:end_y:n_side * 1j]
            points = np.vstack([mesh_x.ravel(), mesh_y.ravel()]).T
            if halfspace:
                upper_part = (points[:, 1] >= 2.5 * 4 / 3).reshape(n_side, n_side)
                cartesi_error[upper_part] = np.nan
            nans_array = np.isnan(cartesi_error.ravel())
            if position_reference is not None:
                dif = position_reference - points
                thetas_reference = (np.arctan2(dif[:, 1], dif[:, 0]) - np.pi / 2).reshape(n_side, n_side)
                radians_array += thetas_reference
            mask = np.invert(self._region.construct_form(points).reshape(n_side, n_side))
            angle_x = np.ma.masked_array(np.cos(radians_array), mask=mask)
            angle_y = np.ma.masked_array(np.sin(radians_array), mask=mask)
            if vmax:
                color_mask = cartesi_error <= vmax
                color_array = cartesi_error * color_mask + np.invert(color_mask) * vmax
            divnorm = colors.TwoSlopeNorm(vmin=0, vmax=vmax, vcenter=5)
            cmap = plt.cm.coolwarm
            im = plt.quiver(mesh_x,
                            mesh_y,
                            angle_x,
                            angle_y,
                            color_array,
                            edgecolor='black',
                            norm=divnorm,
                            cmap=cmap,
                            width= 1.5 * 0.005,
                            headwidth=2 * 2.15,
                            headlength=1.5 * 2.15,
                            scale=35 / 2.5,
                            pivot='middle')
            nan_points = points[nans_array * self._region.construct_form(points)]
            dark_maroon = np.asarray([[.4, .0549, 0.05, 1] for i in range(len(nan_points))])
            red = np.asarray([[.6, .0549, 0.05, 1] for i in range(len(nan_points))])
            plt.scatter(nan_points[:, 0], nan_points[:, 1], c=red, s=5)
            if position_reference is not None:
                plt.scatter(position_reference[0], position_reference[1],
                            s=decibels(self._source.int) / 2,
                            facecolors='none',
                            edgecolors='black')
            array = np.ma.masked_array(5 * np.ones((self._region.n_m2, self._region.n_m2)),
                                     mask=np.invert(self._region.mask))
            im2 = ax.imshow(array,
                            cmap=plt.cm.RdBu_r,
                            norm=divnorm,
                            extent=self._region.extent,
                            interpolation='spline16')
            sweet_spot = np.asarray(cartesi_error <= 5) * np.invert(mask)
            sweet_spot_size = np.nansum(sweet_spot) / np.sum(self._region.construct_form(points))
            print(f'Sweet spot {name}: {sweet_spot_size}')

        elif plot_type == 'positive_angular':
            radians_array = ((array + 90) / 180 * np.pi)
            cartesi_error = np.abs((array - array_reference + 180) % 360 - 180)
            upper_part = self.reshape_2d(self._region.points[:, 1] <= 2.5 * 4 / 3)
            #cartesi_error = np.nan_to_num(cartesi_error, nan=max + 1) * upper_part + np.invert(upper_part) * max + 1
            color_mask = cartesi_error <= max
            cartesi_error = cartesi_error * color_mask + np.asarray(np.invert(color_mask), dtype=float) * max
            divnorm = colors.TwoSlopeNorm(vmin=0, vmax=max, vcenter=5)
            cartesi_error = np.ma.masked_array(cartesi_error, mask=np.invert(self._region.mask))
            im = plt.imshow(rotation_45(cartesi_error),
                            cmap=plt.cm.RdBu_r,
                            norm=divnorm,
                            extent=self._region.extent,
                            interpolation='spline16')
            # ax.contour(rotation_inverse_45(cartesi_error),
            #            levels=[5],
            #            extent=self._region.extent)
        elif plot_type == 'positive':
            divnorm = colors.TwoSlopeNorm(vmin=vmin, vmax=vmax, vcenter=vcenter)
            array = np.ma.masked_array(array, mask=np.invert(self._region.mask))
            im = plt.imshow(rotation_45(array),
                            cmap=plt.cm.RdBu_r,
                            norm=divnorm,
                            extent=self._region.extent,
                            interpolation='nearest')
            sweet_spot = np.asarray(array <= vcenter) * self._region.mask
            sweet_spot_size = np.nansum(sweet_spot) / np.sum(self._region.mask)
            print(f'Sweet spot {name}: {sweet_spot_size}')
            # ax.contour(rotation_inverse_45(cartesi_error),
            #            levels=[5],
            #            extent=self._region.extent)
        else:
            cmap = plt.cm.RdBu_r
            start_x, end_x, start_y, end_y = self._region.extent
            mesh_x, mesh_y = np.mgrid[start_x:end_x:33 * 1j, start_y:end_y:33 * 1j]
            points = np.vstack([mesh_x.ravel(), mesh_y.ravel()]).T
            mask = np.invert(self._region.construct_form(points).reshape(33, 33))
            array = np.ma.masked_array(array, mask=mask)
            divnorm = colors.TwoSlopeNorm(vmin=min, vmax=max, vcenter=center)
            im = ax.imshow(rotation_45(array),
                           cmap=cmap,
                           norm=divnorm,
                           extent=self._region.extent,
                           interpolation='spline16')
            sweet_spot = np.asarray(array <= center) * np.invert(mask)
            sweet_spot_size = np.nansum(sweet_spot) / np.sum(self._region.construct_form(points))
            print(f'Sweet spot {name}: {sweet_spot_size}')
        if loudspeakers:
            self.include_loudspeakers()
        if regions:
            self.include_regions_contours()
        # if name is not None:
        #     plt.title(name)
        self.adjust_layout(im)
        # if plot_type == 'vector':
        #     plt.savefig(fname=f'localization/{name}', dpi=300)
        # else:
        #     plt.savefig(fname=f'coloration/{name}', dpi=300)
        plt.show()
        return sweet_spot_size

    def include_loudspeakers(self, k=None):
        if k is not None:
            solution = np.abs(self._solution[k])
        else:
            solution = np.sum(np.abs(self._solution), axis=0)
        colors = np.ravel(solution / np.nanmax(solution))
        self._loudspeakers.plot_design(colors=colors)

    def include_regions_contours(self):
        for region in self._regions_interest:
            region.plot_design()

    def reshape_3d(self, array):
        return array.reshape((self._source.n_freq, self._region.n_m2, self._region.n_m2))

    def reshape_2d(self, array):
        return array.reshape((self._region.n_m2, self._region.n_m2))

    def adjust_layout(self, im):
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", "5%", pad=f'2%')
        plt.colorbar(im, cax=cax, format=None)
        plt.tight_layout()
