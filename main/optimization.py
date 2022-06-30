import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import warnings
from abc import ABCMeta, abstractmethod
from time import time
from collections.abc import Iterable
from utils_general import square, BIG_NUMBER, SMALL_NUMBER, name_counter, SIDES, row_wise_norm2
from utils_psychoacousticmodels import knobel_model, sherlock_model
from loudspeakers import Loudspeakers
from regions_interest import RegionOfInterest
from source import Source
from green_function import HRTFScattered, MonopoleScattered

SOLVERS = [cp.MOSEK, cp.ECOS, cp.SCS, cp.CPLEX, cp.CVXOPT]


class Optimization(metaclass=ABCMeta):
    def __init__(self, loudspeakers, source, regions_interest, truncation, penalty_norm, lamb):
        self.loudspeakers = loudspeakers
        self.regions_interest = regions_interest
        self.source = source
        self.penalty_norm = penalty_norm
        self.lamb = lamb
        self.truncation = truncation
        self.variable = None
        self.parse()

    def parse(self, c=343):
        # First, we check the correctness of the input types.
        if self.penalty_norm not in {1, 2, 'inf'}:
            if self.penalty_norm is not None:
                warnings.warn('The value for "penalty" must be None, 1, 2 or "inf". None will be used as default')
            self.penalty_norm = 0

        if type(self.lamb) not in {float, int} or self.lamb < 0:
            if self.lamb is not None:
                warnings.warn('The value for "lamb" must be a non-negative number. 0 will be used as default.')
            self.lamb = 0

        if type(self.truncation) not in {float, int} or self.truncation < 0:
            if self.truncation is not None:
                warnings.warn('The value for "truncation" must be a non-negative number. 0 will be used as default.')
            self.truncation = 0

        if not isinstance(self.loudspeakers, Loudspeakers):
            raise TypeError('"loudspeakers" must be a Loudspeakers object.')

        if not isinstance(self.source, Source):
            raise TypeError('"source" must be a Source object.')

        if not isinstance(self.regions_interest, Iterable):
            self.regions_interest = [self.regions_interest]

        if any(not isinstance(x, RegionOfInterest) for x in self.regions_interest):
            raise TypeError('"regions_interest" must be an RegionOfInterest object, or a list of them.')

        # Second, we differentiate the names and colors of the regions.
        if len(self.regions_interest) == 1:
            self.regions_interest[0].name = 'omega'
        else:
            possible_region_types = {'sound', 'silence'}
            counter = dict((x, name_counter(x)) for x in possible_region_types)
            for region_type in possible_region_types:
                if len(type_list := [region for region in self.regions_interest if region.type == region_type]) > 1:
                    for region in type_list:
                        region.name, region.color = next(counter[region_type])

        # Third, we adjust the dimensions of the virtual room asociated to the loudspeakers and the interest regions.
        # Also, we control if the size of the discretization mesh is sufficent to avoid spatial aliasing.
        for region in self.regions_interest:
            if not region.diam == self.loudspeakers.diam:
                raise TypeError(f'Dimension mismatch of the virtual room of the loudspeakers and the interest regions.')
            if self.loudspeakers.form in {'circle', 'octagon'} and region.form in {'square', 'rectangle'}:
                region.ratio *= 1 / np.sqrt(2)
            elif self.loudspeakers.form == 'circle' and region.form == 'octagon':
                region.ratio *= (2 + np.sqrt(2)) / np.sqrt((2 + np.sqrt(2)) ** 2 + 2)
            region.initialize()
            if region.n_points < (minimo := region.area * 2 * (2 * max(self.source.frq) / c) ** 2):
                warnings.warn(f'The size of the mesh of {region.name} is insufficent to avoid spatial aliasing.'
                              f' It should be at least {np.ceil(minimo)}')

        # Fourth, we initialize u0, T_weights (source) and G (loudspeakers) over the regions of interest.
        self.source.initialize(self.regions_interest)
        self.loudspeakers.initialize(self.regions_interest, self.source.frq)
        self.variable = cp.Variable((self.source.n_freq, self.loudspeakers.n_loud), complex=True)
        self.variable.value = np.zeros((self.source.n_freq, self.loudspeakers.n_loud))

    @abstractmethod
    def minimize(self, *args):
        pass

    def minimization_cycle(self, arg1, arg2, arg3, constraint_type):
        p_objective = cp.Minimize(self.objective_function(arg1, arg2, arg3))
        p_constraints = self.constraints(constraint_type)
        p_problem = cp.Problem(p_objective, p_constraints)
        for i in range(len(SOLVERS)):
            try:
                optimum = p_problem.solve(solver=SOLVERS[i], warm_start=True)
                if self.variable.value is None:
                    continue
                return optimum, i == 0

            except cp.error.SolverError:
                if i < len(SOLVERS) - 1:
                    print(f'Warning: Solver "{SOLVERS[i]}" failed. Solver "{SOLVERS[i + 1]}" will be used instead.')
                else:
                    print('All the solvers failed. The solution might be very inaccurate.')
        return -1

    def objective_function(self, arg1, arg2, arg3):
        return self.lamb * self.objective_penalty() + self.objective_error(arg1, arg2, arg3)

    def objective_penalty(self):
        return cp.norm(self.variable, self.penalty_norm)

    @abstractmethod
    def objective_error(self, arg1, arg2, arg3):
        pass

    def constraints(self, constraint_type):
        if constraint_type is None:
            return []
        constraint_types = {'knobel': knobel_model(),
                            'sherlock': sherlock_model()}
        threshold = constraint_types[constraint_type](self.source.frq)
        if self.loudspeakers.GF.form == 'HRTF':
            constraints = [cp.sum(cp.vstack(
                           [cp.abs(self.loudspeakers.G[region.name][side][k] @ self.variable[k]) / threshold[k]
                            for k in range(self.source.n_freq)]), axis=0) <= 1
                           for region in self.regions_interest for side in SIDES]
        else:
            constraints = [
                cp.sum(cp.vstack([cp.abs(self.loudspeakers.G[region.name][k] @ self.variable[k]) / threshold[k]
                                  for k in range(self.source.n_freq)]), axis=0) <= 1
                for region in self.regions_interest]
        return constraints

    def plot_instance(self):
        fig, ax = plt.subplots()
        p_loudspeakers = self.loudspeakers.plot()
        p_regions_interest = [region.plot() for region in self.regions_interest]
        p_source = self.source.plot()
        ax.set_aspect('equal')

        c = 0
        n = len(self.regions_interest) - 1
        fontsize = 10
        y_max = max(np.max(self.source.pos[:, 1]) + 0.15, self.loudspeakers.diam + 0.62 * self.loudspeakers.size)
        y_min = min(np.min(self.source.pos[:, 1]), - 0.62 * self.loudspeakers.size)
        y_dif = abs(y_max - y_min)
        # print((y_min - fontsize / 7.5 * (1 / 6 + n / 12 + c) * abs(y_max - y_min), (1 + c) * y_max))
        plt.ylim((y_min - (fontsize / 6 * (1 / 6 + n / 12) + c) * y_dif, (1 + c) * y_max))
        # plt.ylim((-4.3, 7.741935483870968))

        x_max = max(np.max(self.source.pos[:, 0]), self.loudspeakers.diam + 0.62 * self.loudspeakers.size)
        x_min = min(np.min(self.source.pos[:, 0]), - 0.62 * self.loudspeakers.size)
        x_dif = abs(x_max - x_min)
        # print((x_min - c * x_dif, (1 + c) * x_max))
        plt.xlim((x_min - c * x_dif, (1 + c) * x_max))
        # plt.xlim((-0.6796531104229611, 5.663945147155012))
        plt.xlim((x_min - c * x_dif, x_max + c * x_dif))
        plt.ylim((y_min - c * y_dif, y_max + c * y_dif))
        plt.xlim((-0.5166666666666667, 5.516666666666667))
        plt.ylim((-0.5166666666666667, 7.65))
        # plt.xlim((-0.48694686130641796, 5.486946861306418))
        # plt.ylim((-0.48694686130641796, 7.65))
        scatterers = []
        if isinstance(self.loudspeakers.GF, HRTFScattered) or isinstance(self.loudspeakers.GF, MonopoleScattered):
            scatterers.append(plt.scatter(self.loudspeakers.GF.pos[:, 0], self.loudspeakers.GF.pos[:, 1],
                                     facecolors='darkblue',
                                     edgecolors='darkblue'))
        ax.legend([p_source, p_loudspeakers] + p_regions_interest + scatterers,
                  ['Source', 'Speakers'] + [r.name.capitalize() for r in self.regions_interest] + ['Scatterers'],
                  scatterpoints=1, loc='upper right', ncol=1, fontsize=fontsize)  # 1.55, -0.023, #0.52, 1
        # plt.title('Instance')
        # ax.set_xticks([i for i in range(6)])
        plt.savefig(fname='instance.png', dpi=300, bbox_inches='tight')
        plt.show()


class SweetOptimization(Optimization):
    def __init__(self, loudspeakers, source, regions_interest, truncation=0, penalty_norm=1, lamb=10 ** (-6)):
        super().__init__(loudspeakers, source, regions_interest, truncation, penalty_norm, lamb)
        self.u = dict()
        self.error = dict()

    def parse_epsilon(self, epsilon_sequence):
        adaptative = False
        minimum = None
        if not isinstance(epsilon_sequence, Iterable):
            if type(epsilon_sequence) not in {int, float} or epsilon_sequence >= 100 or epsilon_sequence <= 0:
                raise TypeError('"epsilons" must be a sequence of  arbitrary numbers, or a single number in (0, 100).')
            adaptative = True
            epsilon = BIG_NUMBER
            n_epsilon = BIG_NUMBER
            minimum = max(int(sum((r.n_points for r in self.regions_interest)) * (1 - epsilon_sequence / 100) / 2), 1)
        else:
            epsilon = epsilon_sequence[0]
            n_epsilon = len(epsilon_sequence)
        return adaptative, epsilon, n_epsilon, minimum

    def initialize(self, initial_point_sweet_idx):
        for region in self.regions_interest:
            if initial_point_sweet_idx is not None:
                region.idx_sweet *= initial_point_sweet_idx[region.name]
            self.u[region.name] = np.zeros((self.source.n_freq, region.n_points))
            self.error[region.name] = np.zeros(region.n_points)

    def objective_error(self, epsilon, delta, squared):
        error = 0
        for r in self.regions_interest:
            r.idx_sweet *= self.error[r.name] <= epsilon
            if np.sum(r.idx_sweet) > 0:
                C = r.mu * r.area / r.n_points
                if self.source.n_freq == 1:
                    if self.loudspeakers.GF.form == 'HRTF':
                        u = {s: self.loudspeakers.G[r.name][s][0, r.idx_sweet] @ self.variable[0]
                             for s in SIDES}
                        u0 = {s: self.source.u0[r.name][s][0, r.idx_sweet] for s in SIDES}
                        T_w = {s: self.source.T_weights[r.name][s][0, r.idx_sweet] for s in SIDES}
                        T = cp.maximum(*[cp.multiply(cp.square(cp.abs(u[s] - u0[s])), T_w[s]) for s in SIDES])
                    else:
                        u = self.loudspeakers.G[r.name][0, r.idx_sweet] @ self.variable[0]
                        u0 = self.source.u0[r.name][0, r.idx_sweet]
                        T_w = self.source.T_weights[r.name][0, r.idx_sweet]
                        T = cp.multiply(cp.square(cp.abs(u - u0)), T_w)
                else:
                    if self.loudspeakers.GF.form == 'HRTF':
                        u = {s: cp.vstack([self.loudspeakers.G[r.name][s][k, r.idx_sweet] @ self.variable[k]
                                              for k in range(self.source.n_freq)]) for s in SIDES}
                        u0 = {s: self.source.u0[r.name][s][:, r.idx_sweet] for s in SIDES}
                        T_w = {s: self.source.T_weights[r.name][s][:, r.idx_sweet] for s in SIDES}
                        T = cp.maximum(*[cp.sum(cp.multiply(cp.square(cp.abs(u[s] - u0[s])), T_w[s]), axis=0)
                                         for s in SIDES])
                    else:
                        u = cp.vstack([self.loudspeakers.G[r.name][k, r.idx_sweet] @ self.variable[k]
                                       for k in range(self.source.n_freq)])
                        u0 = self.source.u0[r.name][:, r.idx_sweet]
                        T_w = self.source.T_weights[r.name][:, r.idx_sweet]
                        T = cp.sum(cp.multiply(cp.square(cp.abs(u - u0)), T_w), axis=0)
                error = error + C * cp.sum(cp.pos(T - 1 + delta * epsilon))
        return error

    def minimize(self, initial_point_sweet_idx=None, constraint_type='knobel', epsilon_sequence=99, delta=0,
                 squared=True, n_u=99, show=False):
        if show:
            t0 = time()
            inexact_list = []
        opt_list = []
        sweet_mu_list = []
        sweet_prop_list = []
        solution_list = []
        epsilon_list = [0]

        self.initialize(initial_point_sweet_idx)
        mu_omega = sum((r.mu * r.n_points for r in self.regions_interest))
        n_points = sum((r.n_points for r in self.regions_interest))
        adaptative, epsilon, n_epsilon, minimum = self.parse_epsilon(epsilon_sequence)
        ramp_convergence = False

        # Homotopy cycle.
        for i in range(n_epsilon):
            if show:
                print(f'Epsilon: {epsilon}.')

            # DCA cycle.
            for j in range(n_u):
                # Solve the cycle's convex problem.
                optimum, first_try_accuracy = self.minimization_cycle(epsilon, delta, squared, constraint_type)

                # Compute the error.
                for r in self.regions_interest:
                    if self.loudspeakers.GF.form == 'HRTF':
                        self.u[r.name] = {s: np.asarray([self.loudspeakers.G[r.name][s][k] @ self.variable.value[k]
                                                         for k in range(self.source.n_freq)]) for s in SIDES}
                        self.error[r.name] = np.sum(np.maximum(*[
                             square(self.u[r.name][s] - self.source.u0[r.name][s]) * self.source.T_weights[r.name][s]
                             for s in SIDES]), axis=0) - 1
                    else:
                        self.u[r.name] = np.asarray([self.loudspeakers.G[r.name][k] @ self.variable.value[k]
                                                     for k in range(self.source.n_freq)])
                        self.error[r.name] = np.sum(square(self.u[r.name] - self.source.u0[r.name])
                                                    * self.source.T_weights[r.name], axis=0) - 1

                # Save iteration information.
                sweet_spot_mu = sum(
                    (r.mu * np.sum(self.error[r.name] <= SMALL_NUMBER) for r in self.regions_interest)) / mu_omega
                sweet_spot_prop = sum(
                    (np.sum(self.error[r.name] <= SMALL_NUMBER) for r in self.regions_interest)) / n_points
                sweet_mu_list.append(sweet_spot_mu)
                sweet_prop_list.append(sweet_spot_prop)
                solution_list.append(self.variable.value)

                # Update ramp convergence criteria.
                ramp_qty = sum((np.sum((0 <= self.error[region.name][region.idx_sweet]) &
                                       (self.error[region.name][region.idx_sweet] <= epsilon))
                                for region in self.regions_interest))
                ramp_convergence = True if ramp_qty == 0 else False

                if show:
                    print(f'Epsilon: {epsilon}. DCA cycle: {j + 1}.\n'
                          f'  Optimum value: {optimum}.\n'
                          f'  Sweet Spot measure: {sweet_spot_mu}.\n'
                          f'  Sweet Spot proportion: {sweet_spot_prop}.\n'
                          f'  Ramp quantity: {ramp_qty}/{n_points}.\n'
                          f'  k-Omega size: {sum((np.sum(r.idx_sweet) for r in self.regions_interest))}/{n_points}')
                    if len(self.regions_interest) > 1:
                        for region in self.regions_interest:
                            print(f'  k-Omega size ({region.name}): {np.sum(region.idx_sweet)}/{region.n_points}')
                    if optimum == -1 or not first_try_accuracy:
                        inexact_list.append(epsilon_list[-1] + j)

                # Check DCA-convergence.
                opt_list.append(optimum)
                if j >= 1 and abs(opt_list[-1] - opt_list[-2]) <= 10 ** (-8) \
                        or ramp_convergence or epsilon == BIG_NUMBER:
                    if show:
                        epsilon_list.append(epsilon_list[-1] + j + 1)
                        print('DCA-convergence!\n')
                    break

            # Define 'epsilon' for next iteration.
            if adaptative:
                error = np.concatenate([self.error[r.name][r.idx_sweet] for r in self.regions_interest])
                sorted_error = np.sort(np.asarray([x for x in error if x < max(error)]))
                if len(sorted_error) > 0:
                    safe_minimum = min(len(sorted_error), minimum)
                    epsilon = max(min((sorted_error[-safe_minimum], np.percentile(sorted_error, epsilon_sequence))), 0)
                else:
                    ramp_convergence = True
            else:
                epsilon = epsilon_sequence[i]

            # Check epsilon convergence.
            if ramp_convergence:
                if show:
                    print('Epsilon-convergence!\n')
                break

        idx_max = max(range(len(solution_list)), key=lambda x: sweet_mu_list[x])
        best_solution = solution_list[idx_max]
        last_solution = solution_list[-1]
        if show:
            t1 = time()
            print(f'Best optimum: {sweet_mu_list[idx_max]}')
            print(f'Best discretized sweet spot measure: {sweet_mu_list[idx_max]}. '
                  f'Last discretized sweet spot measure: {sweet_mu_list[-1]}. '
                  f'Computation time: {t1 - t0}')
            self.plot_series(sweet_mu_list, [epsilon_list[1:-1], inexact_list], 'Sweet Spot measure series',
                             'lower right')
            self.plot_series(sweet_prop_list, [epsilon_list[1:-1], inexact_list], 'Sweet Spot proportion series',
                             'lower right')
            self.plot_series(opt_list, [epsilon_list[1:-1], inexact_list], 'Optimum series', 'upper right')

        return last_solution, best_solution

    def plot_series(self, series, events, name, localization):
        colors = ['darkgreen', 'red']
        names = ['New Epsilon Cycle', 'Inaccurate Solution']
        sizes = [50, 15]
        plots = []
        legends = []
        for idx, event_positions in enumerate(events):
            if len(event_positions) > 0:
                event_selection = np.asarray(series)[event_positions]
                plots.append(plt.scatter(event_positions,
                                         event_selection,
                                         s=sizes[idx],
                                         c=colors[idx]))
                legends.append(names[idx])
        plt.plot(series)
        plt.legend(plots,
                   legends,
                   scatterpoints=1,
                   loc=localization,
                   ncol=1,
                   fontsize=8)
        plt.title(name)
        plt.xlabel('Iteration')
        plt.show()


class LpqOptimization(Optimization):
    def __init__(self, loudspeakers, source, regions_interest, truncation=0, penalty_norm=2, lamb=0):
        super().__init__(loudspeakers, source, regions_interest, truncation, penalty_norm, lamb)
        # self.l_weights = {r.name: np.minimum(np.min([row_wise_norm2(r.points - x0) for x0 in self.loudspeakers.pos], axis=0), 1)
        #                   for r in self.regions_interest}
        l_distances = {r.name: np.min([row_wise_norm2(r.points - x0) for x0 in self.loudspeakers.pos], axis=0) for r in self.regions_interest}
        s_distances = {r.name: np.min([row_wise_norm2(r.points - x0) for x0 in self.source.pos], axis=0) for r in self.regions_interest}
        self.l_weights = {
            r.name: self.normalized(np.minimum(l_distances[r.name] / s_distances[r.name], s_distances[r.name] / l_distances[r.name]))
            for r in self.regions_interest}

    def normalized(self, array):
        return array / np.max(array)


    def objective_error(self, p, q, weighted):
        error = 0
        for r in self.regions_interest:
            # start_x, start_y, end_x, end_y = r.start()
            # mesh_x, mesh_y = np.mgrid[start_x:end_x:r.n_m2 * 1j, start_y:end_y:r.n_m2 * 1j]
            # points = np.vstack([mesh_x.ravel(), mesh_y.ravel()]).T
            # l_distances = np.min([row_wise_norm2(points - x0) for x0 in self.loudspeakers.pos], axis=0)
            # s_distances = np.min([row_wise_norm2(points - x0) for x0 in self.source.pos], axis=0)
            # exponent = 1
            # l_weights = np.minimum(l_distances / s_distances, s_distances / l_distances) ** exponent
            # array = np.ma.masked_array(np.flip(self.normalized(l_weights).reshape(r.n_m2, r.n_m2), axis=1).T, mask=np.invert(r.mask))
            # im = plt.imshow(array,
            #                 vmin=0,
            #                 vmax=1,
            #                 cmap=plt.cm.Blues,
            #                 extent=r.extent,
            #                 interpolation='nearest')
            # plt.colorbar(im)
            # plt.title('Spatial Weights')
            # plt.show()

            C = r.area / r.n_points
            if weighted == 'van de Par':
                if self.loudspeakers.GF.form == 'HRTF':
                    error += C * cp.maximum(*[cp.mixed_norm(cp.vstack([cp.multiply(
                                    self.loudspeakers.G[r.name][s][k] @ self.variable[k] - self.source.u0[r.name][s][k],
                                    self.source.T_weights[r.name][s][k]) for s in SIDES])
                                               for k in range(self.source.n_freq)), p, q])
                else:
                    error += C * cp.mixed_norm(cp.vstack(
                        [cp.multiply(self.loudspeakers.G[r.name][k] @ self.variable[k] - self.source.u0[r.name][k],
                                     self.source.T_weights[r.name][k]) for k in range(self.source.n_freq)]), p, q)

            elif weighted == 'loudspeakers':
                if self.loudspeakers.GF.form == 'HRTF':
                    error += C * cp.maximum(*[cp.mixed_norm(cp.multiply(cp.vstack([
                                    self.loudspeakers.G[r.name][s][k] @ self.variable[k] - self.source.u0[r.name][s][k]
                                    for s in SIDES]), self.l_weights[r.name]) for k in range(self.source.n_freq)), p, q])
                else:
                    error += C * cp.mixed_norm(cp.vstack(
                             [cp.multiply(self.loudspeakers.G[r.name][k] @ self.variable[k] - self.source.u0[r.name][k], self.l_weights[r.name])
                              for k in range(self.source.n_freq)]), p, q)
            else:
                if self.loudspeakers.GF.form == 'HRTF':
                    error += C * cp.maximum(*[cp.mixed_norm(cp.vstack(
                        [self.loudspeakers.G[r.name][s][k] @ self.variable[k] - self.source.u0[r.name][s][k]
                         for k in range(self.source.n_freq)]), p, q) for s in SIDES])
                else:
                    error += C * cp.mixed_norm(cp.vstack(
                        [self.loudspeakers.G[r.name][k] @ self.variable[k] - self.source.u0[r.name][k]
                         for k in range(self.source.n_freq)]), p, q)
        return error

    def minimize(self, p=2, q=2, weighted=False, constraint_type=None, show=False):
        if show:
            t0 = time()
        optimum = self.minimization_cycle(p, q, weighted, constraint_type)
        if show:
            t1 = time()
            print(f'Optimum value: {optimum}. Computation time: {t1 - t0}')

        return self.variable.value
