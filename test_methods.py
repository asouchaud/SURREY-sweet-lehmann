import sfs
import json
from copy import deepcopy
from loudspeakers import *
from regions_interest import *
from optimization import *
from post_processing import *
from utils_general import *
from utils_nfchoa import *
from green_function import *
import scipy.io
from time import time


def standard_instance(method, frequency, source_position, source_intensity, region_interest, loudspeakers, beta, green_function):
    radius = loudspeakers.diam / 2
    source = Source(frequency, source_position, source_intensity, regions_interest, green_function=green_function)
    if method in {'HOA', 'WFS'}:
        omega = 2 * np.pi * frequency
        array = sfs.array.circular(loudspeakers.n_loud, radius)
        sfs_normalization = 2 * np.pi * radius * source_intensity / loudspeakers.n_loud
        sfs_position = np.append(source_position - radius * np.ones(2), 0)
        if method == 'WFS':
            if np.linalg.norm(sfs_position) >= radius:
                d, s_wfs, ss_wfs = sfs.fd.wfs.point_25d(omega, array.x, array.n, sfs_position,
                                                        xref=np.asarray([0, 0, 0]))
            else:
                d, s_wfs, ss_wfs = sfs.fd.wfs.focused_25d(omega, array.x, array.n, sfs_position, -sfs_position,
                                                          xref=np.asarray([0, -radius, 0]))
            d *= s_wfs * sfs_normalization
        else:
            d, s_hoa, ss_hoa = sfs.fd.nfchoa.point_25d(omega, array.x, radius, sfs_position)
            d *= sfs_normalization
        d = np.expand_dims(d, axis=0)
        regions_interest.initialize()

    # SWEET
    elif method == 'SWEET':
        t0 = time()
        opt = SweetOptimization(loudspeakers, source, region_interest, truncation=0, penalty_norm=1, lamb=10 ** (-10))
        t1 = time()
        print('Time', t1 - t0)
        opt.plot_instance()
        d, d_sweet_best = opt.minimize(show=True, epsilon_sequence=99)

    # L2
    elif method == 'L2':
        # opt = LpqOptimization(loudspeakers, source, region_interest, truncation=0, penalty_norm=1, lamb=10 ** (-10))
        # d = opt.minimize(p=2, q=2, show=False, constraint_type=None)
        opt_L2 = LpqOptimization(loudspeakers, source, regions_interest, truncation=0,
                                 penalty_norm=1, lamb=0)
        d = opt_L2.minimize(p=2, q=2, weighted=False, constraint_type=None, show=False)

    # L1
    elif method == 'L1':
        opt = LpqOptimization(loudspeakers, source, region_interest, truncation=0, penalty_norm=1, lamb=10 ** (-10))
        d = opt.minimize(p=2, q=1, show=False, constraint_type=None)

    elif method == 'HOA_MANUAL':
        sfs_normalization = 2 * np.pi * radius * source_intensity / loudspeakers.n_loud
        centered_position = position - np.asarray([radius, radius])
        a_l = np.linspace(0, 2 * np.pi, num=loudspeakers.n_loud, endpoint=False)
        r_l = radius
        a_s = np.arctan2(centered_position[1], centered_position[0])
        r_s = np.linalg.norm(centered_position)
        f = frequency
        d = np.expand_dims(sfs_normalization * nfchoa_25d(a_s, r_s, a_l, r_l, f), axis=0)
        regions_interest.initialize()

    elif method == 'WFS_MANUAL':
        sfs_normalization = 2 * np.pi * radius * source_intensity / loudspeakers.n_loud
        x_s = position - np.asarray([radius, radius])
        x_l = loudspeakers.pos - np.asarray([radius, radius])
        x_r = np.zeros(2)
        f = frequency
        d = np.expand_dims(sfs_normalization * wfs_25d(x_s, x_l, x_r, f), axis=0)
        regions_interest.initialize()
    else:
        raise TypeError

    # scipy.io.savemat(f'{method}_FS_solution.mat', {'solution': d})
    # green_function_s = HRTF(frequency, reference=source_position)
    green_function_s = Monopole(frequency)
    # green_function_l = MonopoleScattered(frequency, scatters_positions)
    # green_function_l = HRTFScattered(frequency, scatters_positions, reference=position)
    # green_function_l = HRTFSquareImage2D(frequency, 2 * radius, theta=0, reference=source_position, beta=beta, n_order=3)
    # green_function_l = HRTF(frequency, reference=source_position)
    green_function_l = Monopole(frequency)
    loudspeakers = CircleLoudspeakers(n_loud, 2 * radius, frequency, green_function=green_function_l)
    source = Source(frequency, source_position, source_intensity, regions_interest, green_function_s)
    pp = PostProcessing(None, source, loudspeakers, regions_interest, green_function=green_function_l, n_points=500 ** 2, r=1.2)
    pp.solution = d
    us = pp.u(mask=True, name=f'{method}_{instance}')
    if method == 'L2':
        u0s = pp.u0(mask=True, name={instance})
    # scipy.io.savemat(f'{method}_{instance}_repair.mat', {'solution': d})
    # scipy.io.savemat(f'{method}_{instance}_repair.mat', {'u_left': us[0][0],
    #                                                 'u_right': us[1][0],
    #                                                 'u0_left': u0s[0][0],
    #                                                 'u0_right': u0s[1][0],
    #                                                 'solution': d})
    # scipy.io.savemat(f'{method}_REVERBFOCUS_{beta}.mat', {'u': us[0][0],
    #                                                         'u0': u0s[0][0],
    #                                                         'solution': d})

radius = 2.5
lim = np.asarray([radius, 2 * radius])
frequency = 343
n_loud = 20
position_y_map = {'NF': 3 * radius,
                  'FS': 4/3 * radius
}
methods = ['L2', 'HOA_MANUAL', 'WFS', 'SWEET']
methods = ['L2']
instances = ['NF', 'FS']
instances = ['FS']
beta = 0.2
# for beta in np.linspace(0, 1, 21, endpoint=True):
for instance in instances:
    position = np.asarray([radius, position_y_map[instance]])
    normalization = np.linalg.norm(position - lim) if position[1] > 2 * radius + 1 else 1
    intensity = np.ones(1) * normalization * pressure(60)
    for method in methods:
        regions_interest = CircleRegion(2500, 2 * radius, ratio=0.99)
        #green_function_s = Monopole(frequency)
        green_function_s = HRTF(frequency, reference=position) if method == 'SWEET' else Monopole(frequency)
        green_function_l = HRTF(frequency, reference=position) if method == 'SWEET' else Monopole(frequency)
        #green_function_l = HRTFSquareImage2D(frequency, 2 * radius, theta=0, reference=position, beta=beta, n_order=3) \
        #                   if method == 'SWEET' else GFSquareImage2D(frequency, 2 * radius, beta=beta, n_order=3)
        #green_function_l = Monopole(frequency)
        loudspeakers = CircleLoudspeakers(n_loud, 2 * radius, frequency, green_function=green_function_l)
        standard_instance(method, frequency, position, intensity, regions_interest, loudspeakers, beta, green_function=green_function_s)
