import sfs
import json
from loudspeakers import *
from regions_interest import *
from optimization import *
from post_processing import *
from green_function import *
from utils_general import *
from utils_nfchoa import *

from time import time
import scipy.io
from regions_interest import *

plot_type = 'vector'
instance = 'FS'
reference = {'FS': np.asarray([2.5, 2.5 * 4 / 3]),
             'NF': np.asarray([2.5, 2.5 * 3]),
             '': None
             }
# numbers = ['0.0', '0.05', '0.1', '0.15', '0.2', '0.25', '0.3', '0.35', '0.4', '0.45', '0.5', '0.55', '0.6',
#                 '0.65', '0.7', '0.75', '0.8', '0.85', '0.9', '1.0']
numbers = ['0.2']
numbers = ['4']
# numbers = ['50', '100', '150', '200', '250', '300', '350', '400', '450', '500', '550', '600', '650', '700', '750', '800', '850', '900', '950', '1000', '1050', '1100', '1150', '1200', '1250', '1300', '1350', '1400', '1450', '1500', '1550', '1600', '1650', '1700', '1750', '1800', '1850', '1900', '1950', '2000']
for method in ['L2', 'WFS', 'HOA_MANUAL', 'SWEET']:
    localization_error_method = []
    for number in numbers:

        mat = scipy.io.loadmat(f'localization_error/localization_estimation_{method}_{instance}_{number}.mat')
        #mat = scipy.io.loadmat(f'{method}_{instance}_localization_estimation2.mat')
        localization_u = mat['localization_u']
        mat_reference = scipy.io.loadmat(f'localization_error/localization_estimation_u0_{instance}_{number}.mat')
        #mat_reference = scipy.io.loadmat(f'u0_{instance}_localization_estimation2.mat')
        localization_u0 = mat_reference['localization_u0']
        # localization_sweet = np.abs(localization_error) <= 5
        # r = 1.2
        # n_points = 300 ** 2
        # diameter = 5
        # region = CircleRegion(n_points=n_points,
        #                       diameter=r * diameter,
        #                       ratio=0.999 / r,
        #                       shift=(1 - r) / 2 * np.ones(2) * diameter,
        #                       region_type='complete')
        # region.initialize()
        # localization_error = mat['localization_error'] * region.mask
        # print(localization_error)
        # print(np.abs(localization_error))
        # im = plt.imshow(rotation_45(localization_error),
        #                 cmap=plt.cm.RdBu_r,
        #                 extent=region.extent,
        #                 interpolation='spline16')
        # plt.show()

        radius = 2.5
        n_loud = 20
        frequency = 343
        source_position = np.asarray([radius, 4 / 3 * radius])
        lim = np.asarray([radius, 2 * radius])
        normalization = np.linalg.norm(source_position - lim) if source_position[1] > 2 * radius + 1 else 1
        source_intensity = normalization * pressure(60)

        region_interest = CircleRegion(2000, 2 * radius, ratio=0.99)
        region_interest.initialize()

        green_function = Monopole(frequency)
        loudspeakers = CircleLoudspeakers(n_loud, 2 * radius, frequency, green_function=green_function)
        source = Source(frequency, source_position, source_intensity, region_interest, green_function=green_function)

        # sfs_normalization = 2 * np.pi * radius * source_intensity / loudspeakers.n_loud
        # centered_position = source_position - np.asarray([radius, radius])
        # a_l = np.linspace(0, 2 * np.pi, num=loudspeakers.n_loud, endpoint=False)
        # r_l = radius
        # a_s = np.arctan2(centered_position[1], centered_position[0])
        # r_s = np.linalg.norm(centered_position)
        # f = frequency
        # d = np.expand_dims(sfs_normalization * nfchoa_25d(a_s, r_s, a_l, r_l, f), axis=0)

        d = scipy.io.loadmat(f'{method}_{instance}_repair.mat')['solution']

        pp = PostProcessing(None, source, loudspeakers, region_interest, green_function=green_function, n_points=300 ** 2, r=1.2)
        pp.solution = d
        # pp.external_plot(array=np.log(np.abs(localization_error) / 5),
        #                  plot_type=None,
        #                  max=np.log(4),
        #                  min=-np.log(12),
        #                  center=0,
        #                  mask=True,
        #                  regions=True,
        #                  loudspeakers=True,
        #                  name=f'{method} Localization Error')

        sweet_spot_size = pp.external_plot(array=localization_u,
                         array_reference=localization_u0,
                         plot_type=plot_type,
                         vmax=90,
                         position_reference=reference[instance],
                         mask=True,
                         regions=True,
                         loudspeakers=True,
                         name=f'{method}_{instance}_localization_error')
    #    localization_error_method.append(sweet_spot_size)
    #plt.plot([float(x) for x in numbers], localization_error_method, label=method)
plt.legend(fontsize=18)
plt.grid()
plt.xlabel('Frequency (Hz)', fontsize=18)
#plt.xlabel('Wall Reflection Coefficient', fontsize=18)
plt.ylabel('Localization Sweet Spot', fontsize=18)
plt.xlim(50, 2000)
plt.ylim(0, 1)
plt.savefig(fname=f'localization_linspace', dpi=300)
plt.show()