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

plot_type = 'positive'
instance = 'WEIGHTED'
reference = np.asarray([2.5, 2.5 * 4 / 3]) if instance == 'F' else np.asarray([2.5, 2.5 * 3])
coloration_array = []
# numbers = ['0.0', '0.05', '0.1', '0.15', '0.2', '0.25', '0.3', '0.35', '0.4', '0.45', '0.5', '0.55', '0.6',
#                 '0.65', '0.7', '0.75', '0.8', '0.85', '0.9', '1.0']
numbers = ['NF', 'FS']
#numbers = ['50', '100', '150', '200', '250', '300', '350', '400', '450', '500', '550', '600', '650', '700', '750', '800', '850', '900', '950', '1000', '1050', '1100', '1150', '1200', '1250', '1300', '1350', '1400', '1450', '1500', '1550', '1600', '1650', '1700', '1750', '1800', '1850', '1900', '1950', '2000']
for method in ['L2']:
    coloration_method = []
    for number in numbers:
        mat = scipy.io.loadmat(f'coloration_error/coloration_error_{method}_{instance}_{number}.mat')
        #mat = scipy.io.loadmat(f'{method}_{instance}_coloration_error.mat')
        coloration = mat['coloration_error']
        # mat =


        radius = 2.5
        n_loud = 20
        frequency = 343
        source_position = np.asarray([radius, 3 * radius])
        lim = np.asarray([radius, 2 * radius])
        normalization = np.linalg.norm(source_position - lim) if source_position[1] > 2 * radius + 1 else 1
        source_intensity = normalization * pressure(60)

        region_interest = CircleRegion(300, 2 * radius, ratio=0.99)
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
        # d = scipy.io.loadmat(f'{method}_{instance}_{number}.mat')['solution']
        # d = scipy.io.loadmat(f'{method}_{instance}_solution_correction.mat')['solution']
        # d = scipy.io.loadmat(f'{method}_{instance}_{number}.mat')['solution']
        d = scipy.io.loadmat(f'{method}_WEIGHTED_{number}_solution.mat')['solution']

        pp = PostProcessing(None, source, loudspeakers, region_interest, green_function=green_function, n_points=30 ** 2, r=1.2)
        pp.solution = d

        sweet_spot_size = pp.external_plot(array=coloration,
                                             plot_type=plot_type,
                                             max=100,
                                             min=0,
                                             center=13,
                                             position_reference=reference,
                                             mask=True,
                                             regions=True,
                                             loudspeakers=True,
                                             name=f'{method}_{instance}_coloration_error')
        # coloration_method.append(sweet_spot_size)
    # coloration_array.append(coloration_method)
    # plt.plot([float(x) for x in numbers], coloration_method, label=method)
# plt.legend(fontsize=18)
# plt.xlabel('Frequency (Hz)', fontsize=18)
# #plt.xlabel('Wall Reflection Coefficient', fontsize=18)
# plt.ylabel('Coloration Sweet Spot', fontsize=18)
# plt.xlim(50, 2000)
# plt.ylim(0, 1)
# plt.grid()
# plt.savefig(fname=f'coloration_linspace', dpi=300)
# plt.show()