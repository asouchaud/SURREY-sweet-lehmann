import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import math

distances = [0.5, 1, 2, 3]

preceding_distance_map = {1: 0.5,
                          2: 1,
                          3: 2}

mat = {d: scipy.io.loadmat(f'../../main/hrtfs_wierstorf/QU_KEMAR_anechoic_{str(d).replace(".", "")}m.mat') for d in distances}


def hrtf_delay(signal, new_distance, old_distance, c=343):
    delay_distance = new_distance - old_distance
    delay_time = delay_distance / c
    frequencies = 44100 / 2048 * np.linspace(0, 1024, 1025)
    complex_exponential = np.e ** (-2 * np.pi * 1j * delay_time * frequencies)
    delayed_signal = signal * complex_exponential
    return delayed_signal


def hrtf_atten(signal, new_distance, old_distance):
    attenuated_signal = signal * abs(old_distance / new_distance)
    return attenuated_signal


def hrtf_adjust(signal, new_distance, old_distance):
    signal_delayed = hrtf_delay(signal, new_distance, old_distance)
    signal_del_att = hrtf_atten(signal_delayed, new_distance, old_distance)
    return signal_del_att


hrtf_bank = {'left': {distance: hrtf_delay(np.fft.rfft(mat[distance]['irs'][0][0][17].T), distance, 0.5)
                      for distance in distances},
             'right': {distance: hrtf_delay(np.fft.rfft(mat[distance]['irs'][0][0][18].T), distance, 0.5)
                       for distance in distances}}


def hrtf_generator_interpolator(side, distance, radian_angle):
    sexagesimal_angle = (math.degrees(radian_angle) + 360) % 360  # Returns the positive representation in degrees.
    index = (int(sexagesimal_angle) + 90) % 270  # index = 0 <-> sexagesimal_angle = -90 in the standard reference sys.
    if 0.5 < distance < 3:
        next_distance = min([d for d in [1, 2, 3] if distance <= d])
        preceding_distance = preceding_distance_map[next_distance]
        alpha = (distance - preceding_distance) / (next_distance - preceding_distance)
        hrtf = (1 - alpha) * hrtf_bank[side][preceding_distance][index] + alpha * hrtf_bank[side][next_distance][index]
    else:
        prox_distance = min([0.5, 3], key=lambda d: abs(distance - d))
        hrtf = hrtf_adjust(hrtf_bank[side][prox_distance][index], distance, prox_distance)
    return hrtf


def hrtf_generator(side, distance, radian_angle):
    sexagesimal_angle = radian_angle * 180 / np.pi
    index = int((sexagesimal_angle + 90) % 360) # index = 0 <-> sexagesimal_angle = -90 in the standard reference sys.
    return hrtf_adjust(hrtf_bank[side][3][index], distance, 3)




def hrtf_eval_freq(hrtf, frequency):
    if frequency == 22050:
        return hrtf[-1]
    index = int(frequency * 2048 / 44100)
    alpha = math.modf(frequency * 2048 / 44100)[0]
    hrtf_frequency = (1 - alpha) * hrtf[index] + alpha * hrtf[index + 1]
    return hrtf_frequency


def hrtf_test():
    frequency_domain = np.linspace(0, 1, int(2048 / 2) + 1) * 22050

    # distance > 26 cm
    distance = 1
    angle = np.pi

    hrtf_l = hrtf_generator('left', distance, angle)
    print(f'Power of HRTF Left Ear (distance: {distance}, angle: {angle}°): {np.sum(np.square(np.abs(hrtf_l)))}')

    plt.plot(frequency_domain, np.abs(hrtf_l))
    plt.title(f'HRTF Left Ear. Distance: {distance}. Angle: {angle}°')
    plt.show()

    plt.plot(np.fft.irfft(hrtf_l))
    plt.title(f'HRIR Left Ear. Distance: {distance}. Angle: {angle}°')
    plt.show()

    hrtf_r = hrtf_generator('right', distance, angle)
    print(f'Power of HRTF Right Ear (distance: {distance}, angle: {angle}°): {np.sum(np.square(np.abs(hrtf_r)))}')

    plt.plot(frequency_domain, np.abs(hrtf_r))
    plt.title(f'HRTF Right Ear. Distance: {distance}. Angle: {angle}°')
    plt.show()

    plt.plot(np.fft.irfft(hrtf_r))
    plt.title(f'HRIR Right Ear. Distance: {distance}. Angle: {angle}°')
    plt.show()
