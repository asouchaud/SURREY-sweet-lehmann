import numpy as np
from scipy.special import factorial, factorial2
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from utils_general import pressure


# Paar psychoacoustic model for the threshold of hearing
def absolute_threshold_hearing(f, margin=0):
    dB = 3.64 * (f / 1000) ** (-0.8) - 6.5 * np.e ** (-0.6 * (f / 1000 - 3.3) ** 2) + 10 ** (-3) * (f / 1000) ** 4
    return pressure(dB + margin)


def ERB(f):
    return 24.7 * (4.37 * f / 1000 + 1)


def ERB_scale(f):
    return 21.4 * np.log10(1 + 4.37 * f / 1000)


def ERB_scale_inv(ERBS):
    return (10 ** (ERBS / 21.4) - 1) / 0.00437


def eta(f):
    return 1 / absolute_threshold_hearing(f)


def gamma(f, f0, n=4):
    p = 2 ** (n - 1) * factorial(n - 1) / (np.pi * factorial2(2 * n - 3))
    return (1 + ((f - f0) / (p * ERB(f0))) ** 2) ** (-n / 2)


def bisection_method(g, x0, x1, tol=10 ** (-15)):
    sgn = (g(x1) - g(x0)) / abs(g(x1) - g(x0))
    while True:
        xm = (x1 + x0) / 2
        if abs(g(xm)) < tol:
            break
        if g(xm) * sgn > 0:
            x1 = xm
        else:
            x0 = xm
    return xm


F = np.asarray([ERB_scale_inv(ERBS) for ERBS in np.linspace(ERB_scale(20), ERB_scale(10000), 100)])


def gen_constants():
    def g(x):
        s = np.sum(
            (eta(1000) * gamma(1000, F) * pressure(52)) ** 2 / (
                    (eta(1000) * gamma(1000, F) * pressure(70)) ** 2 + x * np.sum(gamma(1000, F) ** 2)))
        return 1 - x * s

    x0 = 0
    x1 = 100
    xinf = bisection_method(g, x0, x1)
    return xinf, xinf * np.sum(gamma(1000, F) ** 2)


C_s, C_a = gen_constants()


# Knobel psychoacoustic thresholds of LDL
def knobel_model():
    dom = [500, 1000, 2000, 3000, 4000, 6000, 8000]
    imag = [95.5, 94.5, 94.5, 94, 96, 98, 86.75]
    knobel_spline = CubicSpline(dom, imag, bc_type='natural', extrapolate=True)

    def knobel_interpolation(f):
        return pressure(knobel_spline(f))

    return knobel_interpolation


# Sherlock psychoacoustic thresholds for LDL
def sherlock_model():
    dom = [500, 1000, 2000, 4000]
    imag = [102.2, 103.86, 101.65, 100.75]
    sherlock_spline = CubicSpline(dom, imag, bc_type='natural', extrapolate=True)

    def sherlock_interpolation(f):
        return pressure(sherlock_spline(f))

    return sherlock_interpolation

def plot_LDL_interpolations(fontsize_legend, fontsize_label):
    dom = [500, 1000, 2000, 3000, 4000, 6000, 8000]
    imag = [95.5, 94.5, 94.5, 94, 96, 98, 86.75]
    knobel_spline = CubicSpline(dom, imag, bc_type='natural', extrapolate=True)
    xs = np.arange(50, 10000, 10)
    fig, ax = plt.subplots()
    ax.plot(dom, imag, 'o', label='Data')
    ax.plot(xs, knobel_spline(xs), label="Cubic spline interpolation")
    ax.set_xlim(50, 10000)
    ax.legend(loc='lower left', ncol=1, fontsize=fontsize_legend)
    # plt.title('Knobel interpolation')

    dom = [500, 1000, 2000, 4000]
    imag = [102.2, 103.86, 101.65, 100.75]
    sherlock_spline = CubicSpline(dom, imag, bc_type='natural', extrapolate=True)
    plt.xlabel('Frequency (Hz)', fontsize=fontsize_label)
    plt.ylabel('Level (dB)', fontsize=fontsize_label)
    plt.savefig(fname='cubic_splines_knobel.png', dpi=300)
    plt.show()

    xs = np.arange(50, 10000, 10)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(dom, imag, 'o', label='Data')
    ax.plot(xs, sherlock_spline(xs), label="Cubic spline interpolation")
    ax.set_xlim(50, 10000)
    ax.legend(loc='lower left', ncol=1, fontsize=fontsize_legend)
    # plt.title('Sherlock interpolation')

    plt.xlabel('Frequency (Hz)', fontsize=fontsize_label)
    plt.ylabel('Level (dB)', fontsize=fontsize_label)
    plt.savefig(fname='cubic_splines_sherlock.png', dpi=300)
    plt.show()

#plot_LDL_interpolations(15, 15)