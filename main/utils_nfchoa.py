from scipy.special import hankel2, spherical_jn, spherical_yn
import numpy as np
from utils_general import row_wise_norm2


def spherical_hn2(n, z):
    r"""Spherical Hankel function of 2nd kind.
    Defined as https://dlmf.nist.gov/10.47.E6,
    .. math::
        \hankel{2}{n}{z} = \sqrt{\frac{\pi}{2z}}
        \Hankel{2}{n + \frac{1}{2}}{z},
    where :math:`\Hankel{2}{n}{\cdot}` is the Hankel function of the
    second kind and n-th order, and :math:`z` its complex argument.
    Parameters
    ----------
    n : array_like
        Order of the spherical Hankel function (n >= 0).
    z : array_like
        Argument of the spherical Hankel function.
    """
    return spherical_jn(n, z) - 1j * spherical_yn(n, z)


def max_order_circular_harmonics(N):
    r"""Maximum order of 2D/2.5D HOA.
    It returns the maximum order for which no spatial aliasing appears.
    It is given on page 132 of :cite:`Ahrens2012` as
    .. math::
        \mathtt{max\_order} =
            \begin{cases}
                N/2 - 1 & \text{even}\;N \\
                (N-1)/2 & \text{odd}\;N,
            \end{cases}
    which is equivalent to
    .. math::
        \mathtt{max\_order} = \big\lfloor \frac{N - 1}{2} \big\rfloor.
    Parameters
    ----------
    N : int
        Number of secondary sources.
    """
    return (N - 1) // 2


def nfchoa_25d(a_s, r_s, a_l, r_l, f, c=343):  # ojo con c
    k = 2 * np.pi * f / c
    n_l = len(a_l)
    max_order = max_order_circular_harmonics(n_l)
    coefficients_indexes = range(-max_order, max_order + 1)
    coefficients = np.asarray([spherical_hn2(abs(m), k * r_s) / spherical_hn2(abs(m), k * r_l)
                               * np.e ** (1j * m * (a_l - a_s))
                               for m in coefficients_indexes])
    if r_s < r_l:  # weighting como esta propuesto en el libro de Ahrens
        limit = np.floor(k * r_s)
        weighting_vector = np.asarray([1 / 2 * (np.cos(m / limit * np.pi) + 1) if abs(m) <= limit else 0
                                       for m in coefficients_indexes])
        coefficients = (coefficients.T * weighting_vector).T
    return np.sum(coefficients, axis=0) / (2 * np.pi * r_l)


def wfs_25d(x_s, x_l, x_r, f, c=343):
    k = 2 * np.pi * f / c
    r_loud = np.linalg.norm(x_l[0])

    r_srce = np.linalg.norm(x_s)
    x_l_x_s = row_wise_norm2(x_l - x_s)
    x_l_x_r = row_wise_norm2(x_l - x_r)
    a_diff = ((x_l - x_s).T / x_l_x_s).T
    a_loud = (x_l.T / row_wise_norm2(x_l)).T
    a_srce = (x_s.T / row_wise_norm2(x_s)).T
    a_damp = np.asarray([- a_diff[i] @ a_srce for i in range(len(a_diff))])
    weighting_vector = a_loud @ a_srce >= 0.5
    if r_loud < r_srce:
        return np.sqrt(8 * np.pi * 1j * k) * np.sqrt(x_l_x_r * x_l_x_s / x_l_x_r + x_l_x_s) \
               * a_damp * np.e ** (-1j * k * x_l_x_s) / (4 * np.pi * x_l_x_s) * weighting_vector
    else:
        return np.sqrt(1j * k * x_l_x_r / x_l_x_s) * a_damp * np.e ** (1j * k * x_l_x_s) * weighting_vector
