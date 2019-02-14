#!/usr/bin/env python

import scipy.constants as const
import numpy as np


def lB(T, epsilon):
    # Bjerrum length [nm]
    return const.e ** 2 / (np.pi * const.epsilon_0 * epsilon * const.k * T) * 1e9


def K_est(sigma, d, T, q, epsilon, tol=1e-5, max_iterations=1000):
    # Gouy-Chapman length [nm]
    b = 1.0 / (2 * np.pi * lB(T, epsilon) * q * sigma)

    # lower bound
    l = 0.0

    # upper bound
    u = np.pi / d

    # iteration counter
    i = 0

    # Use bisection to solve:
    # K tan(K d/2) = 1 / b
    while u - l > tol:
        i += 1
        if i > max_iterations:
            print('Max iteration limit reached')
            break
        K = (u + l) / 2
        if K * np.tan(K * d / 2.0) > 1.0 / b:
            u = K
        else:
            l = K

    return K


# Surface charge sigma [e/nm^2]
# Surface separation d [nm]
# Temperature T [K]
# valency q
# Relative permittivity of water at 300K epsilon
def p_pb(sigma=1.0, d=1.0, T=300.0, q=1.0, epsilon=80.0):
    # Estimate K [1/nm]
    K = K_est(sigma, d, T, q, epsilon)

    # Poisson Boltzmann Pressure [Bar]
    return const.k * T / (2 * np.pi * lB(T, epsilon) * q) * K**2 * 1e27 * 10e-5
