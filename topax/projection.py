import jax.numpy as np


def proj_fn_guest(x, beta=1.0):
    """Apply the exponential Heaviside projection by Guest et al.

    Parameters
    ----------
    x : ndarray
        Filtered density field.
    beta : float
        Projection sharpness.

    Returns
    -------
    ndarray
        Projected density field.
    """
    return 1.0 - np.exp(-beta * x) + x * np.exp(-beta)


def proj_fn_wang(x, beta=1.0, eta=0.5):
    """Apply the smooth Heaviside projection by Wang et al.

    Parameters
    ----------
    x : ndarray
        Filtered density field.
    beta : float
        Projection sharpness.
    eta : float
        Projection threshold.

    Returns
    -------
    ndarray
        Projected density field.
    """
    numerator = np.tanh(beta * eta) + np.tanh(beta * (x - eta))
    denominator = np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))
    return numerator / denominator
