import jax
import jax.numpy as np

from .base import Transform


class Projection(Transform):

    """Projection with pluggable methods

    Supported methods: guest, wang

    Examples:
        >>> proj = Projection('guest', beta=2.0)
        >>> x_phys = proj(x_tilde)
        >>> dx = proj.grad(x_tilde)
    """

    def __init__(self, method, **params):
        if method not in _METHODS:
            raise ValueError(f"Unknown method: {method}")
        info = _METHODS[method]
        self._fn = info['fn']
        self._citation = info['citation']
        self.method = method
        self.params = {**info['defaults'], **params}
        for k, v in self.params.items():
            setattr(self, k, v)

    def __call__(self, x):
        return self._fn(self, x)

    def value_and_grad(self, x):
        """Compute projection value and gradient for each element

        Parameters
        ----------
        x : ndarray
            Input array.

        Returns
        -------
        tuple of ndarray
            (values, gradients) each of shape (n,).
        """
        return jax.vmap(jax.value_and_grad(self.__call__))(x)

    def grad(self, x):
        """Compute projection gradient for each element

        Parameters
        ----------
        x : ndarray
            Input array.

        Returns
        -------
        ndarray
            Gradients of shape (n,).
        """
        scalar = lambda x: self(x)[0]
        return jax.vmap(jax.grad(scalar))(x)

    def set_params(self, **params):
        """Set projection parameters

        Parameters
        ----------
        **params : dict
            Keyword arguments matching the method's default params.
        """
        for k in params:
            if k not in self.params:
                raise ValueError(f"Unknown parameter: {k}")
        self.params |= params
        for k, v in params.items():
            setattr(self, k, v)

    def __repr__(self):
        items = ', '.join(f'{k}={v}' for k, v in self.params.items())
        methods = sorted(_METHODS)
        avail = ', '.join(
            f"{m}({', '.join(f'{k}={v}' for k, v in _METHODS[m]['defaults'].items())})"
            for m in methods
        )
        return (
            f"Projection(method='{self.method}', {items})\n"
            f"  citation: {self._citation}\n"
            f"  available: {avail}"
        )


def _guest(self, x):
    """Guest et al. (2004) projection"""
    return 1.0 - np.exp(-self.beta * x) + x * np.exp(-self.beta)


def _wang(self, x):
    """Wang et al. (2011) projection"""
    numerator = np.tanh(self.beta * self.eta) + np.tanh(self.beta * (x - self.eta))
    denominator = np.tanh(self.beta * self.eta) + np.tanh(self.beta * (1.0 - self.eta))
    return numerator / denominator


_METHODS = {
    'guest': {
        'fn': _guest,
        'defaults': {'beta': 1.0},
        'citation': (
            "Guest, J.K., Prévost, J.H. and Belytschko, T. (2004) "
            "Achieving minimum length scale in topology optimization "
            "using nodal design variables and projection functions. "
            "IJNME, 61(2), 238-254"
        ),
    },
    'wang': {
        'fn': _wang,
        'defaults': {'beta': 1.0, 'eta': 0.5},
        'citation': (
            "Wang, F., Lazarov, B.S. and Sigmund, O. (2011) "
            "On projection methods, convergence and robust formulations "
            "in topology optimization. SMO, 43(6), 767-784"
        ),
    },
}
