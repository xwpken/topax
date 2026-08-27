from typing import Any

import numpy as onp

from mmapy.mma import mmasub

from .base import Optimizer


_DEFAULT_A0 = 1.0
_DEFAULT_C = 1000.0
_DEFAULT_D = 1.0


class MMA(Optimizer):

    """Method of Moving Asymptotes (MMA) optimizer.

    Wraps the mmasub implementation from the mmapy package
    (based on Svanberg 1987).  Maintains internal state across
    iterations (asymptotes, previous design variables, etc.).

    Parameters
    ----------
    move : float
        Move limit, default 0.5.
    a0 : float
        Constant a_0 in the term a_0*z, default 1.0.
    c_penalty : float
        Constant c_i in the term c_i*y_i, default 1000.0.
    d_penalty : float
        Constant d_i in the term 0.5*d_i*(y_i)^2, default 1.0.
    **kwargs
        Additional keyword arguments forwarded to mmasub
        (asyinit, asydecr, asyincr, asymin, asymax, raa0, albefa).
    """

    def __init__(
        self,
        move: float = 0.5,
        a0: float = _DEFAULT_A0,
        c_penalty: float = _DEFAULT_C,
        d_penalty: float = _DEFAULT_D,
        **kwargs: Any,
    ):
        self.move = move
        self.a0 = a0
        self._c_penalty = c_penalty
        self._d_penalty = d_penalty
        self._mmasub_kwargs = kwargs
        self._initialized = False

    def _init(self, n: int, m: int, x0: onp.ndarray) -> None:
        self.n = n
        self.m = m
        self.xold1 = x0.copy()
        self.xold2 = x0.copy()
        self.low = onp.zeros((n, 1))
        self.upp = onp.ones((n, 1))
        self.iter = 0
        self.a = onp.zeros((m, 1))
        self.c = self._c_penalty * onp.ones((m, 1))
        self.d = self._d_penalty * onp.ones((m, 1))
        self.xmin = onp.zeros((n, 1))
        self.xmax = onp.ones((n, 1))
        self._initialized = True

    def update(self, topology, dJ, dc, J=None, c=None, **kwargs):
        """Perform one MMA update step.

        Parameters
        ----------
        topology : Topology
            Topology object (provides ``.x``).
        dJ : ndarray
            Objective gradient w.r.t. design variables.
        dc : list of ndarray
            List of constraint gradients, one per constraint.
        J : float, optional
            Objective value.  Required on first call; on subsequent
            calls it may be omitted (the stored value is reused).
        c : list of float, optional
            Constraint values.  Required on first call.

        Returns
        -------
        ndarray
            Updated design variables (same shape as ``topology.x``).
        """
        xval = onp.asarray(topology.x).reshape(-1, 1)
        n = xval.shape[0]
        m = len(dc)

        if not self._initialized:
            if J is None or c is None:
                raise ValueError(
                    "J (objective) and c (constraint values) are required "
                    "on the first call to MMA.update()."
                )
            self._init(n, m, xval)

        if n != self.n or m != self.m:
            raise ValueError(
                f"Inconsistent problem dimensions: "
                f"got (n={n}, m={m}) but expected (n={self.n}, m={self.m})."
            )

        self.iter += 1

        f0val = float(onp.asarray(J)) if J is not None else self._f0val_prev
        fval = onp.asarray(c).reshape(-1, 1) if c is not None else self._fval_prev
        df0dx = onp.asarray(dJ).reshape(-1, 1)
        dfdx = onp.asarray(dc).reshape(m, n)

        xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, low, upp = mmasub(
            m, n, self.iter, xval, self.xmin, self.xmax,
            self.xold1, self.xold2, f0val, df0dx, fval, dfdx,
            self.low, self.upp, self.a0, self.a, self.c, self.d,
            self.move, **self._mmasub_kwargs,
        )

        self.xold2 = self.xold1.copy()
        self.xold1 = xval.copy()
        self.low = low
        self.upp = upp
        self._f0val_prev = f0val
        self._fval_prev = fval

        return xmma.reshape(onp.asarray(topology.x).shape)

    def reset(self):
        """Reset internal state.  The next ``update()`` call
        will re-initialise from scratch."""
        self._initialized = False
