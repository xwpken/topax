from abc import ABC, abstractmethod

import jax
import jax.numpy as np

from ..tfp import Pipeline


class Topology(ABC):

    """Base class for topology optimization problems"""

    def __init__(
        self,
        problem,
        x,
        transform,
        obj,
        cons,
        info=None,
    ):
        """Initialize topology optimization problem

        Parameters
        ----------
        problem : Problem
            FEM problem instance from jax-fem.
        x : ndarray
            Initial design variables.
        transform : Transform
            Transform pipeline for density filtering/projection.
        obj : callable
            Objective function f(x_phys) -> scalar.
        cons : list of callable
            List of constraint functions g(x_phys) -> scalar.
        info : dict or None
            Optional metadata dict.
        """
        self.problem = problem
        self.elem_vols = np.sum(problem.fes[0].JxW, axis=1, keepdims=True)
        self.dom_vol = np.sum(self.elem_vols)

        self.transform = transform
        self.obj = obj
        self.cons = cons
        self.info = info or {}

        self._setup_grad_fns()
        self.num_x = len(x)
        self.update(x)

    def _setup_grad_fns(self):
        """Wrap obj/cons with transforms and jax.value_and_grad"""

        def _wrapped_obj(x):
            x_phys = self.transform(x)
            return self.obj(x_phys)

        self._obj_grad_fn = jax.value_and_grad(_wrapped_obj)

        self._cons_grad_fns = []
        for cons_fn in self.cons:
            def _wrap_cons(fn):
                def _wrapped_cons(x):
                    x_phys = self.transform(x)
                    return fn(x_phys)
                return _wrapped_cons

            self._cons_grad_fns.append(jax.value_and_grad(_wrap_cons(cons_fn)))

    def _clear(self):
        """Clear cached evaluation results"""
        self.J = None
        self.dJ = None
        self.c = None
        self.dc = None
        self._valid = False

    def update(self, xnew):
        """Update design variables and clear cache

        Parameters
        ----------
        xnew : ndarray
            New design variables.
        """
        self.x = xnew
        self.x_phys = self.transform(xnew)
        self._clear()

    def eval(self):
        """Evaluate objective and constraints (cached)

        Results are cached and reused until the next update() call.

        Returns
        -------
        J : float
            Objective value.
        dJ : ndarray
            Objective gradient w.r.t. design variables.
        c : list of float
            List of constraint values.
        dc : list of ndarray
            List of constraint gradients.
        """
        if not self._valid:
            self.J, self.dJ = self._obj_grad_fn(self.x)
            self.c = []
            self.dc = []
            for cons_grad_fn in self._cons_grad_fns:
                c_val, c_grad = cons_grad_fn(self.x)
                self.c.append(c_val)
                self.dc.append(c_grad)
            self._valid = True
        return self.J, self.dJ, self.c, self.dc

    def update_eval(self, xnew):
        """Update design variables and evaluate

        Parameters
        ----------
        xnew : ndarray
            New design variables.

        Returns
        -------
        Same as eval().
        """
        self.update(xnew)
        return self.eval()

    def trace(self):
        """Return intermediate states through transform chain

        If the transform is a Pipeline, returns tuples of all
        intermediate fields (x0, x1, ..., x_phys).
        Otherwise returns (x, x_phys).

        Returns
        -------
        tuple of ndarray
            Fields at each stage of the transform chain.
        """
        if isinstance(self.transform, Pipeline):
            x0 = self.x.copy()
            vals = [x0]
            current = x0
            for step in self.transform.steps:
                current = step(current)
                vals.append(current.copy())
            return tuple(vals)
        else:
            x0 = self.x.copy()
            x1 = self.x_phys.copy()
            return x0, x1

    @abstractmethod
    def compute_vf(self):
        """Compute volume fraction

        Returns
        -------
        float
            Volume fraction in [0, 1].
        """
        pass
