import jax.numpy as np
import numpy as onp
from mmapy.mma import mmasub


class OC:
    """Optimality-criteria optimizer for one volume constraint.

    Parameters
    ----------
    move : float
        Maximum change of a design variable per iteration.
    damping : float
        Damping exponent in the OC update.
    tolerance : float
        Relative tolerance of the Lagrange-multiplier bisection.
    """

    def __init__(self, move=0.2, damping=0.5, tolerance=1e-3):
        self.move = move
        self.damping = damping
        self.tolerance = tolerance

    def update(self, x, objective_grad, constraint_grad, volume_fn, target_volume):
        """Compute the next design.

        Parameters
        ----------
        x : ndarray
            Current design variables.
        objective_grad : ndarray
            Objective gradient with respect to ``x``.
        constraint_grad : ndarray
            Volume-constraint gradient with respect to ``x``.
        volume_fn : callable
            Function returning the physical volume fraction of a trial design.
        target_volume : float
            Target volume fraction.

        Returns
        -------
        ndarray
            Updated design variables.
        """
        lag_lower = 1e-9
        lag_upper = 1e9

        while (lag_upper - lag_lower) / (lag_upper + lag_lower) > self.tolerance:
            lag = 0.5 * (lag_lower + lag_upper)
            x_trial = self._update_formula(x, objective_grad, constraint_grad, lag)
            x_trial = self._apply_bounds(x, x_trial)

            if volume_fn(x_trial) > target_volume:
                lag_lower = lag
            else:
                lag_upper = lag

        return x_trial

    def _update_formula(self, x, objective_grad, constraint_grad, lag):
        ratio = -objective_grad / (constraint_grad * lag)
        ratio = np.maximum(ratio, 1e-10)
        return x * ratio**self.damping

    def _apply_bounds(self, x, xnew):
        lower = np.maximum(0.0, x - self.move)
        upper = np.minimum(1.0, x + self.move)
        return np.clip(xnew, lower, upper)


class MMA:
    """Method of Moving Asymptotes optimizer.

    Parameters
    ----------
    move : float
        Move limit.
    a0 : float
        Constant multiplying the MMA ``z`` variable.
    c_penalty : float
        Linear penalty coefficient for constraint slack variables.
    d_penalty : float
        Quadratic penalty coefficient for constraint slack variables.
    **kwargs
        Additional keyword arguments forwarded to ``mmasub``.
    """

    def __init__(
        self,
        move=0.5,
        a0=1.0,
        c_penalty=1000.0,
        d_penalty=1.0,
        **kwargs,
    ):
        self.move = move
        self.a0 = a0
        self.c_penalty = c_penalty
        self.d_penalty = d_penalty
        self.mmasub_kwargs = kwargs
        self.initialized = False

    def _initialize(self, x, num_constraints):
        num_design = x.size
        self.num_design = num_design
        self.num_constraints = num_constraints
        self.xold1 = x.copy()
        self.xold2 = x.copy()
        self.low = onp.zeros((num_design, 1))
        self.upp = onp.ones((num_design, 1))
        self.xmin = onp.zeros((num_design, 1))
        self.xmax = onp.ones((num_design, 1))
        self.a = onp.zeros((num_constraints, 1))
        self.c = self.c_penalty * onp.ones((num_constraints, 1))
        self.d = self.d_penalty * onp.ones((num_constraints, 1))
        self.iteration = 0
        self.initialized = True

    def update(self, x, objective, objective_grad, constraints, constraint_grads):
        """Compute the next design.

        Parameters
        ----------
        x : ndarray
            Current design variables.
        objective : float
            Current objective value.
        objective_grad : ndarray
            Objective gradient with respect to ``x``.
        constraints : float or ndarray
            Current constraint values.
        constraint_grads : ndarray
            Constraint gradients. A single gradient may have the same shape as
            ``x``; multiple gradients are stacked along the first axis.

        Returns
        -------
        ndarray
            Updated design variables with the same shape as ``x``.
        """
        original_shape = onp.asarray(x).shape
        xval = onp.asarray(x).reshape(-1, 1)
        fval = onp.atleast_1d(onp.asarray(constraints)).reshape(-1, 1)
        num_constraints = fval.shape[0]

        if not self.initialized:
            self._initialize(xval, num_constraints)

        if xval.size != self.num_design or num_constraints != self.num_constraints:
            raise ValueError("The MMA problem dimensions changed after initialization.")

        self.iteration += 1
        df0dx = onp.asarray(objective_grad).reshape(-1, 1)
        dfdx = onp.asarray(constraint_grads).reshape(num_constraints, -1)

        result = mmasub(
            self.num_constraints,
            self.num_design,
            self.iteration,
            xval,
            self.xmin,
            self.xmax,
            self.xold1,
            self.xold2,
            float(onp.asarray(objective)),
            df0dx,
            fval,
            dfdx,
            self.low,
            self.upp,
            self.a0,
            self.a,
            self.c,
            self.d,
            self.move,
            **self.mmasub_kwargs,
        )
        xmma, _, _, _, _, _, _, _, _, low, upp = result

        self.xold2 = self.xold1.copy()
        self.xold1 = xval.copy()
        self.low = low
        self.upp = upp
        return xmma.reshape(original_shape)

    def reset(self):
        """Discard the stored MMA iteration state."""
        self.initialized = False
