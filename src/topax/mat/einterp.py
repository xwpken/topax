import jax.numpy as np

from topax.tfp import Projection


class EnergyInterp:
    """Energy interpolation for fictitious domain TO (Wang et al. 2014)

    Blends linear elastic stress with hyperelastic PK1 via
    Heaviside projection gamma (Eq.1):

        P = sigma_lin(nabla u) + P_user(I + gamma * nabla u) - sigma_lin(gamma * nabla u)

    Parameters
    ----------
    gamma_proj : Projection, optional
        Heaviside projection for gamma. Defaults to
        ``Projection('wang', beta=500, eta=0.01)``.
    """

    def __init__(self, gamma_proj=None):
        if gamma_proj is not None:
            self.gamma_proj = gamma_proj  
        else:
            self.gamma_proj = Projection('wang', beta=500, eta=0.01)

    def __call__(self, u_grad, rho, penal, mu, lam, pk1_fn):
        """Compute energy-interpolated PK1 stress

        Parameters
        ----------
        u_grad : ndarray, shape (dim, dim)
            Displacement gradient at quadrature point.
        rho : float
            Physical density at quadrature point.
        penal : float
            SIMP penalty exponent.
        mu : float
            First Lame parameter at quadrature point.
        lam : float
            Second Lame parameter at quadrature point.
        pk1_fn : callable
            Hyperelastic PK1: ``pk1_fn(F)`` returns PK1 of shape (dim, dim).

        Returns
        -------
        ndarray, shape (dim, dim)
            Blended first Piola-Kirchhoff stress.
        """
        # Heaviside projection of penalized density (gamma in Eq.4)
        gamma = self.gamma_proj(rho ** penal)

        # Linear stress from full displacement gradient (first term of Eq.1)
        dim = u_grad.shape[-1]
        I = np.eye(dim)
        eps = 0.5 * (u_grad + u_grad.T)
        sigma = lam * np.trace(eps) * I + 2. * mu * eps

        # Linear stress from gamma-scaled gradient (sigma_lin in Eq.1)
        ug_s = gamma * u_grad
        eps_s = 0.5 * (ug_s + ug_s.T)
        sigma_gamma = lam * np.trace(eps_s) * I + 2. * mu * eps_s

        # Nonlinear PK1 from gamma-scaled deformation gradient (P_user in Eq.1)
        P_gamma = pk1_fn(ug_s + I)

        # Energy interpolation: P = P_user - sigma_lin(gamma * nabla u) + sigma_lin(nabla u)
        return P_gamma - sigma_gamma + sigma
