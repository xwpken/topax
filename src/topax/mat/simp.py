class SIMP:

    """Solid Isotropic Material with Penalization (SIMP) model"""

    def __init__(self, E_max=1.0, E_min=1e-9, penal=3.0):
        """
        Parameters
        ----------
        E_max : float
            Maximum Young's modulus (solid material).
        E_min : float
            Minimum Young's modulus (void material).
        penal : float
            Penalization factor.
        """
        self.E_max = E_max
        self.E_min = E_min
        self.penal = penal
        self.citation = (
            "Bendsøe, M.P. (1989) Optimal shape design as a material "
            "distribution problem. Structural Optimization, 1(4), 193-202"
        )

    def __call__(self, x):
        """Compute Young's modulus distribution using SIMP interpolation

        Parameters
        ----------
        x : ndarray
            Density field.

        Returns
        -------
        ndarray
            Interpolated Young's modulus.
        """
        return self.E_min + (x ** self.penal) * (self.E_max - self.E_min)

    def get_config(self):
        """Get model configuration"""
        return {
            'E_max': self.E_max,
            'E_min': self.E_min,
            'penal': self.penal,
            'citation': self.citation
        }

    def __repr__(self):
        return f"SIMP(E_max={self.E_max}, E_min={self.E_min}, penal={self.penal})"
