import jax
import jax.numpy as np
import numpy as onp
from jax_fem.problem import Problem


class TopOptProblem(Problem):
    """JAX-FEM problem with support for nodal loads."""

    def add_nodal_load(self, load, variable=0):
        """Add a load vector defined directly on mesh nodes.

        Parameters
        ----------
        load : ndarray or callable
            Nodal load with shape ``(num_nodes, vec)``. A callable receives
            the nodal coordinates and must return an array with that shape.
            Callables may read values updated by ``set_params``.
        variable : int
            Index of the finite-element variable receiving the load.
        """
        if not 0 <= variable < self.num_vars:
            raise ValueError(f"Invalid finite-element variable index: {variable}")

        if not hasattr(self, "nodal_loads"):
            self.nodal_loads = []
        self.nodal_loads.append((variable, load))

    def add_point_load(self, location_fn, force, variable=0):
        """Add a concentrated force at one mesh node.

        Parameters
        ----------
        location_fn : callable
            Function selecting the loaded node. It may accept ``point`` or
            ``(point, index)``.
        force : ndarray or callable
            Physical force vector. A callable receives the node coordinate,
            which allows it to read values updated by ``set_params``.
        variable : int
            Index of the finite-element variable receiving the force.

        Returns
        -------
        int
            Selected node index.
        """
        if not 0 <= variable < self.num_vars:
            raise ValueError(f"Invalid finite-element variable index: {variable}")

        fe = self.fes[variable]
        num_args = location_fn.__code__.co_argcount
        if num_args == 1:
            fn = lambda point, index: location_fn(point)
        elif num_args == 2:
            fn = location_fn
        else:
            raise ValueError(
                "location_fn must accept either (point) or (point, index)"
            )

        selected = jax.vmap(fn)(fe.points, np.arange(fe.num_total_nodes))
        node_inds = onp.asarray(np.argwhere(selected).reshape(-1))
        if len(node_inds) != 1:
            raise ValueError(
                "A point load must select exactly one node; "
                f"selected {len(node_inds)} nodes."
            )

        node_ind = int(node_inds[0])

        def nodal_load(points):
            value = force(points[node_ind]) if callable(force) else force
            value = np.asarray(value)
            expected_shape = (fe.vec,)
            if value.shape != expected_shape:
                raise ValueError(
                    f"Point force has shape {value.shape}; expected {expected_shape}."
                )
            return np.zeros((fe.num_total_nodes, fe.vec)).at[node_ind].set(value)

        self.add_nodal_load(nodal_load, variable)
        return node_ind

    def _add_nodal_loads(self, res_list):
        res_list = list(res_list)
        for variable, load in getattr(self, "nodal_loads", []):
            fe = self.fes[variable]
            value = load(fe.points) if callable(load) else load
            value = np.asarray(value)
            expected_shape = (fe.num_total_nodes, fe.vec)
            if value.shape != expected_shape:
                raise ValueError(
                    f"Nodal load has shape {value.shape}; expected {expected_shape}."
                )
            res_list[variable] = res_list[variable] - value
        return res_list

    def compute_residual(self, sol_list):
        """Assemble the residual and add the registered nodal loads."""
        res_list = super().compute_residual(sol_list)
        return self._add_nodal_loads(res_list)

    def newton_update(self, sol_list):
        """Assemble Newton data and add the registered nodal loads."""
        res_list = super().newton_update(sol_list)
        return self._add_nodal_loads(res_list)
