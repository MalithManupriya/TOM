"""
Stage 4: a small linear-elastic solve on the voxel mesh, to verify the
CAD-face -> node-set wiring before trusting it to the nonlinear optimizer.

This is deliberately a *different* solver from top3d_cf.py (scikit-fem
instead of the hand-written H8/neo-Hookean routine), so a mesh/BC bug and a
solver bug are unlikely to cancel out and look like a pass. All the actual
finite-element work -- shape functions, quadrature, the elasticity bilinear
form, the sparse solve -- comes from scikit-fem; nothing here reimplements
FE math, only wires the voxel mesh and boundary conditions into it.

Boundary conditions mirror what Stage 5 will do with top3d_cf.py, so this
check exercises the same roles, just linearly and in one step instead of
incrementally with a nonlinear Newton solve:
    CLAMP  -> Dirichlet, u = 0 in all 3 components
    INPUT  -> Dirichlet, prescribed displacement in one component
    OUTPUT -> grounded spring in one component (added to the diagonal,
              exactly how top3d_cf.py's own `ks` output spring works)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from skfem import Basis, ElementHex1, ElementVector, MeshHex, condense, solve
from skfem.models.elasticity import linear_elasticity

from mesh.voxelize import VoxelGrid

# Permutation from our own H8 corner order (mesh/voxelize.py's
# _CORNER_OFFSETS, shared with top3d_cf.py) into scikit-fem's MeshHex local
# node order. Found empirically (see project notes): scikit-fem does not use
# the common VTK_HEXAHEDRON ordering. Verified by checking that the *total*
# assembled volume over the whole mesh matches the voxel count exactly
# (n_active * h**3), not just that one cube comes out non-degenerate.
_SKFEM_PERM = [0, 1, 3, 4, 2, 5, 7, 6]

_AXIS = {"x": 0, "y": 1, "z": 2}


@dataclass
class LinearCheckResult:
    u: np.ndarray              # (n_used_nodes, 3) displacement, grid node numbering
    reaction: np.ndarray       # (n_used_nodes, 3) nodal out-of-balance force
    max_disp: float
    output_disp_mean: np.ndarray  # (3,) mean displacement at output nodes


def _compact_mesh(grid: VoxelGrid) -> tuple[MeshHex, np.ndarray, np.ndarray]:
    """Build a scikit-fem MeshHex using only the nodes an element touches.

    Returns (mesh, used_idx, new_of) where used_idx[new] is the grid node
    index for compact index `new`, and new_of[old] is the reverse map
    (-1 for a grid node that is not used by any active element).
    """
    used_idx = np.nonzero(grid.used_node)[0]
    new_of = -np.ones(grid.node_xyz.shape[0], dtype=int)
    new_of[used_idx] = np.arange(len(used_idx))

    verts = grid.node_xyz[used_idx]
    conn = new_of[grid.edof[:, _SKFEM_PERM]]
    mesh = MeshHex(verts.T, conn.T)
    return mesh, used_idx, new_of


def run(grid: VoxelGrid, bc_nodes: dict[str, np.ndarray], clamp: dict,
        input_: dict, output: dict, E: float, nu: float) -> LinearCheckResult:
    mesh, used_idx, new_of = _compact_mesh(grid)
    basis = Basis(mesh, ElementVector(ElementHex1()))
    dofs = basis.nodal_dofs  # (3, n_used_nodes): dofs[component, node]

    mu = E / (2 * (1 + nu))
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    K = linear_elasticity(lam, mu).assemble(basis).tocsr()

    f = np.zeros(basis.N)

    clamp_new = new_of[bc_nodes["clamp"]]
    dirichlet = [dofs[:, clamp_new].ravel()]

    in_axis = _AXIS[input_["direction"]]
    input_new = new_of[bc_nodes["input"]]
    input_dofs = dofs[in_axis, input_new]
    dirichlet.append(input_dofs)
    x0 = np.zeros(basis.N)
    x0[input_dofs] = input_["displacement"]
    # condense() requires one flat ndarray of dof indices (a plain Python
    # list raises inside skfem), and np.unique also protects against a node
    # Stage 3 already warned about sharing two roles (e.g. clamp & output on
    # a shared edge) turning into a duplicate/conflicting entry here.
    dirichlet = np.unique(np.concatenate(dirichlet))

    out_axis = _AXIS[output["direction"]]
    output_new = new_of[bc_nodes["output"]]
    output_dofs = dofs[out_axis, output_new]
    k_each = output["spring_stiffness"] / max(len(output_dofs), 1)
    K = (K + sp.diags(np.bincount(output_dofs, minlength=basis.N) * k_each)).tocsr()

    u = solve(*condense(K, f, x=x0, D=dirichlet))

    reaction_flat = K @ u - f
    u_grid = np.zeros((grid.node_xyz.shape[0], 3))
    reaction_grid = np.zeros((grid.node_xyz.shape[0], 3))
    for c in range(3):
        u_grid[used_idx, c] = u[dofs[c]]
        reaction_grid[used_idx, c] = reaction_flat[dofs[c]]

    return LinearCheckResult(
        u=u_grid, reaction=reaction_grid,
        max_disp=float(np.linalg.norm(u_grid, axis=1).max()),
        output_disp_mean=u_grid[bc_nodes["output"]].mean(axis=0))
