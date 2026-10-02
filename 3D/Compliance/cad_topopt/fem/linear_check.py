"""
Stage 4: a small linear-elastic solve on the voxel mesh, to verify the
CAD-face -> node-set wiring before trusting it to the nonlinear optimizer.

This is deliberately a *different* solver from top3d_cf.py (scikit-fem
instead of the hand-written H8/neo-Hookean routine), so a mesh/BC bug and a
solver bug are unlikely to cancel out and look like a pass. All the actual
finite-element work -- shape functions, quadrature, the elasticity bilinear
form, the sparse solve -- comes from scikit-fem; nothing here reimplements
FE math, only wires the voxel mesh and boundary conditions into it.

Boundary conditions are the very same DofBCs object Stage 5 uses (built by
mesh/boundary.py), applied linearly and in one step instead of incrementally
with a nonlinear Newton solve:
    CLAMP  -> Dirichlet, u = 0 on the clamped components
    INPUT  -> Dirichlet, each input group's full-stroke displacement along
              its own direction
    OUTPUT -> grounded springs on each output group's own axis (added to
              the diagonal, exactly how top3d_cf.py's `ks` spring works)
It also reports Fout = out_w @ u, the same output measure the optimiser
uses, so its sign previews which way auto-Ftar will point.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from skfem import Basis, ElementHex1, ElementVector, MeshHex, condense, solve
from skfem.models.elasticity import linear_elasticity

from mesh.boundary import DofBCs
from mesh.voxelize import VoxelGrid

# Permutation from our own H8 corner order (mesh/voxelize.py's
# _CORNER_OFFSETS, shared with top3d_cf.py) into scikit-fem's MeshHex local
# node order. Found empirically (see project notes): scikit-fem does not use
# the common VTK_HEXAHEDRON ordering. Verified by checking that the *total*
# assembled volume over the whole mesh matches the voxel count exactly
# (n_active * h**3), not just that one cube comes out non-degenerate.
_SKFEM_PERM = [0, 1, 3, 4, 2, 5, 7, 6]


@dataclass
class LinearCheckResult:
    u: np.ndarray              # (n_used_nodes, 3) displacement, grid node numbering
    reaction: np.ndarray       # (n_used_nodes, 3) nodal out-of-balance force
    max_disp: float
    fout: float                # out_w @ u -- the optimiser's output measure, N
    # (group, mean displacement of its nodes along its own signed direction)
    group_disp: list


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


def run(grid: VoxelGrid, bcs: DofBCs, E: float, nu: float) -> LinearCheckResult:
    mesh, used_idx, new_of = _compact_mesh(grid)
    basis = Basis(mesh, ElementVector(ElementHex1()))
    dofs = basis.nodal_dofs  # (3, n_used_nodes): dofs[component, node]

    def to_skfem(grid_dofs: np.ndarray) -> np.ndarray:
        """Background-lattice dof (3*node + axis) -> scikit-fem dof."""
        node, comp = np.divmod(grid_dofs, 3)
        return dofs[comp, new_of[node]]

    mu = E / (2 * (1 + nu))
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    K = linear_elasticity(lam, mu).assemble(basis).tocsr()

    f = np.zeros(basis.N)

    input_dofs = to_skfem(bcs.in_dofs)
    x0 = np.zeros(basis.N)
    x0[input_dofs] = bcs.in_vals
    # condense() requires one flat ndarray of dof indices (a plain Python
    # list raises inside skfem). mesh/boundary.py has already rejected any
    # dof that is both clamped and driven, so this union is disjoint.
    dirichlet = np.unique(np.concatenate([to_skfem(bcs.fixed), input_dofs]))

    output_dofs = to_skfem(bcs.out_dofs)
    K = (K + sp.csr_matrix((bcs.out_k, (output_dofs, output_dofs)),
                           shape=K.shape)).tocsr()

    u = solve(*condense(K, f, x=x0, D=dirichlet))

    reaction_flat = K @ u - f
    u_grid = np.zeros((grid.node_xyz.shape[0], 3))
    reaction_grid = np.zeros((grid.node_xyz.shape[0], 3))
    for c in range(3):
        u_grid[used_idx, c] = u[dofs[c]]
        reaction_grid[used_idx, c] = reaction_flat[dofs[c]]

    group_disp = [(g, float(g.sign * u_grid[nodes, g.axis].mean()))
                  for g, nodes in bcs.groups]

    return LinearCheckResult(
        u=u_grid, reaction=reaction_grid,
        max_disp=float(np.linalg.norm(u_grid, axis=1).max()),
        fout=float(bcs.out_w @ u[output_dofs]),
        group_disp=group_disp)
