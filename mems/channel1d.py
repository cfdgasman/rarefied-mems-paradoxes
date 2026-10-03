"""Linearised BGK / Shakhov (S-model) flow in a plane micro-channel.

The gas fills -1/2 < x < 1/2 between two diffuse walls; a small pressure
gradient X_P = dln p/dy or temperature gradient X_T = dln T/dy acts along the
channel (y).  With f = f0 (1 + h), h = c_y * Phi and the transverse velocities
integrated out, two reduced functions Y(x, c) and Z(x, c) obey

    c dY/dx + d Y = d [u + (2/15) a q (c^2 - 1/2)] - 1/2 [X_P + (c^2 - 1/2) X_T]
    c dZ/dx + d Z = d [2u + (4/15) a q (c^2 + 1/2)] -     [X_P + (c^2 + 1/2) X_T]

    u = <Y>,   q = <(c^2 - 5/2) Y + Z>,   <g> = pi^-1/2 int g exp(-c^2) dc

with d = delta = p H / (mu v0) the rarefaction parameter, v0 = sqrt(2kT/m),
a = 1 for the S-model (Pr = 2/3) and a = 0 for BGK (Pr = 1).  Velocities are
in units of v0, lengths in units of H.  Walls re-emit the local equilibrium,
so Y = Z = 0 for molecules leaving a wall.

The equations are linear in (u, q), so the discrete map (u, q) -> (u, q) given
by one transport sweep is assembled column by column and the fixed point is
found with one dense solve; no iteration, no convergence tolerance.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def velocity_quadrature(n_per_panel: int = 16, c_min: float = 1e-6, c_max: float = 6.0):
    """Half-range nodes c > 0 and weights for pi^-1/2 int exp(-c^2) g dc.

    Gauss-Legendre on geometric panels, because at small delta the molecules with
    c -> 0 fly far and the integrand varies on every scale down to c ~ delta.
    """
    edges = np.concatenate([[0.0], np.geomspace(c_min, 1.0, 13), np.linspace(1.0, c_max, 6)[1:]])
    xg, wg = np.polynomial.legendre.leggauss(n_per_panel)
    c, w = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        c.append(0.5 * (b - a) * xg + 0.5 * (b + a))
        w.append(0.5 * (b - a) * wg)
    c = np.concatenate(c)
    w = np.concatenate(w) * np.exp(-c**2) / np.sqrt(np.pi)
    return c, w


def wall_clustered_grid(n: int, delta: float):
    """Cell faces in [-1/2, 1/2], clustered at the walls when the gas is dense."""
    s = np.linspace(-1.0, 1.0, n + 1)
    beta = min(3.0, 0.5 + 0.6 * np.log1p(delta))  # Knudsen layer is ~1/delta thick
    return 0.5 * np.tanh(beta * s) / np.tanh(beta)


@dataclass
class ChannelSolution:
    delta: float
    model: str
    x: np.ndarray          # cell centres
    dx: np.ndarray
    u: np.ndarray          # velocity profile (units of v0 per unit gradient)
    q: np.ndarray          # heat-flux profile (units of p v0 per unit gradient)

    @property
    def flow_rate(self) -> float:
        """G = 2 int u dx (the 2 makes the free-molecular limits classical)."""
        return 2.0 * float(np.sum(self.u * self.dx))

    @property
    def heat_flow(self) -> float:
        return 2.0 * float(np.sum(self.q * self.dx))


def _cell_factors(delta, dx, c):
    """Exact characteristic integration over one cell with a constant source S.

    In a cell Y relaxes towards Y* = S/delta over the optical depth
    kappa = delta dx / |c|:  Y_out = Y_in e + Y* (1 - e)  and the cell average is
    Y* + (Y_in - Y*) g,  e = exp(-kappa),  g = (1 - e)/kappa.
    """
    kappa = delta * dx[:, None] / c[None, :]
    e = np.exp(-kappa)
    g = np.where(kappa > 1e-8, -np.expm1(-kappa) / np.maximum(kappa, 1e-300), 1.0 - 0.5 * kappa)
    return e, g


def solve_channel(delta: float, model: str = "shakhov", gradient: str = "P",
                  n_cells: int = 160, n_per_panel: int = 12) -> ChannelSolution:
    """Solve the linearised Poiseuille ('P') or thermal-creep ('T') problem."""
    if model not in ("bgk", "shakhov"):
        raise ValueError(model)
    a = 1.0 if model == "shakhov" else 0.0
    c, w = velocity_quadrature(n_per_panel)
    faces = wall_clustered_grid(n_cells, delta)
    x = 0.5 * (faces[1:] + faces[:-1])
    dx = np.diff(faces)
    n = n_cells
    c2 = c**2
    xp, xt = (1.0, 0.0) if gradient == "P" else (0.0, 1.0)
    e, g = _cell_factors(delta, dx, c)

    # Unknowns z = (u_1..u_n, q_1..q_n).  Column k of the affine map is its
    # response to the unit vector e_k; the last column is the driving term.
    nrhs = 2 * n + 1
    drive_y = -0.5 * (xp + (c2 - 0.5) * xt)
    drive_z = -(xp + (c2 + 0.5) * xt)
    ky = (2.0 / 15.0) * a * (c2 - 0.5)
    kz = (4.0 / 15.0) * a * (c2 + 0.5)
    wq = w * (c2 - 2.5)
    # Within a cell the source is linear, S(x) = S_i + S'_i (x - x_i), with the
    # slope from central differences of the neighbouring cell values.  The
    # exact solution of c Y' + d Y = S is then Y_p + (Y_in - Y_p(entry)) e^(-kappa)
    # with Y_p = S/d - c S'/d^2.  The c S'/d^2 term is what carries the
    # diffusion limit, so the scheme stays accurate when delta dx >> 1.
    D = np.zeros((n, n))
    for i in range(n):
        lo, hi = max(i - 1, 0), min(i + 1, n - 1)
        D[i, hi] += 1.0 / (x[hi] - x[lo])
        D[i, lo] -= 1.0 / (x[hi] - x[lo])
    u_new = np.zeros((n, nrhs))
    q_new = np.zeros((n, nrhs))
    m = len(c)
    for sgn in (1.0, -1.0):  # c > 0 swept from the lower wall, c < 0 from the upper
        cs = (sgn * c)[:, None]
        order = range(n) if sgn > 0 else range(n - 1, -1, -1)
        y_in = np.zeros((m, nrhs))
        z_in = np.zeros((m, nrhs))
        for i in order:
            sy = np.zeros((m, nrhs))
            sz = np.zeros((m, nrhs))
            sy[:, i] = delta
            sz[:, i] = 2.0 * delta
            sy[:, n + i] = delta * ky
            sz[:, n + i] = delta * kz
            sy[:, -1] = drive_y
            sz[:, -1] = drive_z
            dsy = np.zeros((m, nrhs))
            dsz = np.zeros((m, nrhs))
            dsy[:, :n] = delta * D[i][None, :]
            dsz[:, :n] = 2.0 * delta * D[i][None, :]
            dsy[:, n:2 * n] = delta * ky[:, None] * D[i][None, :]
            dsz[:, n:2 * n] = delta * kz[:, None] * D[i][None, :]
            h = 0.5 * dx[i]
            ei, gi = e[i][:, None], g[i][:, None]
            for s_, ds_, f_in, acc in ((sy, dsy, y_in, "y"), (sz, dsz, z_in, "z")):
                yc = s_ / delta - cs * ds_ / delta**2       # Y_p at the cell centre
                ya = yc - sgn * h * ds_ / delta             # ... at the entry face
                yb = yc + sgn * h * ds_ / delta             # ... at the exit face
                avg = yc + (f_in - ya) * gi
                out = yb + (f_in - ya) * ei
                if acc == "y":
                    u_new[i] += w @ avg
                    q_new[i] += wq @ avg
                    y_in = out
                else:
                    q_new[i] += w @ avg
                    z_in = out
    big = np.vstack([u_new, q_new])
    A, b = big[:, :-1], big[:, -1]
    z = np.linalg.solve(np.eye(2 * n) - A, b)
    return ChannelSolution(delta, model, x, dx, z[:n], z[n:])


def thermomolecular_exponent(delta: float, model: str = "shakhov", **kw) -> float:
    """gamma = G_T / G_P: in a closed channel  dln p / dln T = gamma(delta)."""
    gp = -solve_channel(delta, model, "P", **kw).flow_rate
    gt = solve_channel(delta, model, "T", **kw).flow_rate
    return gt / gp
