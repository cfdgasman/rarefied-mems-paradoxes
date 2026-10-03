"""Nonlinear Shakhov (S-model) kinetic equation in a 2D box with diffuse walls.

    d f/dt + v . grad f = nu (f_S - f),     nu = p / mu(T),   mu = mu_ref T^omega

The flow is planar, so the out-of-plane velocity v_z is integrated out (Chu
reduction) and two functions of (x, y, v_x, v_y) are carried:

    phi = int f dv_z,      psi = int v_z^2 f dv_z.

Units: m = k = 1, reference density and temperature 1, so velocities are in
units of sqrt(k T0 / m) and the box height is 1.

Space: finite volumes, second-order upwind (minmod-limited MUSCL), first order
in the cell next to a wall.  Time: forward Euler for transport and an implicit
(unconditionally stable) BGK-type relaxation step using the moments after
transport.  Walls: diffuse reflection; the re-emitted Maxwellian has the local
wall temperature and velocity and a density that returns exactly the incoming
mass flux, so the box conserves mass to round-off.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numba import njit, prange

MU_HS = 5.0 * math.sqrt(2.0 * math.pi) / 16.0  # mu = MU_HS * n lambda sqrt(kT/m) for hard spheres


def knudsen_to_delta(kn: float) -> float:
    """delta = p H / (mu v0) with v0 = sqrt(2kT/m), for the hard-sphere mean free path."""
    return 1.0 / (MU_HS * kn * math.sqrt(2.0))


@dataclass
class VelocityGrid:
    n: int = 28
    vmax: float = 5.5

    def __post_init__(self):
        v = np.linspace(-self.vmax, self.vmax, self.n)
        w1 = np.full(self.n, v[1] - v[0])
        w1[[0, -1]] *= 0.5  # trapezoid: spectrally accurate for a decayed Gaussian
        vx, vy = np.meshgrid(v, v, indexing="ij")
        wx, wy = np.meshgrid(w1, w1, indexing="ij")
        self.vx = vx.ravel().copy()
        self.vy = vy.ravel().copy()
        self.w = (wx * wy).ravel().copy()


@dataclass
class Box:
    """Rectangle [0, Lx] x [0, 1] with four diffuse walls.

    Wall data are given per boundary face: temperature and tangential velocity.
    """
    nx: int
    ny: int
    lx: float = 1.0
    kn: float = 1.0
    omega: float = 0.81          # argon, variable-hard-sphere exponent
    pr: float = 2.0 / 3.0
    vel: VelocityGrid = field(default_factory=VelocityGrid)
    t_bottom: np.ndarray | None = None
    t_top: np.ndarray | None = None
    t_left: np.ndarray | None = None
    t_right: np.ndarray | None = None
    u_top: float = 0.0           # lid velocity along +x

    def __post_init__(self):
        self.dx = self.lx / self.nx
        self.dy = 1.0 / self.ny
        self.xc = (np.arange(self.nx) + 0.5) * self.dx
        self.yc = (np.arange(self.ny) + 0.5) * self.dy
        one_x, one_y = np.ones(self.nx), np.ones(self.ny)
        self.t_bottom = one_x if self.t_bottom is None else np.asarray(self.t_bottom, float)
        self.t_top = one_x if self.t_top is None else np.asarray(self.t_top, float)
        self.t_left = one_y if self.t_left is None else np.asarray(self.t_left, float)
        self.t_right = one_y if self.t_right is None else np.asarray(self.t_right, float)
        self.mu_ref = MU_HS * self.kn  # box height is the reference length
        nv = len(self.vel.w)
        self.phi = np.empty((self.nx, self.ny, nv))
        self.psi = np.empty_like(self.phi)
        self.t = 0.0
        self.set_equilibrium()

    # ------------------------------------------------------------------ setup
    def set_equilibrium(self, temperature=None):
        """Gas at rest, uniform pressure 1; temperature field optional."""
        T = np.ones((self.nx, self.ny)) if temperature is None else temperature
        n = 1.0 / T
        v2 = self.vel.vx**2 + self.vel.vy**2
        m = (n / (2 * np.pi * T))[..., None] * np.exp(-v2[None, None, :] / (2 * T[..., None]))
        self.phi[:] = m
        self.psi[:] = m * T[..., None]
        self.t = 0.0

    def stable_dt(self, cfl: float = 0.45) -> float:
        vm = self.vel.vmax
        return cfl / (vm / self.dx + vm / self.dy)

    # ----------------------------------------------------------------- update
    def advance(self, t_end: float, cfl: float = 0.45):
        dt = self.stable_dt(cfl)
        nsteps = max(1, int(math.ceil((t_end - self.t) / dt - 1e-9)))
        dt = (t_end - self.t) / nsteps
        v = self.vel
        _run(self.phi, self.psi, v.vx, v.vy, v.w, self.dx, self.dy, dt, nsteps,
             self.t_bottom, self.t_top, self.t_left, self.t_right, self.u_top,
             self.mu_ref, self.omega, self.pr)
        self.t = t_end

    # ---------------------------------------------------------------- moments
    def moments(self):
        """n, ux, uy, T, p, qx, qy on the cell centres."""
        v = self.vel
        n = self.phi @ v.w
        ux = (self.phi @ (v.w * v.vx)) / n
        uy = (self.phi @ (v.w * v.vy)) / n
        e = 0.5 * (self.phi @ (v.w * (v.vx**2 + v.vy**2)) + self.psi @ v.w)
        T = (2.0 / 3.0) * (e / n - 0.5 * (ux**2 + uy**2))
        cx = v.vx[None, None, :] - ux[..., None]
        cy = v.vy[None, None, :] - uy[..., None]
        en = 0.5 * ((cx**2 + cy**2) * self.phi + self.psi)
        qx = np.einsum("ijk,k->ij", cx * en, v.w)
        qy = np.einsum("ijk,k->ij", cy * en, v.w)
        return dict(n=n, ux=ux, uy=uy, T=T, p=n * T, qx=qx, qy=qy)

    @property
    def mass(self) -> float:
        return float((self.phi @ self.vel.w).sum() * self.dx * self.dy)


# ====================================================================== kernels
@njit(inline="always")
def _minmod(a, b):
    if a * b <= 0.0:
        return 0.0
    return a if abs(a) < abs(b) else b


@njit(parallel=True, cache=True)
def _transport(phi, psi, vx, vy, w, dx, dy, dt, tb, tt, tl, tr, utop, out_phi, out_psi):
    nx, ny, nv = phi.shape
    # --- wall re-emission densities (one per wall face), from the incoming mass flux
    nb = np.empty(nx)
    ntp = np.empty(nx)
    nl = np.empty(ny)
    nr = np.empty(ny)
    for i in prange(nx):
        fin_b = 0.0
        fin_t = 0.0
        out_b = 0.0
        out_t = 0.0
        for k in range(nv):
            if vy[k] < 0.0:
                fin_b += -vy[k] * w[k] * phi[i, 0, k]
                out_t += -vy[k] * w[k] * math.exp(-((vx[k] - utop) ** 2 + vy[k] ** 2) / (2 * tt[i])) / (2 * math.pi * tt[i])
            else:
                fin_t += vy[k] * w[k] * phi[i, ny - 1, k]
                out_b += vy[k] * w[k] * math.exp(-(vx[k] ** 2 + vy[k] ** 2) / (2 * tb[i])) / (2 * math.pi * tb[i])
        nb[i] = fin_b / out_b
        ntp[i] = fin_t / out_t
    for j in prange(ny):
        fin_l = 0.0
        fin_r = 0.0
        out_l = 0.0
        out_r = 0.0
        for k in range(nv):
            if vx[k] < 0.0:
                fin_l += -vx[k] * w[k] * phi[0, j, k]
                out_r += -vx[k] * w[k] * math.exp(-(vx[k] ** 2 + vy[k] ** 2) / (2 * tr[j])) / (2 * math.pi * tr[j])
            else:
                fin_r += vx[k] * w[k] * phi[nx - 1, j, k]
                out_l += vx[k] * w[k] * math.exp(-(vx[k] ** 2 + vy[k] ** 2) / (2 * tl[j])) / (2 * math.pi * tl[j])
        nl[j] = fin_l / out_l
        nr[j] = fin_r / out_r

    lx = dt / dx
    ly = dt / dy
    for i in prange(nx):
        for j in range(ny):
            for k in range(nv):
                out_phi[i, j, k] = _update(phi, i, j, k, vx[k], vy[k], 1.0, 0, lx, ly,
                                           nb, ntp, nl, nr, tb, tt, tl, tr, utop)
                out_psi[i, j, k] = _update(psi, i, j, k, vx[k], vy[k], 1.0, 1, lx, ly,
                                           nb, ntp, nl, nr, tb, tt, tl, tr, utop)


@njit(inline="always")
def _wall(nw, tw, a, b, s):
    """Re-emitted reduced distribution: phi_w = n_w M(T_w), psi_w = T_w phi_w."""
    m = nw * math.exp(-(a * a + b * b) / (2.0 * tw)) / (2.0 * math.pi * tw)
    return m * tw if s == 1 else m


@njit(inline="always")
def _update(f, i, j, k, a, b, _one, s, lx, ly, nb, ntp, nl, nr, tb, tt, tl, tr, utop):
    """Forward-Euler finite-volume update of one discrete velocity in one cell."""
    nx, ny = f.shape[0], f.shape[1]
    c = f[i, j, k]
    # ---------------- x direction: values on the right (i+1/2) and left (i-1/2) faces
    if a > 0.0:
        if i == 0 or i == nx - 1:
            fr = c
        else:
            fr = c + 0.5 * _minmod(c - f[i - 1, j, k], f[i + 1, j, k] - c)
        if i == 0:
            fl = _wall(nl[j], tl[j], a, b, s)
        elif i == 1:
            fl = f[0, j, k]
        else:
            d = f[i - 1, j, k]
            fl = d + 0.5 * _minmod(d - f[i - 2, j, k], c - d)
    else:
        if i == nx - 1:
            fr = _wall(nr[j], tr[j], a, b, s)
        elif i == nx - 2:
            fr = f[nx - 1, j, k]
        else:
            d = f[i + 1, j, k]
            fr = d - 0.5 * _minmod(d - c, f[i + 2, j, k] - d)
        if i == 0 or i == nx - 1:
            fl = c
        else:
            fl = c - 0.5 * _minmod(c - f[i - 1, j, k], f[i + 1, j, k] - c)
    # ---------------- y direction: top (j+1/2) and bottom (j-1/2) faces
    if b > 0.0:
        if j == 0 or j == ny - 1:
            ft = c
        else:
            ft = c + 0.5 * _minmod(c - f[i, j - 1, k], f[i, j + 1, k] - c)
        if j == 0:
            fb = _wall(nb[i], tb[i], a, b, s)
        elif j == 1:
            fb = f[i, 0, k]
        else:
            d = f[i, j - 1, k]
            fb = d + 0.5 * _minmod(d - f[i, j - 2, k], c - d)
    else:
        if j == ny - 1:
            ft = _wall(ntp[i], tt[i], a - utop, b, s)
        elif j == ny - 2:
            ft = f[i, ny - 1, k]
        else:
            d = f[i, j + 1, k]
            ft = d - 0.5 * _minmod(d - c, f[i, j + 2, k] - d)
        if j == 0 or j == ny - 1:
            fb = c
        else:
            fb = c - 0.5 * _minmod(c - f[i, j - 1, k], f[i, j + 1, k] - c)
    return c - lx * a * (fr - fl) - ly * b * (ft - fb)


@njit(parallel=True, cache=True)
def _relax(phi, psi, vx, vy, w, dt, mu_ref, omega, pr):
    nx, ny, nv = phi.shape
    for i in prange(nx):
        for j in range(ny):
            n = 0.0
            mx = 0.0
            my = 0.0
            e = 0.0
            for k in range(nv):
                f = w[k] * phi[i, j, k]
                n += f
                mx += f * vx[k]
                my += f * vy[k]
                e += 0.5 * (f * (vx[k] ** 2 + vy[k] ** 2) + w[k] * psi[i, j, k])
            ux = mx / n
            uy = my / n
            T = (2.0 / 3.0) * (e / n - 0.5 * (ux * ux + uy * uy))
            qx = 0.0
            qy = 0.0
            for k in range(nv):
                cx = vx[k] - ux
                cy = vy[k] - uy
                en = 0.5 * ((cx * cx + cy * cy) * phi[i, j, k] + psi[i, j, k]) * w[k]
                qx += cx * en
                qy += cy * en
            p = n * T
            nu = p / (mu_ref * T**omega)
            a = nu * dt
            pre = n / (2.0 * math.pi * T)
            # normalise the discrete Maxwellian so that relaxation conserves mass exactly
            s0 = 0.0
            for k in range(nv):
                s0 += w[k] * math.exp(-((vx[k] - ux) ** 2 + (vy[k] - uy) ** 2) / (2.0 * T))
            pre = n / s0
            for k in range(nv):
                cx = vx[k] - ux
                cy = vy[k] - uy
                c2 = cx * cx + cy * cy
                fm = pre * math.exp(-c2 / (2.0 * T))
                A = (1.0 - pr) * (cx * qx + cy * qy) / (5.0 * p * T)
                fs = fm * (1.0 + A * (c2 / T - 4.0))
                gs = fm * (T + A * (c2 - 2.0 * T))
                phi[i, j, k] = (phi[i, j, k] + a * fs) / (1.0 + a)
                psi[i, j, k] = (psi[i, j, k] + a * gs) / (1.0 + a)


@njit(cache=True)
def _run(phi, psi, vx, vy, w, dx, dy, dt, nsteps, tb, tt, tl, tr, utop, mu_ref, omega, pr):
    bp = np.empty_like(phi)
    bs = np.empty_like(psi)
    for _ in range(nsteps):
        _transport(phi, psi, vx, vy, w, dx, dy, dt, tb, tt, tl, tr, utop, bp, bs)
        phi[:] = bp
        psi[:] = bs
        _relax(phi, psi, vx, vy, w, dt, mu_ref, omega, pr)
