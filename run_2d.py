"""Transient 2D kinetic simulations: lid-driven micro-cavity and closed Knudsen-pump channel.

    python run_2d.py cavity 1.0      # one Knudsen number, writes data/cavity_kn1.0.npz
    python run_2d.py pump 0.5        # closed channel with a wall-temperature ramp
    python run_2d.py plots           # figures and GIFs from the saved data
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
DOCS = os.path.join(HERE, "docs")

# Lid speed 0.2 sqrt(kT0/m): about 48 m/s in argon at 273 K, the regime of
# John, Gu & Emerson (2010); the flow is slow (Mach 0.15), heating is small.
U_LID = 0.2
T_COLD, T_HOT = 1.0, 1.5     # Knudsen-pump wall temperatures at the two ends
PUMP_LX = 6.0


def _record(box, t_frames, path, extra):
    from mems.kinetic2d import Box  # noqa: F401

    keys = ("n", "ux", "uy", "T", "p", "qx", "qy")
    frames = {k: [] for k in keys}
    m0 = box.mass
    t0 = time.time()
    for t in t_frames:
        box.advance(t)
        m = box.moments()
        for k in keys:
            frames[k].append(m[k].astype(np.float32))
        if len(frames["T"]) > 1:
            dT = np.abs(frames["T"][-1] - frames["T"][-2]).max()
            du = np.abs(frames["ux"][-1] - frames["ux"][-2]).max()
            print(f"t={t:7.2f}  max|dT|={dT:.2e}  max|du|={du:.2e}  "
                  f"mass drift={box.mass / m0 - 1:.1e}  wall {time.time() - t0:6.0f}s", flush=True)
    final = box.moments()
    np.savez_compressed(path, t=np.asarray(t_frames), x=box.xc, y=box.yc, lx=box.lx,
                        kn=box.kn, mass_drift=box.mass / m0 - 1,
                        **{k: np.asarray(v) for k, v in frames.items()},
                        **{"final_" + k: v for k, v in final.items()}, **extra)
    print("saved", path)


def cavity(kn: float, n: int = 48, nv: int = 28, t_end: float | None = None):
    from mems.kinetic2d import Box, VelocityGrid

    if t_end is None:  # a few viscous diffusion times H^2 rho/mu, at least 12 acoustic times
        t_end = float(np.clip(3.0 / (0.783 * kn), 12.0, 40.0))
    box = Box(n, n, kn=kn, u_top=U_LID, vel=VelocityGrid(nv, 6.0))
    t_frames = np.round(np.arange(0.25, t_end + 1e-9, 0.25), 6)
    _record(box, t_frames, os.path.join(DATA, f"cavity_kn{kn:g}.npz"),
            dict(u_lid=U_LID, nv=nv))


def pump(kn: float, nx: int = 120, ny: int = 20, nv: int = 28, t_end: float = 40.0):
    from mems.kinetic2d import Box, VelocityGrid

    xc = (np.arange(nx) + 0.5) * PUMP_LX / nx
    tw = T_COLD + (T_HOT - T_COLD) * xc / PUMP_LX
    box = Box(nx, ny, lx=PUMP_LX, kn=kn, vel=VelocityGrid(nv, 6.5),
              t_bottom=tw, t_top=tw, t_left=np.full(ny, T_COLD), t_right=np.full(ny, T_HOT))
    # start from uniform pressure and the wall temperature: the transient is the
    # pump building up its pressure difference
    box.set_equilibrium(np.repeat(tw[:, None], ny, axis=1))
    t_frames = np.round(np.arange(0.5, t_end + 1e-9, 0.5), 6)
    _record(box, t_frames, os.path.join(DATA, f"pump_kn{kn:g}.npz"), dict(nv=nv, t_wall=tw))


def cavity_refinement(kn: float = 1.0, t_end: float = 12.0):
    """Hot/cold spot temperatures and the cold->hot area fraction on three grids."""
    import csv

    from mems.kinetic2d import Box, VelocityGrid
    from plots2d import counter_gradient

    rows = []
    for n, nv in ((32, 28), (48, 28), (64, 28), (48, 40)):
        box = Box(n, n, kn=kn, u_top=U_LID, vel=VelocityGrid(nv, 6.0))
        box.advance(t_end)
        m = box.moments()
        _, _, frac = counter_gradient(m["T"], m["qx"], m["qy"], box.xc, box.yc)
        rows.append([n, nv, f"{(m['T'].max() - 1) / U_LID**2:.4f}", f"{(m['T'].min() - 1) / U_LID**2:.4f}",
                     f"{frac:.3f}", f"{np.hypot(m['ux'], m['uy'])[n // 2, n // 2] / U_LID:.4f}"])
        print(rows[-1], flush=True)
    with open(os.path.join(DATA, f"cavity_refinement_kn{kn:g}.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["cells per side", "velocities per axis", "(Tmax-T0)/U^2", "(Tmin-T0)/U^2",
                     "frac_cold_to_hot", "|u|/U at centre"])
        wr.writerows(rows)


if __name__ == "__main__":
    os.makedirs(DATA, exist_ok=True)
    what = sys.argv[1]
    if what == "cavity":
        cavity(float(sys.argv[2]), *[int(a) for a in sys.argv[3:5]])
    elif what == "pump":
        pump(float(sys.argv[2]))
    elif what == "refine":
        cavity_refinement(float(sys.argv[2]) if len(sys.argv) > 2 else 1.0)
    elif what == "plots":
        import plots2d
        plots2d.main()
