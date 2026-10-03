"""Figures and GIFs for the 2D cavity and Knudsen-pump runs (data/*.npz -> docs/)."""
from __future__ import annotations

import glob
import os

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
DOCS = os.path.join(HERE, "docs")
plt.rcParams.update({"font.size": 11, "figure.dpi": 110})
CMAP = "RdBu_r"


def load(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def grad(f, x, y):
    gx = np.gradient(f, x, axis=0)
    gy = np.gradient(f, y, axis=1)
    return gx, gy


def counter_gradient(T, qx, qy, x, y):
    """cos of the angle between q and -grad T, and the area fraction where heat runs cold -> hot."""
    gx, gy = grad(T, x, y)
    qn = np.hypot(qx, qy)
    gn = np.hypot(gx, gy)
    cos = -(qx * gx + qy * gy) / np.maximum(qn * gn, 1e-30)
    # ignore cells where either field is negligible (both vanish at the vortex centre)
    ok = (qn > 0.02 * qn.max()) & (gn > 0.02 * gn.max())
    return cos, ok, float(np.mean(cos[ok] < 0.0))


def _arrows(ax, x, y, qx, qy, every=3, color="k", scale=None):
    X, Y = np.meshgrid(x, y, indexing="ij")
    s = (slice(every // 2, None, every), slice(every // 2, None, every))
    mag = np.hypot(qx, qy)
    # direction arrows of equal length; the colour map already shows where it is weak
    ux, uy = qx / np.maximum(mag, 1e-30), qy / np.maximum(mag, 1e-30)
    return ax.quiver(X[s], Y[s], ux[s], uy[s], color=color, scale=scale or 1.0 / (every * (x[1] - x[0])) * 1.25,
                     scale_units="xy", width=0.004, headwidth=4, pivot="mid")


def grad13_heat_flux(T, p, n, x, y, kn, omega=0.81):
    """Steady linear Grad-13 heat flux, q = -kappa grad T + (3/2)(mu/rho) grad p.

    From the heat-flux balance (5/2) p grad(RT) + RT div(sigma) = -(2/3)(p/mu) q
    with div(sigma) = -grad p (slow flow), kappa = 15 mu / 4 (Pr = 2/3, m = k = 1).
    """
    from mems.kinetic2d import MU_HS

    mu = MU_HS * kn * T**omega
    tx, ty = grad(T, x, y)
    px, py = grad(p, x, y)
    return -3.75 * mu * tx + 1.5 * mu / n * px, -3.75 * mu * ty + 1.5 * mu / n * py


def alignment(ax_, ay_, bx, by, ok):
    c = (ax_ * bx + ay_ * by) / np.maximum(np.hypot(ax_, ay_) * np.hypot(bx, by), 1e-30)
    return float(np.mean(c[ok]))


# ===================================================================== cavity
def cavity_figures():
    runs = sorted(glob.glob(os.path.join(DATA, "cavity_kn*.npz")),
                  key=lambda p: float(os.path.basename(p)[9:-4]))
    if not runs:
        return []
    rows = []
    fig, axes = plt.subplots(2, len(runs), figsize=(5.6 * len(runs), 10.4), squeeze=False,
                             layout="constrained")
    for col, path in enumerate(runs):
        d = load(path)
        x, y, kn = d["x"], d["y"], float(d["kn"])
        T, qx, qy = d["final_T"], d["final_qx"], d["final_qy"]
        ux, uy = d["final_ux"], d["final_uy"]
        cos, ok, frac = counter_gradient(T, qx, qy, x, y)
        imax = np.unravel_index(np.argmax(T), T.shape)
        imin = np.unravel_index(np.argmin(T), T.shape)
        gx13, gy13 = grad13_heat_flux(T, d["final_p"], d["final_n"], x, y, kn)
        inner = np.zeros_like(ok)
        inner[3:-3, 3:-3] = True  # Knudsen layers excluded
        rows.append(dict(kn=kn, frac=frac, Tmin=T.min(), Tmax=T.max(), qmax=np.hypot(qx, qy).max(),
                         grad13_cos=alignment(qx, qy, gx13, gy13, ok & inner),
                         fourier_cos=alignment(qx, qy, -np.gradient(T, x, axis=0), -np.gradient(T, y, axis=1),
                                               ok & inner),
                         hot=(x[imax[0]], y[imax[1]]), cold=(x[imin[0]], y[imin[1]]),
                         drift=float(d["mass_drift"]), t=float(d["t"][-1])))
        a = axes[0, col]
        lim = np.abs(T - 1).max()
        cf = a.contourf(x, y, (T - 1).T / U_LID**2, 24, cmap=CMAP,
                        vmin=-lim / U_LID**2, vmax=lim / U_LID**2)
        a.streamplot(x, y, qx.T, qy.T, color="k", density=1.1, linewidth=0.8, arrowsize=0.9)
        a.plot(*rows[-1]["hot"], "o", ms=9, mfc="none", mec="darkred", mew=2)
        a.plot(*rows[-1]["cold"], "o", ms=9, mfc="none", mec="navy", mew=2)
        a.set_title(f"Kn = {kn:g}: temperature, heat-flux lines", pad=16)
        fig.colorbar(cf, ax=a, shrink=0.8, label=r"$(T-T_0)/(mU^2/k)$")
        a = axes[1, col]
        sp = np.hypot(ux, uy) / U_LID
        cf = a.contourf(x, y, cos.T, np.linspace(-1, 1, 21), cmap="PuOr")
        a.streamplot(x, y, ux.T, uy.T, color="0.25", density=1.0, linewidth=0.7, arrowsize=0.7)
        a.set_title(f"Kn = {kn:g}: cos(q, −∇T); cold → hot on {100 * frac:.0f}%", pad=16)
        fig.colorbar(cf, ax=a, shrink=0.8, label="cos θ   (+1 Fourier, −1 cold → hot)")
        for a in axes[:, col]:
            a.set_aspect("equal")
            a.set_xlim(0, 1)
            a.set_ylim(0, 1)
            a.annotate("", xy=(0.7, 1.025), xytext=(0.3, 1.025), xycoords="data",
                       arrowprops=dict(arrowstyle="->", lw=2, color="k"), annotation_clip=False)
    fig.suptitle("Lid-driven micro-cavity: lid at 0.2 √(kT₀/m), all walls at T₀")
    fig.savefig(os.path.join(DOCS, "cavity_heat.png"))
    plt.close(fig)

    # fraction vs Kn (only meaningful with several runs)
    if len(rows) < 2:
        cavity_gif(load(min(runs, key=lambda p: abs(np.log(float(os.path.basename(p)[9:-4]))))))
        return rows
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.9), layout="constrained")
    kns = [r["kn"] for r in rows]
    ma = U_LID / np.sqrt(5.0 / 3.0)
    a.semilogx(kns, [100 * r["frac"] for r in rows], "-o", color="#7b3294", lw=2)
    a.set_xlabel("Kn")
    a.set_ylabel("% of cavity with heat flowing\nfrom cold to hot")
    a.set_ylim(0, 100)
    a.grid(alpha=0.3)
    b.semilogx(kns, [r["fourier_cos"] for r in rows], "-s", color="#c0392b", lw=2,
               label=r"Fourier, $-\kappa\nabla T$")
    b.semilogx(kns, [r["grad13_cos"] for r in rows], "-o", color="#1f6fb4", lw=2,
               label=r"Grad-13, $-\kappa\nabla T + \frac{3}{2}\frac{\mu}{\rho}\nabla p$")
    b.axhline(0, color="k", lw=0.6)
    b.set_ylim(-1, 1)
    b.set_xlabel("Kn")
    b.set_ylabel("mean cos(q, model)")
    b.legend(fontsize=9, loc="center right")
    b.grid(alpha=0.3)
    for ax_ in (a, b):
        ax_.axvline(ma, color="0.5", ls="--")
        ax_.text(ma * 1.08, 0.05, "Kn = Ma", color="0.4", transform=ax_.get_xaxis_transform())
        ax_.set_xlim(0.05, 20)
    fig.savefig(os.path.join(DOCS, "cavity_fraction.png"))
    plt.close(fig)

    # transient GIF for Kn = 1 (or the closest run)
    path = min(runs, key=lambda p: abs(np.log(float(os.path.basename(p)[9:-4]))))
    cavity_gif(load(path))
    return rows


U_LID = 0.2
T_COLD, T_HOT = 1.0, 1.5


def cavity_gif(d):
    x, y, ts = d["x"], d["y"], d["t"]
    sel = np.arange(len(ts))
    if len(sel) > 64:
        sel = sel[: 64]  # the start-up is where the action is
    lim = np.abs(d["T"][sel] - 1).max() / U_LID**2
    levels = np.linspace(-lim, lim, 25)
    fig, ax = plt.subplots(figsize=(6.0, 5.6))
    fig.subplots_adjust(left=0.04, right=0.76, top=0.84, bottom=0.04)
    cax = fig.add_axes([0.80, 0.12, 0.03, 0.66])

    def draw(f):
        ax.clear()
        k = sel[f]
        T = (d["T"][k] - 1) / U_LID**2
        cf = ax.contourf(x, y, T.T, levels, cmap=CMAP, extend="both")
        _arrows(ax, x, y, d["qx"][k], d["qy"][k], every=4)
        ax.set_aspect("equal")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.annotate("", xy=(0.7, 1.03), xytext=(0.3, 1.03), arrowprops=dict(arrowstyle="->", lw=2),
                    annotation_clip=False)
        ax.text(0.5, 1.055, "moving lid", ha="center", fontsize=9, transform=ax.transData)
        _, ok, frac = counter_gradient(d["T"][k], d["qx"][k], d["qy"][k], x, y)
        ax.set_title(f"Kn = {float(d['kn']):g},  t = {ts[k]:.2f} H/√(kT₀/m)\n"
                     f"arrows: heat flux;  cold → hot over {100 * frac:.0f}% of the cavity", fontsize=10, pad=36)
        if f == 0:
            cb = fig.colorbar(cf, cax=cax, label=r"$(T-T_0)/(mU^2/k)$")
            cb.ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))

    anim = FuncAnimation(fig, draw, frames=len(sel))
    anim.save(os.path.join(DOCS, "cavity_transient.gif"), writer=PillowWriter(fps=8))
    plt.close(fig)


# ======================================================================= pump
def pump_figures():
    runs = sorted(glob.glob(os.path.join(DATA, "pump_kn*.npz")))
    out = []
    for path in runs:
        d = load(path)
        out.append(pump_one(d))
    return out


def pump_one(d):
    from scipy.interpolate import interp1d

    from mems.channel1d import solve_channel
    from mems.kinetic2d import MU_HS

    x, y, kn = d["x"], d["y"], float(d["kn"])
    T, p, ux, uy, qx, qy, n = (d["final_" + k] for k in ("T", "p", "ux", "uy", "qx", "qy", "n"))
    lx = float(d["lx"])
    tag = f"{kn:g}"

    # ---- validation against linearised theory in the middle third of the channel
    mid = (x > lx / 3) & (x < 2 * lx / 3)
    pbar = p.mean(axis=1)
    tw = d["t_wall"]
    dlnp = np.gradient(np.log(pbar), x)
    dlnT = np.gradient(np.log(tw), x)
    i0 = np.argmin(abs(x - lx / 2))
    omega = 0.81
    delta_loc = pbar / (MU_HS * kn * tw**omega * np.sqrt(2 * tw))
    sp = solve_channel(float(delta_loc[i0]), "shakhov", "P")
    st = solve_channel(float(delta_loc[i0]), "shakhov", "T")
    gamma = st.flow_rate / -sp.flow_rate
    measured = float(np.mean(dlnp[mid] / dlnT[mid]))
    # profile at mid-channel: u = v0 (X_P u_P + X_T u_T), X = H dln/dx, v0 = sqrt(2T)
    v0 = np.sqrt(2 * tw[i0])
    # closed channel: zero net flow fixes X_P = gamma X_T; pure prediction, nothing taken from the 2D run
    # except the local delta and the wall-temperature gradient
    u_theory = v0 * dlnT[i0] * (st.u + gamma * sp.u)
    u_shift = interp1d(sp.x + 0.5, u_theory, fill_value="extrapolate")(y)

    # ---- steady figure
    fig = plt.figure(figsize=(13, 7.4))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.25])
    a = fig.add_subplot(gs[0, :])
    cf = a.contourf(x, y, T.T, 24, cmap="coolwarm")
    X, Y = np.meshgrid(x, y, indexing="ij")
    s = (slice(2, None, 5), slice(1, None, 3))
    a.quiver(X[s], Y[s], qx[s], qy[s], color="k", width=0.0025, scale=None)
    a.set_aspect("equal")
    a.set_title(f"Knudsen pump, Kn = {kn:g}: temperature and heat flux (walls ramp {T_COLD:g} → {T_HOT:g} T₀)")
    fig.colorbar(cf, ax=a, pad=0.01, shrink=0.75, label="T / T₀")
    a = fig.add_subplot(gs[1, :])
    cf = a.contourf(x, y, (ux / np.sqrt(2 * T)).T, 24, cmap="PiYG")
    a.streamplot(x, y, ux.T, uy.T, color="k", density=[2.5, 0.8], linewidth=0.7, arrowsize=0.8)
    a.set_aspect("equal")
    a.set_title(f"velocity: creep and pressure-driven return flow (δ ≈ {delta_loc[i0]:.1f}: core → hot end, wall layers → cold end)")
    fig.colorbar(cf, ax=a, pad=0.01, label=r"$u_x / v_0$")
    a = fig.add_subplot(gs[2, 0])
    a.plot(x, pbar, color="#1f6fb4", lw=2)
    a.set_xlabel("x / H")
    a.set_ylabel("p / p₀ (cross-section mean)")
    a.set_title("pressure builds up at the hot end")
    a.grid(alpha=0.3)
    a = fig.add_subplot(gs[2, 1])
    a.plot(x[mid], dlnp[mid] / dlnT[mid], color="#1f6fb4", lw=2, label="2D nonlinear S-model")
    a.axhline(gamma, color="#e3742b", ls="--", lw=2, label=f"1D linear theory γ(δ={delta_loc[i0]:.2f})")
    a.axhline(0.5, color="0.5", ls=":", label="free molecular 1/2")
    a.set_ylim(0, 0.6)
    a.set_xlabel("x / H")
    a.set_ylabel(r"$d\ln p / d\ln T$")
    a.legend(fontsize=8)
    a.grid(alpha=0.3)
    a.set_title("thermomolecular exponent")
    a = fig.add_subplot(gs[2, 2])
    a.plot(ux[i0], y, "o", ms=4, color="#1f6fb4", label="2D nonlinear")
    a.plot(u_shift, y, "-", color="#e3742b", lw=2, label="1D linear theory, zero net flow")
    a.axvline(0, color="k", lw=0.6)
    a.set_xlabel(r"$u_x$ at mid-channel")
    a.set_ylabel("y / H")
    a.legend(fontsize=8)
    a.grid(alpha=0.3)
    a.set_title("velocity profile at x = L/2")
    fig.tight_layout()
    fig.savefig(os.path.join(DOCS, f"pump_kn{tag}.png"))
    plt.close(fig)

    # pressure ratio from the 1D theory: dp/p = gamma(delta(p, T)) dT/T along the wall ramp
    # (midpoint rule, local delta updated with the integrated pressure)
    def gam(dl):
        return (solve_channel(dl, "shakhov", "T", n_cells=80).flow_rate
                / -solve_channel(dl, "shakhov", "P", n_cells=80).flow_rate)

    def dloc(p, T):
        return p / (MU_HS * kn * T**omega * np.sqrt(2 * T))

    Ts = np.linspace(tw[0], tw[-1], 13)
    lp = np.log(pbar[0])
    for t0, t1 in zip(Ts[:-1], Ts[1:]):
        tm = 0.5 * (t0 + t1)
        lp_half = lp + gam(dloc(np.exp(lp), tm)) * np.log(tm / t0)
        lp += gam(dloc(np.exp(lp_half), tm)) * np.log(t1 / t0)
    ratio_1d = float(np.exp(lp) / pbar[0])

    pump_gif(d, tag)
    err_u = float(np.max(np.abs(ux[i0] - u_shift)) / np.max(np.abs(u_shift)))
    return dict(kn=kn, delta=float(delta_loc[i0]), gamma_1d=gamma, gamma_2d=measured,
                p_ratio=float(pbar[-1] / pbar[0]), p_ratio_1d=ratio_1d, err_u=err_u, drift=float(d["mass_drift"]),
                net_flux=float(np.sum(n[i0] * ux[i0]) / np.sum(np.abs(n[i0] * ux[i0]))))


def pump_gif(d, tag):
    x, y, ts = d["x"], d["y"], d["t"]
    sel = np.unique(np.concatenate([np.arange(0, min(len(ts), 20)), np.arange(20, len(ts), 2)]))
    fig, ax = plt.subplots(2, 1, figsize=(9.5, 4.6), gridspec_kw=dict(height_ratios=[1, 1]))
    fig.subplots_adjust(left=0.07, right=0.86, top=0.88, bottom=0.12, hspace=0.45)
    cax = fig.add_axes([0.88, 0.55, 0.015, 0.32])
    pmin = min(d["p"][k].mean(axis=1).min() for k in sel)
    pmax = max(d["p"][k].mean(axis=1).max() for k in sel)
    levels = np.linspace(T_COLD, T_HOT, 21)
    X, Y = np.meshgrid(x, y, indexing="ij")
    s = (slice(2, None, 5), slice(1, None, 3))

    def draw(f):
        k = sel[f]
        for a in ax:
            a.clear()
        cf = ax[0].contourf(x, y, d["T"][k].T, levels, cmap="coolwarm", extend="both")
        qx, qy = d["qx"][k], d["qy"][k]
        ax[0].quiver(X[s], Y[s], qx[s], qy[s], color="k", width=0.003, scale=0.25, scale_units="xy")
        ux, uy = d["ux"][k], d["uy"][k]
        ax[0].quiver(X[s], Y[s], ux[s], uy[s], color="#2ca02c", width=0.003, scale=0.05, scale_units="xy")
        ax[0].set_aspect("equal")
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ax[0].set_title(f"Knudsen pump, Kn = {tag}, t = {ts[k]:.1f}: black = heat flux, green = gas velocity",
                        fontsize=10)
        ax[1].plot(x, d["p"][k].mean(axis=1), color="#1f6fb4", lw=2)
        ax[1].set_ylim(pmin - 0.01, pmax + 0.01)
        ax[1].set_xlim(0, x[-1] + x[0])
        ax[1].set_xlabel("x / H   (cold end on the left, hot end on the right)")
        ax[1].set_ylabel("p / p₀")
        ax[1].grid(alpha=0.3)
        if f == 0:
            fig.colorbar(cf, cax=cax, label="T / T₀")

    anim = FuncAnimation(fig, draw, frames=len(sel))
    anim.save(os.path.join(DOCS, f"pump_kn{tag}.gif"), writer=PillowWriter(fps=8))
    plt.close(fig)


def overview():
    """Opening figure of the README: one real result per puzzle."""
    import csv

    cav = os.path.join(DATA, "cavity_kn1.npz")
    pmp = os.path.join(DATA, "pump_kn0.5.npz")
    if not (os.path.exists(cav) and os.path.exists(pmp)):
        return
    fig = plt.figure(figsize=(13, 9.6), layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1])

    a = fig.add_subplot(gs[0, 0])
    rows = [r for r in csv.DictReader(open(os.path.join(DATA, "channel_table.csv"))) if r["model"] == "shakhov"]
    dl = np.array([float(r["delta"]) for r in rows])
    gp = np.array([float(r["G_P"]) for r in rows])
    a.semilogx(dl, gp, "-o", ms=4, color="#1f6fb4")
    dd = np.geomspace(dl[0], dl[-1], 200)
    a.semilogx(dd, dd / 6, ":", color="0.45", label="Navier–Stokes, no slip")
    k = np.argmin(gp)
    a.plot(dl[k], gp[k], "o", ms=12, mfc="none", mec="#c0392b", mew=2)
    a.annotate("minimum", (dl[k], gp[k]), (dl[k] * 0.12, gp[k] - 0.9), color="#c0392b",
               arrowprops=dict(arrowstyle="->", color="#c0392b"))
    a.set_ylim(0, 4.5)
    a.set_xlabel(r"rarefaction  $\delta \approx 0.9/\mathrm{Kn}$")
    a.set_ylabel(r"reduced flow rate $G_P$")
    a.set_title("Knudsen paradox: the flow rate has a minimum", weight="bold")
    a.legend(fontsize=9, loc="upper center")
    a.grid(alpha=0.3)

    d = load(pmp)
    x, y = d["x"], d["y"]
    a = fig.add_subplot(gs[1, :])
    cf = a.contourf(x, y, d["final_T"].T, 24, cmap="coolwarm")
    a.set_xlim(0, float(d["lx"]))
    a.set_ylim(0, 1)
    a.streamplot(x, y, d["final_ux"].T, d["final_uy"].T, color="k", density=[2.2, 0.9],
                 linewidth=0.7, arrowsize=0.8)
    a.set_aspect("equal")
    a.set_xticks([])
    a.set_yticks([])
    a.set_title("Knudsen pump: no moving parts, pressure builds up at the hot end", weight="bold")
    fig.colorbar(cf, ax=a, pad=0.01, shrink=0.75, label="T / T₀")

    d = load(cav)
    x, y = d["x"], d["y"]
    a = fig.add_subplot(gs[0, 1])
    T = (d["final_T"] - 1) / U_LID**2
    lim = np.abs(T).max()
    cf = a.contourf(x, y, T.T, 24, cmap=CMAP, vmin=-lim, vmax=lim)
    a.streamplot(x, y, d["final_qx"].T, d["final_qy"].T, color="k", density=0.9, linewidth=0.8)
    a.set_aspect("equal")
    a.set_xticks([])
    a.set_yticks([])
    a.annotate("", xy=(0.7, 1.03), xytext=(0.3, 1.03), arrowprops=dict(arrowstyle="->", lw=2),
               annotation_clip=False)
    a.set_title("Lid-driven cavity: heat-flux lines run from cold to hot", weight="bold", pad=22)
    a.set_aspect("equal")
    a.set_xlim(0, 1)
    a.set_ylim(0, 1)
    a.set_box_aspect(1)
    fig.colorbar(cf, ax=a, shrink=0.9, label=r"$(T-T_0)/(mU^2/k)$")
    fig.savefig(os.path.join(DOCS, "overview.png"))
    plt.close(fig)


def main():
    import csv

    rows = cavity_figures()
    with open(os.path.join(DATA, "cavity_summary.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["Kn", "t_end", "T_min", "T_max", "hot_spot", "cold_spot", "frac_cold_to_hot",
                     "mean cos(q, Fourier)", "mean cos(q, Grad13)", "mass_drift"])
        for r in rows:
            wr.writerow([r["kn"], r["t"], f"{r['Tmin']:.5f}", f"{r['Tmax']:.5f}",
                         f"({r['hot'][0]:.2f},{r['hot'][1]:.2f})", f"({r['cold'][0]:.2f},{r['cold'][1]:.2f})",
                         f"{r['frac']:.3f}", f"{r['fourier_cos']:.3f}", f"{r['grad13_cos']:.3f}",
                         f"{r['drift']:.1e}"])
            print(r)
    prow = pump_figures()
    with open(os.path.join(DATA, "pump_summary.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["Kn", "delta_mid", "gamma_1D", "gamma_2D", "p_hot/p_cold 2D", "p_hot/p_cold 1D theory",
                     "max profile difference / max|u|", "net mid-section flux / total", "mass_drift"])
        for r in prow:
            wr.writerow([r["kn"], f"{r['delta']:.3f}", f"{r['gamma_1d']:.4f}", f"{r['gamma_2d']:.4f}",
                         f"{r['p_ratio']:.5f}", f"{r['p_ratio_1d']:.5f}", f"{r['err_u']:.3f}",
                         f"{r['net_flux']:.4f}", f"{r['drift']:.1e}"])
            print(r)
    overview()


if __name__ == "__main__":
    main()
