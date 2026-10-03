"""Knudsen paradox and thermal transpiration in a plane micro-channel (linearised kinetic theory).

Writes data/channel_table.csv and docs/knudsen_paradox.png, docs/knudsen_pump.png,
docs/channel_profiles.png.
"""
from __future__ import annotations

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mems.channel1d import solve_channel
from mems.reference import SHARIPOV_TABLE1

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(HERE, "docs")
DATA = os.path.join(HERE, "data")
os.makedirs(DOCS, exist_ok=True)
os.makedirs(DATA, exist_ok=True)

plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 110})
BLUE, ORANGE, GREY, RED = "#1f6fb4", "#e3742b", "#6b6b6b", "#c0392b"


def table(deltas, model):
    rows = []
    for d in deltas:
        p = solve_channel(d, model, "P")
        t = solve_channel(d, model, "T")
        rows.append(dict(delta=d, GP=-p.flow_rate, GT=t.flow_rate, JP=p.heat_flow,
                         QT=-t.heat_flow))
        print(f"{model:8s} delta={d:9.4f}  G_P={rows[-1]['GP']:.5f}  G_T={rows[-1]['GT']:.5f}  "
              f"J_P={rows[-1]['JP']:.5f}", flush=True)
    return rows


def main():
    deltas = np.round(np.geomspace(0.01, 100, 25), 6)
    res = {m: table(deltas, m) for m in ("bgk", "shakhov")}
    with open(os.path.join(DATA, "channel_table.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["model", "delta", "G_P", "G_T", "J_P", "Lambda_TT"])
        for m, rows in res.items():
            for r in rows:
                wr.writerow([m, r["delta"], f"{r['GP']:.6f}", f"{r['GT']:.6f}", f"{r['JP']:.6f}", f"{r['QT']:.6f}"])

    # --- asymptotic constants from the solver itself
    big = [25.0, 50.0, 100.0]
    consts = {}
    for m in ("bgk", "shakhov"):
        gp = np.array([-solve_channel(d, m, "P", n_cells=320).flow_rate for d in big])
        gt = np.array([solve_channel(d, m, "T", n_cells=320).flow_rate for d in big])
        # G_P - delta/6 = sigma_P + b/delta + c/delta^2 ;  delta G_T likewise
        A = np.vstack([np.ones(3), 1.0 / np.array(big), 1.0 / np.array(big) ** 2]).T
        sp = np.linalg.lstsq(A, gp - np.array(big) / 6, rcond=None)[0][0]
        st = np.linalg.lstsq(A, gt * np.array(big), rcond=None)[0][0]
        small = [1e-5, 1e-4]
        g_small = [-solve_channel(d, m, "P", n_cells=40).flow_rate for d in small]
        t_small = [solve_channel(d, m, "T", n_cells=40).flow_rate for d in small]
        slope = -(g_small[1] - g_small[0]) / np.log(small[1] / small[0])
        diff = [g - 2 * t for g, t in zip(g_small, t_small)]
        consts[m] = dict(sigma_P=sp, sigma_T=st, fm_slope=slope, gp_minus_2gt=diff)
        print(m, consts[m], flush=True)
    with open(os.path.join(DATA, "asymptotes.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["model", "sigma_P", "sigma_T", "-dG_P/dln(delta) (delta 1e-5..1e-4)",
                     "G_P-2G_T at 1e-5", "G_P-2G_T at 1e-4"])
        for m, c in consts.items():
            wr.writerow([m, f"{c['sigma_P']:.4f}", f"{c['sigma_T']:.4f}", f"{c['fm_slope']:.4f}",
                         f"{c['gp_minus_2gt'][0]:.4f}", f"{c['gp_minus_2gt'][1]:.4f}"])

    with open(os.path.join(DATA, "sharipov_check.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["delta", "G_P this code (BGK)", "Cercignani & Daneri", "Cercignani & Pagani",
                     "Huang et al.", "spread of the three"])
        for d, ref in SHARIPOV_TABLE1.items():
            g = -solve_channel(d, "bgk", "P").flow_rate
            wr.writerow([d, f"{g:.4f}", *ref, f"{min(ref):.4f}-{max(ref):.4f}"])

    figure_paradox(deltas, res, consts)
    figure_pump(deltas, res)
    figure_profiles()


def figure_paradox(deltas, res, consts):
    s = res["shakhov"]
    b = res["bgk"]
    gp_s = np.array([r["GP"] for r in s])
    gp_b = np.array([r["GP"] for r in b])
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw=dict(width_ratios=[1.35, 1]))
    a = ax[0]
    dd = np.geomspace(0.01, 100, 200)
    a.semilogx(dd, dd / 6, ":", color=GREY, label=r"no-slip Navier–Stokes  $\delta/6$")
    a.semilogx(dd, dd / 6 + consts["shakhov"]["sigma_P"], "--", color=GREY,
               label=r"first-order slip  $\delta/6+\sigma_P$")
    a.semilogx(deltas, gp_b, "s", ms=4, mfc="none", color=ORANGE, label="kinetic, BGK")
    a.semilogx(deltas, gp_s, "-o", ms=4, color=BLUE, label="kinetic, S-model")
    ref_d = [d for d in SHARIPOV_TABLE1 if d >= 0.1]
    a.semilogx(ref_d, [SHARIPOV_TABLE1[d][0] for d in ref_d], "kx", ms=8, mew=1.5,
               label="BGK, Cercignani & Daneri (1963)")
    k = np.argmin(gp_s)
    a.annotate(f"Knudsen minimum\n$\\delta\\approx{deltas[k]:.2g}$, $G_P={gp_s[k]:.3f}$",
               xy=(deltas[k], gp_s[k]), xytext=(0.012, 0.35), textcoords="data",
               arrowprops=dict(arrowstyle="->", color=RED), color=RED)
    a.set_ylim(0, 5)
    a.set_xlabel(r"rarefaction parameter  $\delta = pH/(\mu v_0)\;\approx\;0.9/\mathrm{Kn}$")
    a.set_ylabel(r"reduced flow rate  $G_P$")
    a.set_title("Knudsen paradox: the flow rate has a minimum")
    a.legend(loc="upper right", fontsize=9)
    sec = a.secondary_xaxis("top", functions=(lambda d: 0.9027 / np.maximum(d, 1e-9),
                                              lambda kn: 0.9027 / np.maximum(kn, 1e-9)))
    sec.set_xlabel("Kn")

    a = ax[1]
    for d, col in zip([0.1, 1.0, 10.0], [RED, BLUE, ORANGE]):
        p = solve_channel(d, "shakhov", "P")
        u = -p.u / p.flow_rate * -1  # normalise by the mean velocity
        a.plot(p.x, -p.u / (-p.flow_rate / 2), color=col, lw=2, label=f"δ = {d:g}")
    xx = np.linspace(-0.5, 0.5, 100)
    a.plot(xx, 1.5 * (1 - 4 * xx**2), ":", color=GREY, label="parabola, no slip")
    a.set_xlabel("x / H")
    a.set_ylabel(r"$u(x)\,/\,\bar u$")
    a.set_title("velocity profiles: slip grows as the gas thins")
    a.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(DOCS, "knudsen_paradox.png"))
    plt.close(fig)


def figure_pump(deltas, res):
    s = res["shakhov"]
    b = res["bgk"]
    gt_s = np.array([r["GT"] for r in s])
    gp_s = np.array([r["GP"] for r in s])
    jp_s = np.array([r["JP"] for r in s])
    gam_s = gt_s / gp_s
    gam_b = np.array([r["GT"] / r["GP"] for r in b])
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    a = ax[0]
    a.loglog(deltas, gt_s, "-o", ms=4, color=BLUE, label=r"$G_T$: mass flow per $\nabla\ln T$")
    a.loglog(deltas, jp_s, "x", ms=7, color=RED, label=r"$J_P$: heat flow per $\nabla\ln p$")
    a.set_xlabel(r"$\delta$")
    a.set_title("thermal creep and its Onsager twin")
    a.legend(fontsize=9)
    a.text(0.05, 0.08, r"Onsager:  $G_T = J_P$" + f"\nmax |G_T − J_P| = {np.max(abs(gt_s - jp_s)):.0e}",
           transform=a.transAxes, fontsize=9)

    a = ax[1]
    a.semilogx(deltas, gam_s, "-o", ms=4, color=BLUE, label="S-model (Pr = 2/3)")
    a.semilogx(deltas, gam_b, "s", ms=4, mfc="none", color=ORANGE, label="BGK (Pr = 1)")
    a.axhline(0.5, color=GREY, ls="--")
    a.text(0.012, 0.47, "Knudsen's law  1/2", color=GREY, va="top")
    a.set_ylim(0, 0.55)
    a.set_xlabel(r"$\delta$")
    a.set_ylabel(r"$\gamma = G_T / G_P = d\ln p / d\ln T$")
    a.set_title("thermomolecular pressure exponent")
    a.legend(fontsize=9, loc="lower left")

    a = ax[2]
    # one pump stage between a cold and a hot reservoir: integrate dp/p = gamma(delta(p,T)) dT/T
    from scipy.interpolate import interp1d
    gam = interp1d(np.log(deltas), gam_s, bounds_error=False, fill_value=(gam_s[0], gam_s[-1]))
    for d0, col in zip([0.1, 1.0, 10.0], [RED, BLUE, ORANGE]):
        T = np.linspace(1.0, 3.0, 400)
        lp = np.zeros_like(T)
        for i in range(1, len(T)):
            Tm = 0.5 * (T[i] + T[i - 1])
            dloc = d0 * np.exp(lp[i - 1]) * Tm ** (-0.81 - 0.5)  # delta ~ p T^-(omega+1/2)
            lp[i] = lp[i - 1] + float(gam(np.log(dloc))) * np.log(T[i] / T[i - 1])
        a.plot(T, np.exp(lp), color=col, lw=2, label=f"δ(cold end) = {d0:g}")
    a.plot(T, np.sqrt(T), "--", color=GREY, label=r"free molecular  $\sqrt{T_h/T_c}$")
    a.set_xlabel(r"$T_{hot}/T_{cold}$")
    a.set_ylabel(r"$p_{hot}/p_{cold}$")
    a.set_title("Knudsen pump: closed channel, no moving parts")
    a.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(DOCS, "knudsen_pump.png"))
    plt.close(fig)


def figure_profiles():
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    for d, col in zip([0.1, 1.0, 10.0], [RED, BLUE, ORANGE]):
        t = solve_channel(d, "shakhov", "T")
        p = solve_channel(d, "shakhov", "P")
        ax[0].plot(t.x, t.u, color=col, lw=2, label=f"δ = {d:g}")
        ax[1].plot(p.x, p.q, color=col, lw=2, label=f"δ = {d:g}")
    cmap = plt.get_cmap("viridis")
    ds = [0.1, 1.0, 2.0, 5.0, 10.0, 30.0]
    for j, d in enumerate(ds):
        t = solve_channel(d, "shakhov", "T")
        p = solve_channel(d, "shakhov", "P")
        u = t.u + (t.flow_rate / -p.flow_rate) * p.u   # zero net flow
        ax[2].plot(t.x, u / np.abs(u).max(), color=cmap(j / (len(ds) - 1)), lw=2, label=f"δ = {d:g}")
    ax[2].set_title("closed channel: creep + return flow (zero net)")
    ax[2].set_xlabel("x / H")
    ax[2].set_ylabel(r"$u / \max|u|$   (+ towards the hot end)")
    ax[0].set_title(r"thermal creep: $u_T(x)$ for $\nabla\ln T=1$ (towards the hot end)")
    ax[0].set_xlabel("x / H")
    ax[1].set_title(r"mechanocaloric heat flux $q_P(x)$ for $\nabla\ln p=1$")
    ax[1].set_xlabel("x / H")
    for a in ax:
        a.axhline(0, color="k", lw=0.6)
    ax[0].legend(fontsize=9)
    ax[1].legend(fontsize=9)
    ax[2].legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    fig.tight_layout()
    fig.savefig(os.path.join(DOCS, "channel_profiles.png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
