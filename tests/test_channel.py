"""Checks of the linearised channel solver against exact results of kinetic theory."""
import numpy as np
import pytest

from mems.channel1d import solve_channel, velocity_quadrature
from mems.reference import SHARIPOV_TABLE1


def test_quadrature_moments():
    c, w = velocity_quadrature()
    # half-range integrals of exp(-c^2)/sqrt(pi): 1/2, 1/4 for c^2, 3/8 for c^4
    assert np.isclose(w.sum(), 0.5, atol=1e-12)
    assert np.isclose((w * c**2).sum(), 0.25, atol=1e-12)
    assert np.isclose((w * c**4).sum(), 0.375, atol=1e-12)


@pytest.mark.parametrize("model", ["bgk", "shakhov"])
@pytest.mark.parametrize("delta", [0.05, 1.0, 20.0])
def test_onsager_reciprocity(model, delta):
    """Mass flow per temperature gradient equals heat flow per pressure gradient.

    Exact for the continuous problem; the discretisation error is O(dx^2) and
    only visible when the cells are optically thick.
    """
    p = solve_channel(delta, model, "P", n_cells=60)
    t = solve_channel(delta, model, "T", n_cells=60)
    assert abs(t.flow_rate - p.heat_flow) < 3e-5 * abs(t.flow_rate)


@pytest.mark.parametrize("delta", [0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
def test_sharipov_table(delta):
    """BGK flow rate within the spread of the published solutions (Sharipov & Seleznev 1998, Table 1)."""
    ref = SHARIPOV_TABLE1[delta]
    g = -solve_channel(delta, "bgk", "P", n_cells=80).flow_rate
    lo, hi = min(r for r in ref if r > 0.9 * max(ref)), max(ref)  # drops the outlier at small delta
    assert lo - 1e-3 <= g <= hi + 1e-3


def test_slip_limit():
    """Dense gas: G_P -> delta/6 + sigma_P, with sigma_P = 1.016 for BGK."""
    d = 100.0
    g = -solve_channel(d, "bgk", "P", n_cells=160).flow_rate
    assert abs(g - d / 6 - 1.016) < 0.02  # O(1/delta) correction is ~0.01 here


def test_symmetric_profiles_and_signs():
    p = solve_channel(1.0, "shakhov", "P", n_cells=60)
    t = solve_channel(1.0, "shakhov", "T", n_cells=60)
    assert np.allclose(p.u, p.u[::-1], atol=1e-10)
    assert p.flow_rate < 0          # gas flows down the pressure gradient
    assert t.flow_rate > 0          # ... and up the temperature gradient (thermal creep)


def test_free_molecular_limit():
    """delta -> 0: G_P ~ -ln(delta)/sqrt(pi) and G_T ~ G_P/2 (Knudsen's law p_h/p_c = sqrt(T_h/T_c)).

    Both diverge logarithmically with the same coefficient, so G_P - 2 G_T tends
    to a constant while the ratio approaches 1/2 only like 1/ln(1/delta).
    """
    g = {}
    for d in (1e-3, 1e-4):
        g[d] = (-solve_channel(d, "bgk", "P", n_cells=40).flow_rate,
                solve_channel(d, "bgk", "T", n_cells=40).flow_rate)
    slope = (g[1e-4][0] - g[1e-3][0]) / np.log(10.0)
    assert abs(slope - 1 / np.sqrt(np.pi)) < 0.02
    assert abs((g[1e-4][0] - 2 * g[1e-4][1]) - (g[1e-3][0] - 2 * g[1e-3][1])) < 0.01
