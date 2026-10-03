"""Basic properties of the 2D Shakhov solver on small grids."""
import numpy as np

from mems.kinetic2d import Box, VelocityGrid, knudsen_to_delta


def test_velocity_grid_moments():
    v = VelocityGrid(28, 6.0)
    m = np.exp(-(v.vx**2 + v.vy**2) / 2) / (2 * np.pi)
    assert abs((v.w * m).sum() - 1) < 1e-7
    assert abs((v.w * m * v.vx**2).sum() - 1) < 1e-6


def test_equilibrium_is_preserved():
    """Gas at rest between walls at the same temperature must stay at rest.

    Only to the truncation level of the velocity grid: the discrete Maxwellian
    cut at |v| = 6 has a temperature moment off by 1.2e-7.
    """
    b = Box(8, 8, kn=0.5, vel=VelocityGrid(24, 6.0))
    b.advance(0.5)
    m = b.moments()
    assert np.abs(m["ux"]).max() < 1e-7 and np.abs(m["uy"]).max() < 1e-7
    assert np.abs(m["T"] - 1.0).max() < 1e-6


def test_cavity_conserves_mass_and_turns_clockwise():
    b = Box(12, 12, kn=1.0, u_top=0.2, vel=VelocityGrid(16, 5.0))
    m0 = b.mass
    b.advance(2.0)
    m = b.moments()
    assert abs(b.mass / m0 - 1) < 1e-8  # round-off plus the Shakhov term on a truncated grid
    assert m["ux"][:, -1].mean() > 0 > m["ux"][:, 0].mean()  # driven along the lid, back along the floor
    # the lid compresses the gas into the downstream corner: hotter on the right
    assert m["T"][-1, -1] > m["T"][0, -1]


def test_delta_kn_relation():
    assert abs(knudsen_to_delta(1.0) - 0.9027) < 1e-3
