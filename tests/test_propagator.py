"""Tests for the analytical Kepler propagator (orbits.kepler.propagate_kepler)."""

import numpy as np
import pytest

from orbits.constants import MU_EARTH
from orbits.kepler import (
    E_to_theta,
    kepler_solve,
    lagrange_coeffs_theta,
    orbit_timing,
    propagate_kepler,
)
from orbits.twobody import orbital_quantities, propagate

# Orbit A (same as Project 1).
R0_A = np.array([-6045.0, -3490.0, 2500.0])  # km
V0_A = np.array([-3.457, 6.618, 2.533])  # km/s
T_A = orbit_timing(R0_A, V0_A)["T"]  # s

# Orbit B: starts at perigee, r_p = 7000 km, e = 0.7.
RP_B = 7000.0  # km
E_B = 0.7
VP_B = np.sqrt(MU_EARTH * (1 + E_B) / RP_B)  # km/s, vis-viva at perigee
RA_B = RP_B * (1 + E_B) / (1 - E_B)  # km
R0_B = np.array([RP_B, 0.0, 0.0])
V0_B = np.array([0.0, VP_B, 0.0])
T_B = 2 * np.pi * np.sqrt((RP_B / (1 - E_B)) ** 3 / MU_EARTH)  # s

# Reference tables are quoted to 3 decimals (km) and 5 decimals (km/s).
TOL_R = 1e-3  # km
TOL_V = 1e-5  # km/s

# (dt [s], r [km], v [km/s], |r| [km])
ORBIT_A_CASES = [
    (0.0, [-6045, -3490, 2500], [-3.457, 6.618, 2.533], 7414.319),
    (1800.0, [-3657.634, 8032.697, 2812.013], [4.68311, 3.95139, -1.77696], 9263.365),
    (3600.0, [5331.624, 8676.857, -1487.861], [4.18571, -2.95444, -2.41901], 10292.123),
    (T_A, [-6045, -3490, 2500], [-3.457, 6.618, 2.533], 7414.319),
    (-3600.0, [8301.949, 4352.225, -3489.854], [1.53590, -5.46693, -1.44900], 10002.164),
    (3 * T_A + 1000.0, [-6546.850, 3683.615, 3663.491], [2.18568, 6.72377, -0.20433], 8357.717),
]

ORBIT_B_CASES = [
    (1000.0, [3768.776, 8460.397, 0], [-5.28674, 6.40633, 0], 9261.857),
    (T_B / 2, [-39666.667, 0, 0], [0, -1.73627, 0], 39666.667),
    (5000.0, [-16678.669, 16661.508, 0], [-4.09031, -0.04324, 0], 23575.068),
]

ORBITS = {"A": (R0_A, V0_A, T_A), "B": (R0_B, V0_B, T_B)}
# Negative, sub-period, dtheta = k*pi and multi-revolution times, as multiples of T plus offsets.
DT_GRID = [(-3, -500.0), (-1, 0.0), (0, -3600.0), (0, -1.0), (0, 1.0), (0, 1800.0),
           (0.5, 0.0), (1, 0.0), (2.5, 0.0), (10, 123.0)]  # fmt: skip


def _dt_values(T: float) -> list[float]:
    return [k * T + offset for k, offset in DT_GRID]


def _assert_state(r, v, r_ref, v_ref, r_norm_ref) -> None:
    np.testing.assert_allclose(r, r_ref, rtol=0, atol=TOL_R)
    np.testing.assert_allclose(v, v_ref, rtol=0, atol=TOL_V)
    assert np.linalg.norm(r) == pytest.approx(r_norm_ref, abs=TOL_R)


@pytest.mark.parametrize("dt, r_ref, v_ref, r_norm_ref", ORBIT_A_CASES)
def test_orbit_a_reference_states(dt, r_ref, v_ref, r_norm_ref) -> None:
    r, v = propagate_kepler(R0_A, V0_A, dt)
    _assert_state(r, v, r_ref, v_ref, r_norm_ref)


def test_orbit_a_one_period_returns_initial_state() -> None:
    r, v = propagate_kepler(R0_A, V0_A, T_A)
    np.testing.assert_allclose(r, R0_A, rtol=0, atol=1e-8)
    np.testing.assert_allclose(v, V0_A, rtol=0, atol=1e-11)


def test_orbit_b_semi_major_axis_and_period() -> None:
    q = orbital_quantities(R0_B, V0_B)
    assert q.a == pytest.approx(23333.333, abs=TOL_R)
    assert q.period == pytest.approx(35471.223, abs=TOL_R)


@pytest.mark.parametrize("dt, r_ref, v_ref, r_norm_ref", ORBIT_B_CASES)
def test_orbit_b_reference_states(dt, r_ref, v_ref, r_norm_ref) -> None:
    r, v = propagate_kepler(R0_B, V0_B, dt)
    _assert_state(r, v, r_ref, v_ref, r_norm_ref)


def test_orbit_b_apogee_physics() -> None:
    r, v = propagate_kepler(R0_B, V0_B, T_B / 2)
    # At apogee the velocity is perpendicular to the radius: r.v / (|r||v|) ~ 0.
    assert np.dot(r, v) / (np.linalg.norm(r) * np.linalg.norm(v)) == pytest.approx(0.0, abs=1e-12)
    # Conservation of angular momentum between perigee and apogee: v_a = v_p r_p / r_a.
    assert np.linalg.norm(v) == pytest.approx(VP_B * RP_B / RA_B, rel=1e-12)
    # Planar orbit in the xy plane.
    assert r[2] == 0.0 and v[2] == 0.0


@pytest.mark.parametrize("orbit", ORBITS)
def test_lagrange_identity(orbit) -> None:
    r0, v0, T = ORBITS[orbit]
    for dt in _dt_values(T):
        _, _, (f, g, fd, gd) = propagate_kepler(r0, v0, dt, return_coeffs=True)
        assert f * gd - fd * g == pytest.approx(1.0, abs=1e-10)


@pytest.mark.parametrize("orbit", ORBITS)
def test_forward_backward_round_trip(orbit) -> None:
    # Measured worst case: 1.5e-7 km and 1.3e-10 km/s (orbit B, dt = 10 T + 123 s);
    # round-off grows with the mean anomaly swept, so the margin is ~7x.
    r0, v0, T = ORBITS[orbit]
    for dt in _dt_values(T):
        r, v = propagate_kepler(r0, v0, dt)
        r_back, v_back = propagate_kepler(r, v, -dt)
        np.testing.assert_allclose(r_back, r0, rtol=0, atol=1e-6)
        np.testing.assert_allclose(v_back, v0, rtol=0, atol=1e-9)


@pytest.mark.parametrize("orbit", ORBITS)
def test_energy_and_angular_momentum_conserved(orbit) -> None:
    r0, v0, T = ORBITS[orbit]
    q0 = orbital_quantities(r0, v0)
    for dt in _dt_values(T):
        q = orbital_quantities(*propagate_kepler(r0, v0, dt))
        assert abs(q.energy / q0.energy - 1) < 1e-12
        assert abs(q.h_norm / q0.h_norm - 1) < 1e-12


@pytest.mark.parametrize("dt", [1800.0, 3600.0])
def test_delta_E_form_matches_curtis_delta_theta_form(dt) -> None:
    timing = orbit_timing(R0_A, V0_A)
    E = kepler_solve(timing["M0"] + timing["n"] * dt, timing["e"])
    dtheta = E_to_theta(E, timing["e"]) - timing["theta0"]

    f, g, fd, gd = lagrange_coeffs_theta(R0_A, V0_A, dtheta)
    r, v = propagate_kepler(R0_A, V0_A, dt)

    # Two analytical forms: they agree to round-off (measured < 1e-11 km, < 1e-13 km/s).
    np.testing.assert_allclose(f * R0_A + g * V0_A, r, rtol=0, atol=1e-8)
    np.testing.assert_allclose(fd * R0_A + gd * V0_A, v, rtol=0, atol=1e-11)


@pytest.mark.parametrize("dtheta", [np.pi, 2 * np.pi, 0.0])
def test_curtis_form_singular_at_k_pi(dtheta) -> None:
    with pytest.raises(ValueError, match="singular"):
        lagrange_coeffs_theta(R0_B, V0_B, dtheta)


def test_hyperbolic_orbit_rejected() -> None:
    v_escape = np.sqrt(2 * MU_EARTH / np.linalg.norm(R0_A))
    v0 = 1.5 * v_escape * V0_A / np.linalg.norm(V0_A)
    with pytest.raises(ValueError):
        propagate_kepler(R0_A, v0, 1000.0)


def test_matches_solve_ivp_with_tight_tolerances() -> None:
    dt = 3600.0
    sol = propagate(R0_A, V0_A, dt, 2, rtol=1e-10, atol=1e-12)  # RK45, as in Project 1
    r, _ = propagate_kepler(R0_A, V0_A, dt)
    assert np.linalg.norm(r - sol.y[:3, -1]) < 1e-3
