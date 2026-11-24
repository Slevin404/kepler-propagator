"""Tests for the Kepler solver, anomaly conversions and orbit timing (orbits.kepler)."""

import numpy as np
import pytest

from orbits.kepler import E_to_theta, kepler_solve, orbit_timing, theta_to_E

# Orbit A (same as Project 1).
R0_A = np.array([-6045.0, -3490.0, 2500.0])  # km
V0_A = np.array([-3.457, 6.618, 2.533])  # km/s

# (M [rad], e, E [rad], theta [deg]); E quoted to 5 decimals, theta to 3.
SOLVER_CASES = [
    (1.0, 0.5, 1.49870, 116.357),
    (3.6029, 0.37255, 3.47942, 193.155),
    (0.1, 0.9, 0.63084, 109.782),
    (6.0, 0.8, 5.36383, 247.908),
    (0.05, 0.95, 0.53142, 119.056),
]


@pytest.mark.parametrize("M, e, E_ref, theta_ref_deg", SOLVER_CASES)
def test_kepler_solve_reference_values(M, e, E_ref, theta_ref_deg) -> None:
    E = kepler_solve(M, e)

    assert E == pytest.approx(E_ref, abs=1e-5)
    assert np.degrees(E_to_theta(E, e)) == pytest.approx(theta_ref_deg, abs=1e-3)
    assert abs(E - e * np.sin(E) - M) < 1e-12


@pytest.mark.parametrize("e", [0.0, 0.1, 0.5, 0.9, 0.99])
def test_theta_E_round_trip(e) -> None:
    E_grid = np.concatenate(([0.0, np.pi], np.linspace(0.0, 2 * np.pi, 37, endpoint=False)))
    for E in E_grid:
        assert theta_to_E(E_to_theta(E, e), e) == pytest.approx(E, abs=1e-12)


@pytest.mark.parametrize("M_shift", [-2 * np.pi, -6 * np.pi, 2 * np.pi, 10 * np.pi])
def test_kepler_solve_reduces_mean_anomaly(M_shift) -> None:
    M, e = 1.0, 0.5
    assert kepler_solve(M + M_shift, e) == pytest.approx(kepler_solve(M, e), abs=1e-12)


def test_kepler_solve_negative_mean_anomaly() -> None:
    M, e = -0.3, 0.4
    assert kepler_solve(M, e) == pytest.approx(kepler_solve(np.mod(M, 2 * np.pi), e), abs=1e-12)


@pytest.mark.parametrize("e", [1.0, 1.5, -0.1])
def test_kepler_solve_rejects_non_elliptic(e) -> None:
    with pytest.raises(ValueError):
        kepler_solve(1.0, e)


def test_kepler_solve_raises_when_not_converged() -> None:
    with pytest.raises(RuntimeError, match="did not converge"):
        kepler_solve(0.05, 0.95, max_iter=1)


def test_orbit_timing_orbit_a() -> None:
    t = orbit_timing(R0_A, V0_A)

    # Tolerances: one unit in the last quoted digit of each reference value.
    assert np.degrees(t["theta0"]) == pytest.approx(28.446, abs=1e-3)
    assert t["E0"] == pytest.approx(0.42014, abs=1e-5)
    assert t["M0"] == pytest.approx(0.35031, abs=1e-5)
    assert t["n"] == pytest.approx(7.6635e-4, abs=1e-8)
    assert t["t0"] == pytest.approx(457.1, abs=0.1)
    assert t["T"] == pytest.approx(8198.834, abs=1e-3)
    assert t["a"] == pytest.approx(8788.08, abs=1e-2)
    assert t["e"] == pytest.approx(0.17121, abs=1e-5)
