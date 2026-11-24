"""Smoke tests: the package imports and the two-body utilities from Project 1 work."""

import numpy as np
import pytest

# Orbit A (same as Project 1).
R0_A = np.array([-6045.0, -3490.0, 2500.0])  # km
V0_A = np.array([-3.457, 6.618, 2.533])  # km/s


def test_package_imports() -> None:
    import orbits  # noqa: F401
    from orbits import constants, twobody  # noqa: F401


def test_orbit_a_semi_major_axis_and_eccentricity() -> None:
    from orbits.twobody import orbital_quantities

    q = orbital_quantities(R0_A, V0_A)

    # Reference values are quoted to 2 decimals (km) and 5 decimals (-).
    assert q.a == pytest.approx(8788.08, abs=1e-2)
    assert q.e_norm == pytest.approx(0.17121, abs=1e-5)
