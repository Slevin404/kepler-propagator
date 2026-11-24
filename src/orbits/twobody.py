# Adapted from Project 1 (orbital-mechanics-python)
"""Two-body problem: orbital quantities, equations of motion and numerical propagation.

Units: km, s (km, km/s, km^3/s^2). Angles in radians, except where noted.

References
----------
H. D. Curtis, "Orbital Mechanics for Engineering Students", ch. 2-4.
"""

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import solve_ivp
from scipy.optimize import OptimizeResult

from orbits.constants import MU_EARTH


@dataclass(frozen=True)
class OrbitalQuantities:
    """Orbital quantities derived from a position/velocity state vector.

    Attributes
    ----------
    r_norm : float
        Position magnitude |r| [km].
    v_norm : float
        Velocity magnitude |v| [km/s].
    h_vec : np.ndarray
        Specific angular momentum vector h = r x v [km^2/s].
    h_norm : float
        Specific angular momentum magnitude |h| [km^2/s].
    energy : float
        Specific mechanical energy eps = v^2/2 - mu/r [km^2/s^2].
    a : float or None
        Semi-major axis [km]. None if the orbit is not closed (eps >= 0).
    period : float or None
        Orbital period [s]. None if the orbit is not closed (eps >= 0).
    e_vec : np.ndarray
        Eccentricity vector [-].
    e_norm : float
        Eccentricity magnitude [-].
    inclination_deg : float
        Inclination [deg].
    r_perigee : float
        Perigee radius [km].
    r_apogee : float or None
        Apogee radius [km]. None if the orbit is not closed (eps >= 0).
    is_closed : bool
        True if the orbit is elliptical (eps < 0).
    """

    r_norm: float
    v_norm: float
    h_vec: np.ndarray
    h_norm: float
    energy: float
    a: Optional[float]
    period: Optional[float]
    e_vec: np.ndarray
    e_norm: float
    inclination_deg: float
    r_perigee: float
    r_apogee: Optional[float]
    is_closed: bool


def orbital_quantities(r: ArrayLike, v: ArrayLike, mu: float = MU_EARTH) -> OrbitalQuantities:
    """Compute orbital quantities from a state vector.

    Formulas (Curtis, ch. 2 and 4):
        h     = r x v
        eps   = v^2/2 - mu/r
        e_vec = (v x h)/mu - r/|r|
        a     = -mu / (2 eps)              (ellipse only)
        T     = 2 pi sqrt(a^3/mu)          (ellipse only)
        r_p   = a(1 - e),  r_a = a(1 + e)  (ellipse);  r_p = h^2 / (mu (1 + e)) otherwise
        i     = arccos(h_z / |h|)

    Parameters
    ----------
    r : array_like, shape (3,)
        Position vector [km].
    v : array_like, shape (3,)
        Velocity vector [km/s].
    mu : float
        Gravitational parameter of the central body [km^3/s^2].

    Returns
    -------
    OrbitalQuantities
        For open orbits (eps >= 0) the fields ``a``, ``period`` and
        ``r_apogee`` are None, since they are not defined; all other
        fields are still computed.
    """
    r = np.asarray(r, dtype=float)
    v = np.asarray(v, dtype=float)

    r_norm = np.linalg.norm(r)
    v_norm = np.linalg.norm(v)

    h_vec = np.cross(r, v)
    h_norm = np.linalg.norm(h_vec)

    energy = v_norm**2 / 2 - mu / r_norm

    e_vec = np.cross(v, h_vec) / mu - r / r_norm
    e_norm = np.linalg.norm(e_vec)

    # Clip guards against round-off pushing the argument slightly outside [-1, 1].
    inclination_deg = np.degrees(np.arccos(np.clip(h_vec[2] / h_norm, -1.0, 1.0)))

    is_closed = energy < 0
    if is_closed:
        a = -mu / (2 * energy)
        period = 2 * np.pi * np.sqrt(a**3 / mu)
        r_perigee = a * (1 - e_norm)
        r_apogee = a * (1 + e_norm)
    else:
        a = None
        period = None
        r_apogee = None
        # Valid for any conic, including the parabola where a is infinite.
        r_perigee = h_norm**2 / (mu * (1 + e_norm))

    return OrbitalQuantities(
        r_norm=float(r_norm),
        v_norm=float(v_norm),
        h_vec=h_vec,
        h_norm=float(h_norm),
        energy=float(energy),
        a=None if a is None else float(a),
        period=None if period is None else float(period),
        e_vec=e_vec,
        e_norm=float(e_norm),
        inclination_deg=float(inclination_deg),
        r_perigee=float(r_perigee),
        r_apogee=None if r_apogee is None else float(r_apogee),
        is_closed=bool(is_closed),
    )


def two_body_ode(t: float, y: ArrayLike, mu: float = MU_EARTH) -> np.ndarray:
    """Right-hand side of the two-body equations of motion.

    Formula (Curtis, eq. 2.22):
        r_ddot = -mu r / |r|^3

    Parameters
    ----------
    t : float
        Time [s]. Unused (the system is autonomous), kept for solve_ivp.
    y : array_like, shape (6,)
        State [x, y, z, vx, vy, vz] in km and km/s.
    mu : float
        Gravitational parameter [km^3/s^2].

    Returns
    -------
    np.ndarray, shape (6,)
        Derivative [vx, vy, vz, ax, ay, az] in km/s and km/s^2.
    """
    r = np.asarray(y[:3], dtype=float)
    v = np.asarray(y[3:], dtype=float)
    a = -mu * r / np.linalg.norm(r) ** 3
    return np.concatenate((v, a))


def propagate(
    r0: ArrayLike,
    v0: ArrayLike,
    t_final: float,
    n_points: int,
    mu: float = MU_EARTH,
    method: str = "RK45",
    **solver_kwargs: Any,
) -> OptimizeResult:
    """Propagate a two-body orbit numerically with scipy.integrate.solve_ivp.

    Integrates ``two_body_ode`` on [0, t_final] and samples the solution at
    ``n_points`` evenly spaced instants (passed to solve_ivp as ``t_eval``).

    Parameters
    ----------
    r0, v0 : array_like, shape (3,)
        Initial position [km] and velocity [km/s].
    t_final : float
        Final time [s]; integration runs on [0, t_final].
    n_points : int
        Number of output samples, evenly spaced and including t_final.
    mu : float
        Gravitational parameter [km^3/s^2].
    method : str
        solve_ivp integration method. Default "RK45", as in Project 1.
    **solver_kwargs
        Extra keyword arguments passed to solve_ivp (e.g. rtol, atol).

    Returns
    -------
    OptimizeResult
        ``t`` has shape (n_points,), ``y`` has shape (6, n_points).

    Raises
    ------
    RuntimeError
        If the integration fails.
    """
    y0 = np.concatenate((np.asarray(r0, dtype=float), np.asarray(v0, dtype=float)))
    t_eval = np.linspace(0.0, t_final, n_points)
    sol = solve_ivp(
        two_body_ode,
        (0.0, t_final),
        y0,
        method=method,
        t_eval=t_eval,
        args=(mu,),
        **solver_kwargs,
    )
    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")
    return sol
