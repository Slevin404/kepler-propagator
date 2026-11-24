"""Kepler's equation and anomaly conversions for elliptical orbits.

Units: km, s. Angles in radians.

References
----------
H. D. Curtis, "Orbital Mechanics for Engineering Students", ch. 3.
G. Mengali, "Space Flight Mechanics", Universita di Pisa, ch. 3-4.
"""

from typing import Optional, Union

import numpy as np
from numpy.typing import ArrayLike

from orbits.constants import MU_EARTH
from orbits.twobody import orbital_quantities

TWO_PI = 2.0 * np.pi
CIRCULAR_E_TOL = 1e-12  # below this the eccentricity vector (and theta) is undefined
SIN_DTHETA_TOL = 1e-10  # below this the Curtis dtheta form of fd is singular


def _check_elliptic(e: float) -> None:
    """Raise ValueError unless 0 <= e < 1."""
    if not 0.0 <= e < 1.0:
        raise ValueError(f"Eccentricity must satisfy 0 <= e < 1 (elliptical orbit), got e={e}")


def kepler_solve(
    M: float,
    e: float,
    tol: float = 1e-12,
    max_iter: int = 50,
    E0: Optional[float] = None,
    return_iterations: bool = False,
) -> Union[float, tuple[float, int]]:
    """Solve Kepler's equation E - e sin(E) = M for the eccentric anomaly E.

    Newton-Raphson iteration (Curtis, Algorithm 3.1):
        E_{k+1} = E_k - (E_k - e sin(E_k) - M) / (1 - e cos(E_k))
    stopped when |E_{k+1} - E_k| < tol.

    M is first reduced to [0, 2pi), so any real M (negative, or several
    revolutions) is accepted; the returned E is the solution for that reduced M.

    Parameters
    ----------
    M : float
        Mean anomaly [rad].
    e : float
        Eccentricity, 0 <= e < 1.
    tol : float
        Convergence threshold on |delta E| [rad].
    max_iter : int
        Maximum number of Newton iterations.
    E0 : float, optional
        Initial guess [rad]. Default pi, which converges for every e < 1.
    return_iterations : bool
        If True, also return the number of Newton iterations performed.

    Returns
    -------
    float or (float, int)
        E [rad], or (E, iterations) if ``return_iterations`` is True.

    Raises
    ------
    ValueError
        If e is outside [0, 1).
    RuntimeError
        If the iteration does not converge within ``max_iter`` iterations.
    """
    _check_elliptic(e)
    M = float(np.mod(M, TWO_PI))
    E = np.pi if E0 is None else float(E0)

    for k in range(1, max_iter + 1):
        delta = (E - e * np.sin(E) - M) / (1.0 - e * np.cos(E))
        E -= delta
        if abs(delta) < tol:
            return (E, k) if return_iterations else E

    residual = E - e * np.sin(E) - M
    raise RuntimeError(
        f"Kepler solver did not converge in {max_iter} iterations "
        f"(M={M}, e={e}, last residual={residual:.3e})"
    )


def E_to_theta(E: float, e: float) -> float:
    """Convert eccentric anomaly E to true anomaly theta.

    From tan(theta/2) = sqrt((1+e)/(1-e)) tan(E/2) (Curtis, eq. 3.10b),
    written with arctan2 on the half angles so that E = pi is not singular:
        theta = 2 arctan2(sqrt(1+e) sin(E/2), sqrt(1-e) cos(E/2))

    Parameters
    ----------
    E : float
        Eccentric anomaly [rad].
    e : float
        Eccentricity, 0 <= e < 1.

    Returns
    -------
    float
        True anomaly theta in [0, 2pi) [rad].
    """
    _check_elliptic(e)
    theta = 2.0 * np.arctan2(np.sqrt(1.0 + e) * np.sin(E / 2), np.sqrt(1.0 - e) * np.cos(E / 2))
    return float(np.mod(theta, TWO_PI))


def theta_to_E(theta: float, e: float) -> float:
    """Convert true anomaly theta to eccentric anomaly E.

    From tan(E/2) = sqrt((1-e)/(1+e)) tan(theta/2) (Curtis, eq. 3.10a),
    written with arctan2 on the half angles so that theta = pi is not singular:
        E = 2 arctan2(sqrt(1-e) sin(theta/2), sqrt(1+e) cos(theta/2))

    Parameters
    ----------
    theta : float
        True anomaly [rad].
    e : float
        Eccentricity, 0 <= e < 1.

    Returns
    -------
    float
        Eccentric anomaly E in [0, 2pi) [rad].
    """
    _check_elliptic(e)
    E = 2.0 * np.arctan2(np.sqrt(1.0 - e) * np.sin(theta / 2), np.sqrt(1.0 + e) * np.cos(theta / 2))
    return float(np.mod(E, TWO_PI))


def _true_anomaly(r: np.ndarray, v: np.ndarray, e_vec: np.ndarray) -> float:
    """True anomaly from the state vector and the eccentricity vector (Curtis, eq. 4.13a).

        theta = arccos(e_vec . r / (e |r|)),  theta -> 2pi - theta if r . v < 0

    Raises ValueError for a (near-)circular orbit, where e_vec and theta are undefined.
    """
    e = np.linalg.norm(e_vec)
    if e < CIRCULAR_E_TOL:
        raise ValueError(f"True anomaly is undefined for a (near-)circular orbit, e={e:.3e}")
    theta = np.arccos(np.clip(np.dot(e_vec, r) / (e * np.linalg.norm(r)), -1.0, 1.0))
    if np.dot(r, v) < 0:  # moving towards perigee: second half of the orbit
        theta = TWO_PI - theta
    return float(theta)


def orbit_timing(r0: ArrayLike, v0: ArrayLike, mu: float = MU_EARTH) -> dict[str, float]:
    """Compute anomalies and timing of an elliptical orbit from a state vector.

    Formulas (Curtis, ch. 3):
        theta0 = arccos(e_vec . r0 / (e |r0|)),  theta0 -> 2pi - theta0 if r0 . v0 < 0
        E0     from theta0 (see ``theta_to_E``)
        M0     = E0 - e sin(E0)           (Kepler's equation, eq. 3.14)
        n      = sqrt(mu / a^3),  T = 2pi / n
        t0     = M0 / n                   (time since perigee passage)

    Parameters
    ----------
    r0, v0 : array_like, shape (3,)
        Position [km] and velocity [km/s].
    mu : float
        Gravitational parameter [km^3/s^2].

    Returns
    -------
    dict
        Keys: a [km], e [-], theta0 [rad], E0 [rad], M0 [rad],
        n [rad/s], T [s], t0 [s].

    Raises
    ------
    ValueError
        If the orbit is not elliptical, or is circular (theta undefined).
    """
    r0 = np.asarray(r0, dtype=float)
    v0 = np.asarray(v0, dtype=float)
    q = orbital_quantities(r0, v0, mu)
    if not q.is_closed:
        raise ValueError("orbit_timing requires an elliptical orbit (specific energy < 0)")
    e = q.e_norm
    theta0 = _true_anomaly(r0, v0, q.e_vec)

    E0 = theta_to_E(theta0, e)
    M0 = E0 - e * np.sin(E0)
    n = np.sqrt(mu / q.a**3)

    return {
        "a": q.a,
        "e": e,
        "theta0": float(theta0),
        "E0": E0,
        "M0": float(M0),
        "n": float(n),
        "T": float(TWO_PI / n),
        "t0": float(M0 / n),
    }


def propagate_kepler(
    r0: ArrayLike,
    v0: ArrayLike,
    dt: float,
    mu: float = MU_EARTH,
    return_coeffs: bool = False,
) -> Union[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]]:
    """Propagate an elliptical two-body orbit analytically by dt.

    Lagrange coefficients written in terms of the eccentric-anomaly increment
    dE (Curtis, ch. 3; equivalent to the universal-variable form of
    Mengali, ch. 4, with chi = sqrt(a) dE):

        a      = 1 / (2/r0 - v0^2/mu),  n = sqrt(mu/a^3)
        e cosE0 = 1 - r0/a,  e sinE0 = (r0 . v0) / sqrt(mu a)
        M      = E0 - e sin(E0) + n dt  ->  E from Kepler's equation
        dE     = E + 2pi floor(M / 2pi) - E0      (unwrapped increment)
        r      = a (1 - e cos E)
        f  = 1 - (a/r0)(1 - cos dE)          g  = dt - (dE - sin dE)/n
        fd = -sqrt(mu a) sin(dE) / (r r0)    gd = 1 - (a/r)(1 - cos dE)
        r_vec = f r0 + g v0,  v_vec = fd r0 + gd v0

    No term divides by sin(dtheta), so the formulation is regular for any dt
    (negative, multi-revolution, dtheta = k pi) and also for near-circular
    orbits, where theta and the eccentricity vector are ill-defined.

    Parameters
    ----------
    r0, v0 : array_like, shape (3,)
        Initial position [km] and velocity [km/s].
    dt : float
        Time of flight [s]; may be negative or longer than one period.
    mu : float
        Gravitational parameter [km^3/s^2].
    return_coeffs : bool
        If True, also return the Lagrange coefficients (f, g, fd, gd).

    Returns
    -------
    (r, v) or (r, v, (f, g, fd, gd))
        Position [km] and velocity [km/s] at t0 + dt; f, gd [-], g [s], fd [1/s].

    Raises
    ------
    ValueError
        If the orbit is not elliptical (a <= 0).
    """
    r0 = np.asarray(r0, dtype=float)
    v0 = np.asarray(v0, dtype=float)
    r0n = np.linalg.norm(r0)

    a = 1.0 / (2.0 / r0n - np.dot(v0, v0) / mu)
    if a <= 0:
        raise ValueError(f"propagate_kepler requires an elliptical orbit, got a={a:.6g} km")
    n = np.sqrt(mu / a**3)

    ecosE0 = 1.0 - r0n / a
    esinE0 = np.dot(r0, v0) / np.sqrt(mu * a)
    e = np.hypot(ecosE0, esinE0)
    E0 = np.arctan2(esinE0, ecosE0)

    M_tot = E0 - e * np.sin(E0) + n * dt  # not reduced: keeps the revolution count
    E = kepler_solve(M_tot, e)  # E corresponds to M_tot reduced to [0, 2pi)
    # Restore the full revolutions removed by the reduction; without this, g is
    # wrong as soon as M_tot leaves [0, 2pi) (multi-revolution or negative dt).
    dE = E + TWO_PI * np.floor(M_tot / TWO_PI) - E0
    r = a * (1.0 - e * np.cos(E))

    f = 1.0 - (a / r0n) * (1.0 - np.cos(dE))
    g = dt - (dE - np.sin(dE)) / n
    fd = -np.sqrt(mu * a) * np.sin(dE) / (r * r0n)
    gd = 1.0 - (a / r) * (1.0 - np.cos(dE))

    r_vec = f * r0 + g * v0
    v_vec = fd * r0 + gd * v0
    if return_coeffs:
        return r_vec, v_vec, (float(f), float(g), float(fd), float(gd))
    return r_vec, v_vec


def lagrange_coeffs_theta(
    r0: ArrayLike, v0: ArrayLike, dtheta: float, mu: float = MU_EARTH
) -> tuple[float, float, float, float]:
    """Lagrange coefficients as functions of the true-anomaly increment dtheta.

    Curtis form (sec. 2.11, "The Lagrange coefficients"), used here only to cross-check
    ``propagate_kepler``:

        h = |r0 x v0|,  theta = theta0 + dtheta,  r = (h^2/mu) / (1 + e cos theta)
        f  = 1 - (mu r / h^2)(1 - cos dtheta)
        g  = (r r0 / h) sin dtheta
        fd = (mu/h) ((1 - cos dtheta)/sin dtheta) ((mu/h^2)(1 - cos dtheta) - 1/r0 - 1/r)
        gd = 1 - (mu r0 / h^2)(1 - cos dtheta)

    fd divides by sin(dtheta), so this form is singular for dtheta = k pi
    (e.g. dt = T, or dt = T/2 from perigee). This is why ``propagate_kepler``
    uses the dE form instead.

    Parameters
    ----------
    r0, v0 : array_like, shape (3,)
        Initial position [km] and velocity [km/s].
    dtheta : float
        True-anomaly increment [rad].
    mu : float
        Gravitational parameter [km^3/s^2].

    Returns
    -------
    (f, g, fd, gd)
        f, gd [-], g [s], fd [1/s].

    Raises
    ------
    ValueError
        If |sin(dtheta)| < 1e-10 (singular form), or the orbit is circular.
    """
    if abs(np.sin(dtheta)) < SIN_DTHETA_TOL:
        raise ValueError(
            f"Curtis f, g form is singular for dtheta = k*pi (sin(dtheta) = {np.sin(dtheta):.1e}); "
            "use propagate_kepler instead"
        )
    r0 = np.asarray(r0, dtype=float)
    v0 = np.asarray(v0, dtype=float)
    q = orbital_quantities(r0, v0, mu)
    h = q.h_norm
    r0n = q.r_norm
    theta = _true_anomaly(r0, v0, q.e_vec) + dtheta
    r = (h**2 / mu) / (1.0 + q.e_norm * np.cos(theta))

    one_minus_cos = 1.0 - np.cos(dtheta)
    f = 1.0 - (mu * r / h**2) * one_minus_cos
    g = (r * r0n / h) * np.sin(dtheta)
    fd = (mu / h) * (one_minus_cos / np.sin(dtheta)) * ((mu / h**2) * one_minus_cos - 1.0 / r0n - 1.0 / r)
    gd = 1.0 - (mu * r0n / h**2) * one_minus_cos
    return float(f), float(g), float(fd), float(gd)
