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
    if e < CIRCULAR_E_TOL:
        raise ValueError(f"True anomaly is undefined for a (near-)circular orbit, e={e:.3e}")

    cos_theta0 = np.dot(q.e_vec, r0) / (e * q.r_norm)
    theta0 = np.arccos(np.clip(cos_theta0, -1.0, 1.0))
    if np.dot(r0, v0) < 0:  # moving towards perigee: second half of the orbit
        theta0 = TWO_PI - theta0

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
