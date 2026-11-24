"""Compare the analytical Kepler propagator with numerical integration (solve_ivp, RK45).

Orbit A is propagated over three periods with both methods; the position error
|r_kepler - r_numerical| is plotted for tight and loose integrator tolerances.

Usage: python scripts/compare_numerical.py   (writes figures/position_error.png)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike

from orbits.kepler import orbit_timing, propagate_kepler
from orbits.twobody import propagate

R0_A = np.array([-6045.0, -3490.0, 2500.0])  # km
V0_A = np.array([-3.457, 6.618, 2.533])  # km/s
N_PERIODS = 3
N_POINTS = 1500

TOLERANCES = {
    "tight (rtol=1e-10, atol=1e-12)": {"rtol": 1e-10, "atol": 1e-12},
    "loose (rtol=1e-3, atol=1e-6)": {"rtol": 1e-3, "atol": 1e-6},
}
COLORS = ["#2a78d6", "#eb6834"]  # categorical slots 1-2 (blue, orange)

FIGURE_PATH = Path(__file__).resolve().parents[1] / "figures" / "position_error.png"


def position_errors(
    r0: ArrayLike, v0: ArrayLike, t_final: float, n_points: int, **solver_kwargs: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return sample times [s] and |r_kepler - r_numerical| [km] on [0, t_final].

    The analytical solution is always propagated from the initial state, so it
    carries no accumulated error; the numerical one comes from solve_ivp (RK45).
    """
    sol = propagate(r0, v0, t_final, n_points, **solver_kwargs)
    r_kepler = np.array([propagate_kepler(r0, v0, t)[0] for t in sol.t])
    return sol.t, np.linalg.norm(r_kepler - sol.y[:3].T, axis=1)


def compute_all_errors() -> tuple[float, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Return the period of orbit A [s] and (t, error) for each tolerance set."""
    period = orbit_timing(R0_A, V0_A)["T"]
    errors = {
        label: position_errors(R0_A, V0_A, N_PERIODS * period, N_POINTS, **tol)
        for label, tol in TOLERANCES.items()
    }
    return period, errors


def plot_position_errors(
    period: float, errors: dict[str, tuple[np.ndarray, np.ndarray]]
) -> plt.Figure:
    """Plot position error vs time (in orbital periods), log scale."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for color, (label, (t, err)) in zip(COLORS, errors.items()):
        # Skip t = 0, where both solutions coincide exactly (log(0) is undefined).
        ax.semilogy(t[1:] / period, err[1:], color=color, linewidth=2, label=label)
    ax.set_xlabel("Time [orbital periods, t/T]")
    ax.set_ylabel("Position error [km]")
    ax.set_title("Orbit A: analytical Kepler propagator vs solve_ivp (RK45)")
    ax.set_xlim(0, N_PERIODS)
    ax.grid(True, which="major", color="0.85", linewidth=0.8)
    ax.grid(True, which="minor", color="0.93", linewidth=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def main() -> None:
    period, errors = compute_all_errors()
    for label, (_, err) in errors.items():
        print(f"{label:32s} max error = {err.max():.3e} km   final error = {err[-1]:.3e} km")

    FIGURE_PATH.parent.mkdir(exist_ok=True)
    plot_position_errors(period, errors).savefig(FIGURE_PATH, dpi=150)
    print(f"Saved {FIGURE_PATH}")


if __name__ == "__main__":
    main()
