"""Compare Newton iteration counts for Kepler's equation with E0 = M and E0 = pi.

Usage: python scripts/newton_iterations.py
"""

import numpy as np

from orbits.kepler import kepler_solve

# (M [rad], e): the five reference cases plus a hard case with e close to 1 and small M.
CASES = [
    (1.0, 0.5),
    (3.6029, 0.37255),
    (0.1, 0.9),
    (6.0, 0.8),
    (0.05, 0.95),
    (0.01, 0.99),
]


def iteration_table(cases: list[tuple[float, float]]) -> list[tuple[float, float, float, int, int]]:
    """Return (M, e, E, iterations with E0 = M, iterations with E0 = pi) for each case."""
    rows = []
    for M, e in cases:
        E, it_m = kepler_solve(M, e, E0=np.mod(M, 2 * np.pi), return_iterations=True)
        _, it_pi = kepler_solve(M, e, E0=np.pi, return_iterations=True)
        rows.append((M, e, E, it_m, it_pi))
    return rows


def format_table(rows: list[tuple[float, float, float, int, int]]) -> str:
    """Format the iteration table as fixed-width text."""
    lines = [
        f"{'M [rad]':>8} {'e':>8} {'E [rad]':>10} {'it E0=M':>8} {'it E0=pi':>9}",
        "-" * 47,
    ]
    for M, e, E, it_m, it_pi in rows:
        lines.append(f"{M:8.4f} {e:8.5f} {E:10.5f} {it_m:8d} {it_pi:9d}")
    return "\n".join(lines)


def main() -> None:
    print(format_table(iteration_table(CASES)))


if __name__ == "__main__":
    main()
