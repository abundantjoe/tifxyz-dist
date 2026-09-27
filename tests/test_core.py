"""Point-to-surface exactness on synthetic sheets with analytic answers, and against brute force."""
import numpy as np
import pytest

from tifxyz_dist import Surface, closest_point_dist, set_min

TOL = 1e-6


def plane(n=40, step=20.0, z0=1000.0, off=500.0):
    j, i = np.meshgrid(np.arange(n), np.arange(n))
    return np.stack([off + j * step, off + i * step, np.full(i.shape, z0)], -1).astype(np.float32)


def test_triangle_regions():
    a, b, c = np.array([[0, 0, 0.]]), np.array([[10, 0, 0.]]), np.array([[0, 10, 0.]])
    qs = np.array([[2, 2, 5.], [-3, -4, 0.], [5, -2, 0.], [10, 10, 0.], [20, 0, 0.]])
    got = closest_point_dist(qs, *(np.repeat(t, 5, 0) for t in (a, b, c)))
    assert np.allclose(got, [5, 5, 2, np.sqrt(50), 10], atol=TOL)


def test_plane_heights_at_cell_centre():
    # nearest vertex is 14.1 vox away laterally; a vertex-only method would be wrong here
    S = Surface(plane())
    hs = [0, 0.5, 1, 2, 7.3, 100, 499.9, 500.1]
    P = np.array([[810, 810, 1000.0 + h] for h in hs])
    assert np.allclose(S.point_dist(P), hs, atol=TOL)
    assert abs(S.min_dist(P)[0]) <= TOL
    assert np.allclose(S.point_dist(np.array([[810, 810, 4000.0]])), [3000.0], atol=TOL)
    r = set_min(np.array([[810, 810, 1003.0], [830, 850, 1250.0]]), S)
    assert abs(r['upper'] - 3) <= TOL and r['lower'] == r['upper'] and r['argmin'] == 0


def test_beyond_plane_edge():
    S = Surface(plane())
    assert np.allclose(S.point_dist(np.array([[470.0, 700, 1040.0]])), [50.0], atol=TOL)


def fold():
    n = 40; j, i = np.meshgrid(np.arange(n), np.arange(n))
    x = 500 + j * 20.0; y = 500 + i * 20.0
    return Surface(np.stack([x, y, 1000 + np.abs(x - 900)], -1).astype(np.float32))


def test_fold():
    Sf = fold()
    assert np.allclose(Sf.point_dist(np.array([[900, 800, 1100.0]])), [100 / np.sqrt(2)], atol=TOL)
    assert np.allclose(Sf.point_dist(np.array([[900, 800, 963.0]])), [37.0], atol=TOL)
    foot = np.array([1100, 810, 1200.0]); nrm = np.array([1, 0, -1]) / np.sqrt(2)
    assert np.allclose(Sf.point_dist((foot + 13 * nrm)[None]), [13.0], atol=TOL)


def test_u_fold_between_layers():
    n = 40; j, i = np.meshgrid(np.arange(n), np.arange(n))
    zu = np.where(i < 20, 1000.0, 1020.0)
    yu = np.where(i < 20, 500 + i * 20.0, 500 + (39 - i) * 20.0)
    Su = Surface(np.stack([500 + j * 20.0, yu, zu], -1).astype(np.float32))
    assert np.allclose(Su.point_dist(np.array([[810, 700, 1006.0], [810, 700, 1014.0]])), [6, 6], atol=TOL)


def test_hole_rim():
    g = plane(); g[10:20, 10:20] = -1
    assert np.allclose(Surface(g).point_dist(np.array([[790, 790, 1000.0]])), [110.0], atol=TOL)


@pytest.mark.parametrize('t', range(6))
def test_random_sheet_vs_brute(t):
    rng = np.random.default_rng(20260926 + t)
    n = 25; j, i = np.meshgrid(np.arange(n), np.arange(n))
    x = 1000 + j * 20.0 + rng.normal(0, 3, (n, n)); y = 1000 + i * 20.0 + rng.normal(0, 3, (n, n))
    z = 2000 + 80 * np.sin(j / 4.0 + t) + 60 * np.cos(i / 5.0) + rng.normal(0, 2, (n, n))
    g = np.stack([x, y, z], -1).astype(np.float32)
    g[rng.random((n, n)) < 0.05] = -1
    S = Surface(g)
    Q = np.column_stack([rng.uniform(900, 1600, 60), rng.uniform(900, 1600, 60), rng.uniform(1700, 2400, 60)])
    ref = S.brute(Q)
    assert np.abs(S.point_dist(Q) - ref).max() <= TOL
    assert np.abs(S.point_dist_loop(Q) - ref).max() <= TOL
    assert abs(S.min_dist(Q)[0] - ref.min()) <= TOL


def test_equilateral_lattice_worst_case_radius():
    # point above a triangle centroid: nearest vertex exactly at the covering radius L / sqrt(3)
    rng = np.random.default_rng(7)
    n = 30; j, i = np.meshgrid(np.arange(n), np.arange(n))
    x = 1000 + 20.0 * (j + 0.5 * (i % 2)); y = 1000 + 20.0 * np.sqrt(3) / 2 * i
    S = Surface(np.stack([x, y, np.full(x.shape, 3000.0)], -1).astype(np.float32))
    Q = np.column_stack([rng.uniform(1100, 1400, 300), rng.uniform(1100, 1400, 300),
                         3000 + rng.choice([0.01, 0.3, 2, 40, 900], 300)])
    ref = S.brute(Q)
    assert np.abs(S.point_dist(Q) - ref).max() <= TOL
    assert abs(S.min_dist(Q)[0] - ref.min()) <= TOL


def test_on_surface_points_are_zero():
    rng = np.random.default_rng(3)
    S = fold()
    qi, qj = np.nonzero(S.quad_valid)
    k = rng.integers(0, len(qi), 2000)
    g = S.grid.astype(np.float64)
    u = rng.random((2000, 1)); v = rng.random((2000, 1)) * (1 - u)
    a, b, e = g[qi[k], qj[k]], g[qi[k], qj[k] + 1], g[qi[k] + 1, qj[k] + 1]
    P = a + u * (b - a) + v * (e - a)
    assert S.point_dist(P).max() <= 1e-6


@pytest.mark.parametrize('t', range(4))
def test_stretched_quads_tiered_vs_brute(t):
    # a few quads stretched 10x (as on real traced meshes) go to a separate tier with its own radius
    rng = np.random.default_rng(99 + t)
    n = 30; j, i = np.meshgrid(np.arange(n), np.arange(n))
    x = 1000 + j * 20.0; y = 1000 + i * 20.0
    z = 2000 + 40 * np.sin(j / 5.0 + t) + rng.normal(0, 1, (n, n))
    x[:, 15:] += 180.0                                   # one column of quads 200 vox wide
    y[20:, :] += 150.0 * (t % 2)                         # and, for odd t, one tall row
    g = np.stack([x, y, z], -1).astype(np.float32)
    g[rng.random((n, n)) < 0.03] = -1
    S = Surface(g)
    assert len(S.tiers) >= 2 and S.tiers[0]['Lc'] < 0.2 * S.tiers[-1]['Lc']
    Q = np.column_stack([rng.uniform(950, 1900, 400), rng.uniform(950, 1900, 400), rng.uniform(1900, 2150, 400)])
    ref = S.brute(Q)
    assert np.abs(S.point_dist(Q) - ref).max() <= TOL
    assert abs(S.min_dist(Q)[0] - ref.min()) <= TOL
    assert (S.lower(Q) <= ref + TOL).all()


def test_lower_bounds_with_cell_screening():
    # cell-screened, capped lower bounds are never above the truth; negative coordinates included
    rng = np.random.default_rng(5)
    S = fold()
    Q = np.column_stack([rng.uniform(-3000, 4000, 6000), rng.uniform(-3000, 4000, 6000), rng.uniform(-2000, 5000, 6000)])
    Q[:500] = [900, 800, 1000] + rng.normal(0, 30, (500, 3))
    ref = S.point_dist(Q)
    for cap in (5.0, 50.0, 400.0, np.inf):
        lb = S.lower(Q, cap=cap, min_points=100)
        assert (lb <= ref + 1e-9).all() and (lb <= cap).all()
    assert abs(S.min_dist(Q)[0] - ref.min()) <= TOL
