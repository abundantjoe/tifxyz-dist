"""Surface-to-surface clearance bounds and the train/prediction certificate."""
import json
import subprocess
import sys

import numpy as np

from tifxyz_dist import Surface, clearance, certify_disjoint, save_grid
from tifxyz_dist.cli import main


def grid(xs, ys, f):
    X, Y = np.meshgrid(xs, ys)
    return np.stack(f(X, Y), -1).astype(np.float32)


def test_parallel_planes_exact():
    xs = ys = np.arange(0, 400, 20.0)
    A = grid(xs, ys, lambda X, Y: (X, Y, np.full(X.shape, 100.0)))
    B = grid(xs + 7, ys + 3, lambda X, Y: (X, Y, np.full(X.shape, 112.5)))
    c = clearance(A, B)
    assert abs(c['upper'] - 12.5) < 1e-6
    assert c['lower'] <= 12.5 + 1e-9


def test_crossing_planes_zero():
    xs = ys = np.arange(0, 400, 20.0)
    A = grid(xs, ys, lambda X, Y: (X, Y, np.full(X.shape, 200.0)))
    B = grid(xs, ys, lambda X, Y: (np.full(X.shape, 205.0), X, Y))   # plane x = 205 through A
    c = clearance(A, B)
    assert c['lower'] == 0.0 and c['upper'] < 1e-6


def edge_edge_pair(gap):
    # A: plane y = 0, top edge z = 0 along x; B: plane x = 105, bottom edge z = gap along y.
    # Nearest points are interior points of two EDGES: true clearance = gap, while every
    # vertex is at least sqrt(25 + gap^2) from the other surface.
    A = grid(np.arange(0, 220, 20.0), np.arange(-100, 1, 20.0), lambda X, Z: (X, np.zeros_like(X), Z + 100))
    B = grid(np.arange(-95, 106, 20.0), np.arange(0, 101, 20.0), lambda Y, Z: (np.full(Y.shape, 105.0), Y + 200, Z + 100 + gap))
    return A, B


def test_edge_edge_needs_refine():
    A, B = edge_edge_pair(1.0)
    # shift so all coordinates are non-negative (tifxyz invalid marker is < 0)
    A[..., 1] += 200
    c0 = clearance(A, B)
    assert c0['lower'] <= 1.0 <= c0['upper']                 # sound but loose
    assert c0['upper'] > 5.0
    c1 = clearance(A, B, refine=20)
    assert abs(c1['upper'] - 1.0) < 1e-6                     # witness found on the edge
    assert c1['lower'] <= 1.0 + 1e-9
    assert c1['upper'] - c1['lower'] <= Surface(A).Lc / 20 + 1e-9
    assert c1['lower'] > 0                                # certified: not touching


def dense_min(A, B, N):
    SA, SB = Surface(A), Surface(B)
    best = np.inf
    for S, T in ((SA, SB), (SB, SA)):
        qi, qj = np.nonzero(S.quad_valid)
        g = S.grid.astype(np.float64)
        a, b, c, e = g[qi, qj], g[qi, qj + 1], g[qi + 1, qj], g[qi + 1, qj + 1]
        w = np.array([(u / N, v / N) for u in range(N + 1) for v in range(N + 1 - u)])
        for P0, P1, P2 in ((a, b, e), (a, e, c)):
            pts = (P0[:, None] * (1 - w[:, 0] - w[:, 1])[None, :, None] + P1[:, None] * w[None, :, 0, None]
                   + P2[:, None] * w[None, :, 1, None]).reshape(-1, 3)
            best = min(best, T.point_dist(pts).min())
    return best, max(SA.Lc, SB.Lc) / N


def test_random_curved_sheets_bounds_contain_truth():
    rng = np.random.default_rng(11)
    for t in range(4):
        n = 16; xs = 1000 + np.arange(n) * 20.0
        A = grid(xs, xs, lambda X, Y: (X + rng.normal(0, 2, X.shape), Y, 2000 + 30 * np.sin(X / 90.0 + t)))
        B = grid(xs + 5, xs + 9, lambda X, Y: (X, Y + rng.normal(0, 2, X.shape), 2045 + 30 * np.cos(Y / 70.0 + t)))
        D, gap = dense_min(A, B, 16)          # true value lies in [D - gap, D]
        for refine in (0, 6):
            c = clearance(A, B, refine)
            assert c['lower'] <= D + 1e-9
            assert c['upper'] >= D - gap - 1e-9
            assert c['lower'] <= c['upper']
        assert c['upper'] - c['lower'] <= Surface(A).Lc / 6 + 1e-9


def test_certificate_verdicts_and_cli(tmp_path):
    xs = ys = np.arange(0, 400, 20.0)
    tr = grid(xs, ys, lambda X, Y: (X, Y, np.full(X.shape, 100.0)))
    far = grid(xs, ys, lambda X, Y: (X, Y, np.full(X.shape, 160.0)))
    near = grid(xs, ys, lambda X, Y: (X, Y, np.full(X.shape, 104.0)))
    r = certify_disjoint({'t': tr}, {'far': far, 'near': near}, margin=20.0)
    v = {p['pred']: p['verdict'] for p in r['pairs']}
    assert v == {'far': 'PASS', 'near': 'FAIL'} and r['verdict'] == 'FAIL'
    for name, g in (('t', tr), ('far', far), ('near', near)):
        save_grid(str(tmp_path / name), g)
    assert main(['certify', '--train', str(tmp_path / 't'), '--pred', str(tmp_path / 'far'), '--margin', '20']) == 0
    assert main(['certify', '--train', str(tmp_path / 't'), '--pred', str(tmp_path / 'near'), '--margin', '20']) == 1
    np.save(tmp_path / 'p.npy', np.array([[110.0, 110.0, 130.0]]))
    out = subprocess.run([sys.executable, '-m', 'tifxyz_dist', 'points', str(tmp_path / 't'), str(tmp_path / 'p.npy')],
                         capture_output=True, text=True, check=True).stdout
    assert abs(json.loads(out)['min'] - 30.0) < 1e-6


def test_stretched_quads_clearance_tight():
    # one 400-vox-wide quad column must not loosen the bound elsewhere: parallel sheets 10 apart
    xs = np.arange(0, 600, 20.0); xs = np.where(xs > 300, xs + 400, xs)
    ys = np.arange(0, 400, 20.0)
    A = grid(xs, ys, lambda X, Y: (X, Y, np.full(X.shape, 100.0)))
    B = grid(np.arange(0, 1000, 20.0) + 3, ys + 5, lambda X, Y: (X, Y, np.full(X.shape, 110.0)))
    c = clearance(A, B)
    assert abs(c['upper'] - 10) < 1e-6 and c['lower'] <= 10
    c = clearance(A, B, refine=8)
    Rmed = 20 * np.sqrt(2) / np.sqrt(3)                  # regular quad covering radius
    assert abs(c['upper'] - 10) < 1e-6
    assert c['lower'] >= 10 - Rmed / 8 - 1e-9            # the 400-vox quads do not loosen it
