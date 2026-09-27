"""Accuracy and speed numbers quoted in the README (synthetic sheets, fixed seeds).

Usage: python demo/accuracy.py [MESH_DIR]
Optional MESH_DIR (a tifxyz) adds an on-surface test on a real mesh: random points sampled on its
own triangles must have distance 0."""
import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from tifxyz_dist import Surface, load_grid

rng = np.random.default_rng(20260927)
out = {}

def on_surface(S, n):
    qi, qj = np.nonzero(S.quad_valid)
    k = rng.integers(0, len(qi), n); g = S.grid.astype(np.float64)
    u = rng.random((n, 1)); v = rng.random((n, 1)) * (1 - u)
    lower = rng.random(n) < 0.5
    a = g[qi[k], qj[k]]; e = g[qi[k] + 1, qj[k] + 1]
    b = np.where(lower[:, None], g[qi[k], qj[k] + 1], g[qi[k] + 1, qj[k]])
    return a + u * (b - a) + v * (e - a)

# 0. triangle kernel vs an independent implementation (plane projection + barycentric solve,
#    else min over the three edge segments), 100k random point/triangle pairs incl. slivers
from tifxyz_dist import closest_point_dist
m = 100000
A, B, C = (rng.normal(0, 10, (m, 3)) for _ in range(3))
sl = rng.random(m) < 0.1; C[sl] = A[sl] + (B[sl] - A[sl]) * rng.random((sl.sum(), 1)) + rng.normal(0, 1e-3, (sl.sum(), 3))
Q = rng.normal(0, 15, (m, 3))
def seg(q, s, e):
    d = e - s; t = np.clip(((q - s) * d).sum(1) / np.maximum((d * d).sum(1), 1e-300), 0, 1)
    return np.linalg.norm(q - s - d * t[:, None], axis=1)
nrm = np.cross(B - A, C - A); nn = np.linalg.norm(nrm, axis=1); nh = nrm / nn[:, None]
h = ((Q - A) * nh).sum(1); F = Q - h[:, None] * nh
M = np.stack([B - A, C - A, nh], -1)
uvw = np.linalg.solve(M, (F - A)[..., None])[..., 0]
inside = (uvw[:, 0] >= 0) & (uvw[:, 1] >= 0) & (uvw[:, 0] + uvw[:, 1] <= 1)
ref = np.minimum.reduce([seg(Q, A, B), seg(Q, B, C), seg(Q, C, A)])
ref = np.where(inside, np.minimum(np.abs(h), ref), ref)
out['kernel_pairs'] = m
out['kernel_max_abs_diff_vs_independent'] = float(np.abs(closest_point_dist(Q, A, B, C) - ref).max())

# 1. random curved (pruning check: KD-tree + covering bound vs every triangle, same kernel) sheets vs brute force (every triangle)
worst = 0.0; nq = 0
for t in range(20):
    n = 30; j, i = np.meshgrid(np.arange(n), np.arange(n))
    x = 1000 + j * 20.0 + rng.normal(0, 3, (n, n)); y = 1000 + i * 20.0 + rng.normal(0, 3, (n, n))
    z = 2000 + 80 * np.sin(j / 4.0 + t) + 60 * np.cos(i / 5.0) + rng.normal(0, 2, (n, n))
    g = np.stack([x, y, z], -1).astype(np.float32); g[rng.random((n, n)) < 0.05] = -1
    S = Surface(g)
    Q = np.column_stack([rng.uniform(900, 1700, 200), rng.uniform(900, 1700, 200), rng.uniform(1800, 2200, 200)])
    worst = max(worst, float(np.abs(S.point_dist(Q) - S.brute(Q)).max())); nq += len(Q)
    P = on_surface(S, 2000)
    out.setdefault('synthetic_on_surface_max_vox', 0.0)
    out['synthetic_on_surface_max_vox'] = max(out['synthetic_on_surface_max_vox'], float(S.point_dist(P).max()))
out['brute_force_queries'] = nq
out['brute_force_max_abs_diff_vox'] = worst

# 2. nearest-vertex error that the exact method removes (plane, 20-vox grid, points 1 vox above)
j, i = np.meshgrid(np.arange(60), np.arange(60))
S = Surface(np.stack([j * 20.0, i * 20.0, np.full(i.shape, 500.0)], -1).astype(np.float32))
P = np.column_stack([rng.uniform(100, 1000, 10000), rng.uniform(100, 1000, 10000), np.full(10000, 501.0)])
dv, _ = S.tree.query(P)
out['plane_1vox_nearest_vertex_error_max_vox'] = float((dv - 1).max())
out['plane_1vox_exact_error_max_vox'] = float(np.abs(S.point_dist(P) - 1).max())

if len(sys.argv) > 1:
    S = Surface(load_grid(sys.argv[1]))
    P = on_surface(S, 100000)
    t0 = time.time(); d = S.point_dist(P); dt = time.time() - t0
    out['real_mesh'] = {'vertices': int(len(S.V)), 'quads': S.n_quads, 'Lmax_vox': S.L,
                        'on_surface_points': len(P), 'on_surface_max_vox': float(d.max()),
                        'points_per_second': round(len(P) / dt)}
print(json.dumps(out, indent=1))
