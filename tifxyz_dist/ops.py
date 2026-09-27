"""Certified distance queries built on core.Surface.

Every function returns an interval [lower, upper] that provably contains the true value of the
stated quantity (up to float64 rounding on float32 input coordinates). When lower == upper the
value is exact.

Proof sketch (see README): every point of a triangle lies within Lc = Lmax / sqrt(3) of one of
its vertices, and point-to-surface distance is 1-Lipschitz.
"""
import numpy as np

from .core import Surface


def as_surface(x):
    return x if isinstance(x, Surface) else Surface(x)


def points_to_surface(P, surf):
    """Exact per-point distances (n,) from points P (n, 3) to the triangulated surface."""
    return as_surface(surf).point_dist(np.asarray(P, np.float64).reshape(-1, 3))


def set_min(P, surf, stop_below=0.0):
    """min over points P of the point-to-surface distance.

    Returns dict(lower, upper, argmin). Exact (lower == upper) unless stop_below > 0 and some
    point is found within stop_below, in which case refinement stops early."""
    S = as_surface(surf)
    P = np.asarray(P, np.float64).reshape(-1, 3)
    ub, arg, info = S.min_dist(P, stop_vox=stop_below)
    lb = info['lower_bound_vox'] if info['early_stop'] else ub
    return {'lower': float(lb), 'upper': float(ub), 'argmin': int(arg), 'refined_points': info['refined_points']}


def _quad_min(S, vals):
    """min of a per-vertex quantity over the 4 corners of every quad (inf where invalid)."""
    v = np.full(S.valid.shape, np.inf)
    v[S.ij[:, 0], S.ij[:, 1]] = vals
    q = np.minimum.reduce([v[:-1, :-1], v[:-1, 1:], v[1:, :-1], v[1:, 1:]])
    return np.where(S.quad_valid, q, np.inf)


def _bary(N):
    return np.array([(u / N, v / N) for u in range(N + 1) for v in range(N + 1 - u)])


def _one_way(A, B, refine, max_samples):
    """Bounds on min over the triangulated surface A of dist(., B).

    Uses per-quad covering radii R_q = Lmax_q / sqrt(3): any point of quad q is within R_q of a
    corner, so dist(p, B) >= min_corner dist(corner, B) - R_q."""
    r = set_min(A.V, B)
    ub = r['upper']                                      # attained by vertex argmin
    witness = A.V[r['argmin']]
    R = A.quad_covering_radius()
    # certified per-vertex lower bounds, capped at ub + (radius of the vertex's largest incident
    # quad tier): a capped corner can then never make its quad a candidate (lq >= ub)
    vt = A.vertex_tier()
    vd = np.empty(len(A.V))
    for k in np.unique(vt):
        m = vt == k
        vd[m] = B.lower(A.V[m], cap=ub + (A.tiers[k]['Lc'] if k >= 0 else 0.0))
    lq = _quad_min(A, vd) - R
    ids = np.unique(A.vid[_corners(A, lq < ub)])
    ids = ids[ids >= 0]
    if ids.size:                                         # exact distances where it matters
        vd[ids] = B.point_dist(A.V[ids])
        lq = _quad_min(A, vd) - R
    cand = lq < ub
    samples = 0
    if refine and refine > 1 and cand.any():
        # per-quad subdivision N_q = ceil(N * Lmax_q / median Lmax) so sub-triangle edges are
        # <= median edge / N everywhere, including stretched quads
        N = int(refine)
        Lmed = float(np.median(A.Lq[A.quad_valid]))
        Nq = np.ceil(N * np.maximum(1.0, A.Lq / max(Lmed, 1e-12))).astype(np.int64)
        qi, qj = np.nonzero(cand)
        o = np.argsort(lq[qi, qj], kind='stable'); qi, qj = qi[o], qj[o]
        g = A.grid.astype(np.float64)
        pos = 0
        while pos < len(qi) and samples < max_samples:
            if lq[qi[pos], qj[pos]] >= ub:
                break
            n0 = int(Nq[qi[pos], qj[pos]])
            step = max(1, 100000 // (n0 + 1) ** 2)
            i, j = qi[pos:pos + step], qj[pos:pos + step]
            same = Nq[i, j] == n0                        # batch quads sharing one subdivision
            if not same.all():
                cut = int(np.argmin(same)); i, j = i[:cut], j[:cut]
            w = _bary(n0)
            a, b, c, e = g[i, j], g[i, j + 1], g[i + 1, j], g[i + 1, j + 1]
            dq = np.full(len(i), np.inf)
            for P0, P1, P2 in ((a, b, e), (a, e, c)):
                pts = (P0[:, None] * (1 - w[:, 0] - w[:, 1])[None, :, None] + P1[:, None] * w[None, :, 0, None]
                       + P2[:, None] * w[None, :, 1, None]).reshape(-1, 3)
                d = B.point_dist(pts).reshape(len(i), len(w))
                k = int(np.argmin(d))
                if d.flat[k] < ub:
                    ub, witness = float(d.flat[k]), pts[k]
                dq = np.minimum(dq, d.min(1))
                samples += pts.shape[0]
            # sub-triangles have edges <= Lmax_q / N_q, hence covering radius <= R_q / N_q
            lq[i, j] = np.maximum(lq[i, j], dq - R[i, j] / n0)
            pos += len(i)
    lb = max(0.0, min(ub, float(lq.min()) if A.quad_valid.any() else ub))
    return {'lower': lb, 'upper': float(ub), 'witness': np.asarray(witness, float).tolist(),
            'refine_samples': int(samples)}


def _corners(S, qmask):
    m = np.zeros(S.valid.shape, bool)
    m[:-1, :-1] |= qmask; m[:-1, 1:] |= qmask; m[1:, :-1] |= qmask; m[1:, 1:] |= qmask
    return m


def clearance(A, B, refine=0, max_samples=20_000_000):
    """Certified bounds on the minimum distance between two triangulated tifxyz surfaces.

    upper: an attained point-to-surface distance (a witness point exists at that distance).
    lower: no pair of surface points is closer. refine=N samples candidate triangles at
    barycentric step 1/N (nearest first, at most max_samples points per direction; the bounds are
    sound whenever it stops). Each quad is subdivided N_q = ceil(N * Lmax_q / median Lmax) times,
    so the gap is at most (median quad edge) / (N sqrt 3)."""
    A, B = as_surface(A), as_surface(B)
    ab = _one_way(A, B, refine, max_samples)
    ba = _one_way(B, A, refine, max_samples)
    ub = min(ab['upper'], ba['upper'])
    lb = min(ub, max(ab['lower'], ba['lower']))
    w = ab['witness'] if ab['upper'] <= ba['upper'] else ba['witness']
    return {'lower': lb, 'upper': ub, 'witness': w, 'Lmax_a': A.L, 'Lmax_b': B.L,
            'refine_samples': ab['refine_samples'] + ba['refine_samples']}


def certify_disjoint(train, pred, margin, refine=0):
    """Train/prediction overlap certificate.

    For every (train, prediction) pair the clearance interval is computed and classified:
      PASS          lower >= margin  (certified: nothing of the prediction surface is within margin)
      FAIL          upper <  margin  (certified: a witness point is closer than margin)
      INCONCLUSIVE  otherwise        (increase refine)
    The overall verdict is FAIL if any pair fails, else INCONCLUSIVE if any is inconclusive, else PASS."""
    rows = []
    for tn, t in train.items():
        for pn, p in pred.items():
            c = clearance(t, p, refine)
            v = 'PASS' if c['lower'] >= margin else ('FAIL' if c['upper'] < margin else 'INCONCLUSIVE')
            rows.append({'train': tn, 'pred': pn, **c, 'verdict': v})
    vs = {r['verdict'] for r in rows}
    overall = 'FAIL' if 'FAIL' in vs else ('INCONCLUSIVE' if 'INCONCLUSIVE' in vs else 'PASS')
    return {'margin': margin, 'verdict': overall, 'pairs': rows}
