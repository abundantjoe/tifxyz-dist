"""Exact point-to-triangulated-surface distance for tifxyz meshes.

Geometry only: operates on x/y/z coordinate grids (voxel units of the volume the mesh lives on).

Triangulation: every grid quad whose 4 corners are valid (x, y, z >= 0 and finite) is split into
triangles (a, b, e) and (a, e, c), a=(i,j) b=(i,j+1) c=(i+1,j) e=(i+1,j+1). Valid vertices that are
in no valid quad are kept as point primitives (distance = vertex distance).

Exactness argument (no densification needed):
  L = max edge length over all triangles, Lc = L / sqrt(3). Every point p of a triangle T lies
  within Lc of some vertex of T (the vertex covering radius of a triangle is its circumradius
  R = l / (2 sin t) <= l / sqrt(3) when acute, t >= 60 deg being the angle opposite the longest
  edge l, and l / 2 when right or obtuse). For a query q, d_v = distance to the nearest vertex
  (KD-tree, exact). The closest surface point p lies in some triangle T with |q-p| = d_s <= d_v,
  and some vertex of T is within d_s + Lc of q. So the exact distance is the min over triangles
  incident to the vertices inside B(q, r) for any r >= d_s + Lc, and d_s >= d_v - Lc. For the
  minimum over a SET of points (a tile) with upper bound ub, only points with d_v - Lc <= ub can
  attain it, and r = ub + Lc suffices for them.

The argument holds per tier of quads with its own Lc (see Surface).
"""
import numpy as np
from scipy.spatial import cKDTree

WORKERS = -1  # scipy cKDTree workers (-1 = all cores)


def closest_point_dist(q, a, b, c):
    """Exact Euclidean distance from points q (n,3) to triangles (a,b,c) (n,3 each), float64.
    Ericson, Real-Time Collision Detection 5.1.5, vectorised; degenerate triangles handled by
    also taking the minimum with the three edge-segment distances."""
    q, a, b, c = (np.asarray(t, np.float64) for t in (q, a, b, c))
    ab = b - a; ac = c - a; ap = q - a
    d1 = (ab * ap).sum(1); d2 = (ac * ap).sum(1)
    bp = q - b; d3 = (ab * bp).sum(1); d4 = (ac * bp).sum(1)
    cp = q - c; d5 = (ab * cp).sum(1); d6 = (ac * cp).sum(1)
    va = d3 * d6 - d5 * d4; vb = d5 * d2 - d1 * d6; vc = d1 * d4 - d3 * d2
    den = va + vb + vc
    with np.errstate(divide='ignore', invalid='ignore'):
        v = vb / den; w = vc / den
    res = a + ab * v[:, None] + ac * w[:, None]          # interior (face region)
    out = np.full(len(q), np.inf)
    face = (va >= 0) & (vb >= 0) & (vc >= 0) & (den > 0)
    out[face] = np.linalg.norm(q[face] - res[face], axis=1)
    # boundary: exact distance to the 3 edge segments (covers vertex and edge regions exactly)
    for s, e in ((a, b), (b, c), (c, a)):
        out = np.minimum(out, seg_dist(q, s, e))
    return out


def seg_dist(q, s, e):
    d = e - s
    dd = (d * d).sum(1)
    with np.errstate(divide='ignore', invalid='ignore'):
        t = np.where(dd > 0, ((q - s) * d).sum(1) / dd, 0.0)
    t = np.clip(t, 0.0, 1.0)
    return np.linalg.norm(q - (s + d * t[:, None]), axis=1)


class Surface:
    """Grid mesh (H, W, 3) float32 with invalid cells NaN (or any coordinate < 0).

    Quads are split into tiers by their longest edge: "regular" quads (longest edge <= tau,
    default 1.25 x median) and doubling bins above it. Each tier has its own vertex KD-tree and covering radius
    Lc_t = (max edge in tier) / sqrt(3), so a few stretched quads do not inflate the search
    radius, or loosen the bounds, for the whole mesh. Any tau is correct; it only affects speed."""

    def __init__(self, grid, tau=None):
        g = np.asarray(grid, np.float32)
        valid = np.isfinite(g).all(-1) & (g >= 0).all(-1)
        self.H, self.W = valid.shape
        self.valid = valid
        qv = valid[:-1, :-1] & valid[:-1, 1:] & valid[1:, :-1] & valid[1:, 1:]
        self.quad_valid = qv
        ii, jj = np.nonzero(valid)
        if len(ii) == 0:
            raise ValueError('mesh has no valid vertices')
        self.vid = np.full(valid.shape, -1, np.int64)
        self.vid[ii, jj] = np.arange(len(ii))
        self.V = g[ii, jj].astype(np.float64)
        self.ij = np.stack([ii, jj], 1).astype(np.int32)
        self.grid = g
        # per-quad longest edge over its two triangles (4 sides + the a-e diagonal)
        Lq = np.zeros(qv.shape)
        a = g[:-1, :-1]; b = g[:-1, 1:]; c = g[1:, :-1]; e = g[1:, 1:]
        for u, w in ((a, b), (a, c), (b, e), (c, e), (a, e)):
            dl = np.linalg.norm((u - w).astype(np.float64), axis=-1)
            Lq = np.maximum(Lq, np.where(qv, dl, 0.0))
        self.Lq = Lq
        self.L = float(Lq.max()) if qv.any() else 0.0
        self.Lc = self.L / np.sqrt(3.0)
        self.tree = cKDTree(self.V)
        self.n_quads = int(qv.sum())
        # tiers by longest quad edge: <= 1.25 med, then doubling bins (1.25, 2.5, 5, ...] x med
        med = float(np.median(Lq[qv])) if qv.any() else 0.0
        t0 = 1.25 * med if tau is None else float(tau)
        edges = [t0]
        while qv.any() and edges[-1] < self.L and edges[-1] > 0:
            edges.append(edges[-1] * 2)
        reg = qv & (Lq <= t0)
        lng = qv & ~reg
        self.qtier = np.where(reg, 0, -1).astype(np.int16)
        self.tiers = [dict(tree=self.tree, vidx=np.arange(len(self.V)), qmask=reg,
                           Lc=float(Lq[reg].max()) / np.sqrt(3.0) if reg.any() else 0.0)]
        for lo, hi in zip(edges[:-1], edges[1:]):
            qm = qv & (Lq > lo) & (Lq <= hi)
            if not qm.any():
                continue
            m = np.zeros(valid.shape, bool)
            m[:-1, :-1] |= qm; m[:-1, 1:] |= qm; m[1:, :-1] |= qm; m[1:, 1:] |= qm
            vidx = self.vid[m]
            self.qtier[qm] = len(self.tiers)
            self.tiers.append(dict(tree=cKDTree(self.V[vidx]), vidx=vidx, qmask=qm,
                                   Lc=float(Lq[qm].max()) / np.sqrt(3.0)))
        self.n_long_quads = int(lng.sum())

    def quad_covering_radius(self):
        """(H-1, W-1) covering radius bound Lmax_q / sqrt(3) per valid quad (0 elsewhere)."""
        return self.Lq / np.sqrt(3.0)

    def vertex_tier(self):
        """(n_vertices,) highest tier index among the quads incident to each vertex (-1: none)."""
        t = np.full(self.valid.shape, -1, np.int16)
        q = self.qtier
        for sl in ((slice(None, -1), slice(None, -1)), (slice(None, -1), slice(1, None)),
                   (slice(1, None), slice(None, -1)), (slice(1, None), slice(1, None))):
            t[sl] = np.maximum(t[sl], q)
        return t[self.ij[:, 0], self.ij[:, 1]]

    def _lower_direct(self, P, cap):
        lb = np.full(len(P), float(cap))
        for t in self.tiers:
            dv, _ = t['tree'].query(P, workers=WORKERS, distance_upper_bound=cap + t['Lc'])
            lb = np.minimum(lb, dv - t['Lc'])                # inf (beyond bound) keeps cap
        return np.maximum(0.0, lb)

    def lower(self, P, cap=np.inf, cells=(1024.0, 256.0, 64.0), min_points=4096):
        """Certified per-point lower bounds on the point-to-surface distance, clipped at `cap`.

        With a finite cap the KD searches are bounded, and points are first screened in cubic
        cells of the given sizes: every point p of a cell with centre c and half-diagonal r has
        d(p) >= d(c) - r, so a cell whose bound reaches cap is settled without per-point work."""
        P = np.asarray(P, np.float64).reshape(-1, 3)
        if not np.isfinite(cap) or len(P) < min_points:
            return self._lower_direct(P, cap)
        out = np.full(len(P), float(cap))
        idx = np.arange(len(P))
        for h in cells:
            if len(idx) < min_points:
                break
            c = np.floor(P[idx] / h).astype(np.int64)
            c -= c.min(0)
            if c.max() >= 1 << 21:                           # key would not be injective
                continue
            key = (c[:, 0] << 42) + (c[:, 1] << 21) + c[:, 2]
            uk, first, inv = np.unique(key, return_index=True, return_inverse=True)
            ctr = (c[first] + np.floor(P[idx].min(0) / h) + 0.5) * h
            r = h * np.sqrt(3.0) / 2
            clb = self._lower_direct(ctr, cap + r) - r
            idx = idx[clb[inv.ravel()] < cap]
        out[idx] = self._lower_direct(P[idx], cap)
        return out

    def _tri_dist_for_vertices(self, q, verts):
        """exact min distance from q (3,) to all triangles incident to vertex ids `verts`."""
        i = self.ij[verts, 0][:, None] + np.array([-1, -1, 0, 0])[None]
        j = self.ij[verts, 1][:, None] + np.array([-1, 0, -1, 0])[None]
        i = i.ravel(); j = j.ravel()
        ok = (i >= 0) & (j >= 0) & (i < self.H - 1) & (j < self.W - 1)
        i, j = i[ok], j[ok]
        ok = self.quad_valid[i, j]
        i, j = i[ok], j[ok]
        if i.size == 0:
            return np.inf
        k = np.unique(i.astype(np.int64) * self.W + j)
        i, j = k // self.W, k % self.W
        g = self.grid
        A = g[i, j]; B = g[i, j + 1]; C = g[i + 1, j]; E = g[i + 1, j + 1]
        Q = np.broadcast_to(q, A.shape)
        d = np.minimum(closest_point_dist(Q, A, B, E), closest_point_dist(Q, A, E, C))
        return float(d.min())

    def _batch_dist(self, Q, base):
        """For points Q (m,3) and base radii (m,): min over the vertex distances and all triangles
        of tier t incident to tier-t vertices within base + Lc_t. Exact wherever the true
        distance d satisfies d <= base (the closest triangle then has a vertex within d + Lc_t)."""
        out = np.full(len(Q), np.inf)
        nq = np.int64((self.H - 1) * (self.W - 1))
        g = self.grid
        for t in self.tiers:
            lists = t['tree'].query_ball_point(Q, base + t['Lc'], workers=WORKERS)
            cnt = np.fromiter((len(x) for x in lists), np.int64, len(lists))
            if cnt.sum() == 0:
                continue
            verts = t['vidx'][np.concatenate([np.asarray(x, np.int64) for x in lists if len(x)])]
            pid = np.repeat(np.arange(len(Q)), cnt)
            dvv = np.linalg.norm(self.V[verts] - Q[pid], axis=1)
            np.minimum.at(out, pid, dvv)
            i = (self.ij[verts, 0][:, None] + np.array([-1, -1, 0, 0])).ravel()
            j = (self.ij[verts, 1][:, None] + np.array([-1, 0, -1, 0])).ravel()
            pq = np.repeat(pid, 4)
            ok = (i >= 0) & (j >= 0) & (i < self.H - 1) & (j < self.W - 1)
            i, j, pq = i[ok], j[ok], pq[ok]
            ok = t['qmask'][i, j]
            i, j, pq = i[ok], j[ok], pq[ok]
            if i.size == 0:
                continue
            key = np.unique(pq * nq + i.astype(np.int64) * (self.W - 1) + j)
            pq = key // nq; lin = key % nq; i = lin // (self.W - 1); j = lin % (self.W - 1)
            for s0 in range(0, len(key), 2_000_000):
                sl = slice(s0, s0 + 2_000_000)
                ii, jj, pp = i[sl], j[sl], pq[sl]
                A = g[ii, jj]; B = g[ii, jj + 1]; C = g[ii + 1, jj]; E = g[ii + 1, jj + 1]; X = Q[pp]
                d = np.minimum(closest_point_dist(X, A, B, E), closest_point_dist(X, A, E, C))
                np.minimum.at(out, pp, d)
        return out

    def min_dist(self, P, stop_vox=0.0, chunk=256, seed=2048):
        """Exact min over points P (n,3) of point-to-surface distance.
        Returns (dist, argmin index, info). If stop_vox > 0, refinement stops once the attained
        distance is <= stop_vox (the returned value is then an attained exact distance of one
        point, i.e. an upper bound on the set minimum, with lower bound info['lower_bound_vox'])."""
        P = np.asarray(P, np.float64).reshape(-1, 3)
        # seed an attained upper bound from a spread subset, then bounded KD searches
        k0 = np.unique(np.linspace(0, len(P) - 1, min(seed, len(P))).astype(np.int64))
        d0 = self.point_dist(P[k0])
        ub = float(d0.min()); arg = int(k0[int(d0.argmin())])
        lbp = self.lower(P, cap=ub)
        order = np.argsort(lbp, kind='stable')
        pos = 0; nref = len(k0); stopped = False
        while pos < len(order):
            if lbp[order[pos]] >= ub:
                break
            if stop_vox > 0 and ub <= stop_vox:
                stopped = True
                break
            sel = order[pos:pos + chunk]
            sel = sel[lbp[sel] < ub]
            pos += chunk
            d = self._batch_dist(P[sel], np.full(len(sel), ub))
            nref += len(sel)
            k = int(np.argmin(d))
            if d[k] < ub:
                ub, arg = float(d[k]), int(sel[k])
        lb = min(ub, float(lbp.min()))
        return ub, arg, {'refined_points': nref, 'early_stop': stopped, 'lower_bound_vox': lb}

    def point_dist(self, P, chunk=4096):
        """Exact per-point distances, computed in chunks of `chunk` points."""
        P = np.asarray(P, np.float64).reshape(-1, 3)
        dv, _ = self.tree.query(P, workers=WORKERS)
        out = dv.copy()
        for s0 in range(0, len(P), chunk):
            sl = slice(s0, s0 + chunk)
            out[sl] = np.minimum(dv[sl], self._batch_dist(P[sl], dv[sl]))
        return out

    def point_dist_loop(self, P):
        """Per-point reference with the looser radius d_v + L (tests only)."""
        P = np.asarray(P, np.float64)
        dv, _ = self.tree.query(P)
        out = dv.copy()
        for k in range(len(P)):
            verts = self.tree.query_ball_point(P[k], dv[k] + self.L)
            out[k] = min(dv[k], self._tri_dist_for_vertices(P[k], np.asarray(verts)))
        return out

    def brute(self, P):
        """O(n * triangles) reference, tests only."""
        P = np.asarray(P, np.float64)
        i, j = np.nonzero(self.quad_valid)
        g = self.grid
        A = g[i, j]; B = g[i, j + 1]; C = g[i + 1, j]; E = g[i + 1, j + 1]
        out = []
        for p in P:
            Q = np.broadcast_to(p, A.shape)
            d = np.minimum(closest_point_dist(Q, A, B, E), closest_point_dist(Q, A, E, C))
            dv = np.linalg.norm(self.V - p, axis=1).min()
            out.append(min(float(d.min()) if d.size else np.inf, dv))
        return np.array(out)
