# tifxyz-dist

Exact, certified distances between tifxyz surfaces and points or other meshes, for train/prediction overlap checks.

`tifxyz-dist` triangulates a tifxyz quadmesh (each valid quad → 2 triangles) and returns:

* **points → surface**: the exact Euclidean distance from any points to the triangulated surface
  (not to its nearest vertex);
* **surface ↔ surface**: an interval `[lower, upper]` that provably contains the minimum distance
  between two segments, with a witness point that attains `upper`;
* **certificate**: for training segments, prediction segments and a required margin, a
  `PASS` / `FAIL` / `INCONCLUSIVE` verdict per pair. `PASS` and `FAIL` are both proven.

It reads mesh coordinates only (`x.tif`, `y.tif`, `z.tif`). No CT volume is needed.

## 1. Why it matters

The Vesuvius Challenge milestone prizes require that training and prediction regions do not
overlap. If an ink model is scored on a surface that lies on, or within a few voxels of, a
training surface, the reported result is not independent evidence of ink. Neighbouring wraps are
often only tens of voxels apart, and independently traced segments of the same sheet can
coincide. Checking this by nearest vertices is not reliable:

* with a 20 vox vertex spacing, the nearest-vertex distance from a point 1 vox above a sheet can
  be 13.1 vox (measured, `demo/accuracy.py`);
* where two sheets approach edge-to-edge, every vertex can be 5.1 vox from the other sheet while
  the sheets are 1.0 vox apart (unit test `test_edge_edge_needs_refine`).

`tifxyz-dist` gives exact point distances and proven lower and upper bounds for mesh-to-mesh
clearance, so an entrant can state, and the team can re-check with one command, the clearance
between a candidate region and every training segment. It also measures sheet spacing and
near-contacts between segments, for example to find segments that duplicate or cross each other
(§6).

## 2. Install

```bash
pip install git+https://github.com/abundantjoe/tifxyz-dist
# from a checkout, with the tests:
pip install -e '.[test]' && pytest -q
```

Python ≥ 3.9. Dependencies: numpy, scipy, tifffile, imagecodecs (for LZW-compressed tifxyz files).
Tested on Linux x86-64 with Python 3.11. A GitHub Actions workflow runs the tests on Python 3.9 and
3.12 (`.github/workflows/test.yml`).

## 3. Quick start on public data

```bash
bash demo/quickstart.sh            # downloads ~6 MB of public PHerc0172 mesh coordinates
```

This fetches two public PHerc0172 segments traced on volume `20241024131838` (7.91 µm voxels),
`20250917143559-w062…` and `20250926112011-w078…`, and runs `tifxyz-dist clearance` on them.
Expected output (7.5 s on a 4-core container):

```json
{
 "lower": 0.0,
 "upper": 0.11009975877937576,
 "witness": [6755.7509765625, 2695.60595703125, 7445.3388671875],
 "Lmax_a": 55.08481698146976,
 "Lmax_b": 227.5809296933753,
 "refine_samples": 0
}
```

So the two segments come within 0.11 vox (0.9 µm) of each other at the witness point: they touch
or cross. They must not be split across a training set and a validation set.

## 4. CLI and API reference

All distances are in voxels of the volume the meshes were traced on. Both meshes of a comparison
must be on the same volume. A tifxyz input is a directory with `x.tif`, `y.tif`, `z.tif`; vertices
with any coordinate `< 0` or non-finite are invalid, and quads with an invalid corner are dropped.

```bash
tifxyz-dist points    SEGMENT.tifxyz POINTS [--out D.npy]
tifxyz-dist clearance A.tifxyz B.tifxyz [--refine N]
tifxyz-dist certify   --train T1.tifxyz [T2 ...] --pred P1.tifxyz [P2 ...] --margin M [--refine N]
```

| command | input | JSON on stdout | exit code |
|---|---|---|---|
| `points` | `POINTS`: `.npy` (n×3) or `.csv`/`.txt` with x, y, z columns | `{n, min, argmin, median, max, exact: true}`; `--out` writes all n exact distances (float64 `.npy`) | 0 |
| `clearance` | two tifxyz | `{lower, upper, witness: [x,y,z], Lmax_a, Lmax_b, refine_samples}` | 0 |
| `certify` | tifxyz lists, margin in vox | `{margin, verdict, pairs: [{train, pred, lower, upper, witness, Lmax_a, Lmax_b, refine_samples, verdict}]}` | 1 if any pair is `FAIL`, else 0 |

`--refine N` subdivides the candidate quads (`N_q = ⌈N · Lmax_q / median quad edge⌉`), which
narrows `upper − lower` to at most (median quad edge)/(N√3). `refine=0` is fastest; the bounds
are valid either way.

```python
from tifxyz_dist import Surface, load_grid, points_to_surface, set_min, clearance, certify_disjoint
A = Surface(load_grid("a.tifxyz")); B = Surface(load_grid("b.tifxyz"))
d = points_to_surface(P, A)                    # (n,) exact distances
m = set_min(P, A)                              # {'lower', 'upper', 'argmin', ...}, exact
c = clearance(A, B, refine=8)                  # certified interval + witness
r = certify_disjoint({"t": A}, {"p": B}, margin=20.0)
```

### How it works, and why the bounds hold

For a triangle with longest edge `l`, every point lies within `l/√3` of one of its vertices (the
covering radius is the circumradius when the triangle is acute, `≤ l/√3`, and `l/2` otherwise).

1. **Tiers.** Quads are grouped by their longest edge: `≤ 1.25 ×` the median, then doubling bins.
   Each tier `t` has its own vertex KD-tree (SciPy `cKDTree`) and radius
   `Lc_t = (max edge in tier)/√3`. Real traced meshes have a few stretched quads (up to 516 vox on a
   20 vox grid in the demo data). With tiers, they do not inflate the radius used for the rest of
   the mesh: 20.3 vox instead of 297.7 vox on the worst demo mesh.
2. **Exact point distance without densifying.** Let `d_v` be the nearest-vertex distance of a query
   `q`. The closest surface point, at distance `d ≤ d_v`, lies in a triangle of some tier `t`, and
   one vertex of that triangle is within `d + Lc_t ≤ d_v + Lc_t` of `q`. The tool tests every
   tier-`t` triangle incident to a tier-`t` vertex in that ball. It uses an exact point–triangle
   routine (Ericson, *Real-Time Collision Detection*, §5.1.5), plus exact edge-segment distances
   for degenerate triangles.
3. **Lower bounds and set minima.** `d ≥ min_t (d_v,t − Lc_t)`. Points whose lower bound is above
   the best distance found so far cannot hold the minimum, so they are skipped. KD searches are
   bounded by that distance. Points are first screened in cubic cells of 1024, 256 and 64 vox: for
   a cell with centre `c` and half-diagonal `r`, `d(p) ≥ d(c) − r` for every point `p` in it.
4. **Surface ↔ surface.** Distance to a surface is 1-Lipschitz. For a quad `q` of A with
   `R_q = Lmax_q/√3`, every point of `q` is at least `min_corner d(corner, B) − R_q` from B. That
   gives the lower bound; corner distances are exact wherever the bound matters. The minimum over
   A's vertices is attained, which gives the upper bound and the witness. Both directions are
   combined. Refinement samples candidate quads nearest-first and is anytime: stopping at the
   sample budget still leaves both bounds valid.
5. **Certificate.** `PASS` iff `lower ≥ margin`. `FAIL` iff `upper < margin`, with a witness.
   Otherwise `INCONCLUSIVE`; raise `--refine`. Floating-point rounding is the only error source:
   ~1e-12 vox in the tests below.

## 5. Validation

All numbers come from `pytest -q` and `python demo/accuracy.py <MESH>`, where `<MESH>` is the public
segment PHerc0172 `20250917143559-w062_20250917143559205_flatboi`, volume `20241024131838`.

| check | result |
|---|---|
| unit tests (`pytest -q`) | **25 / 25 pass** |
| triangle kernel vs an independent implementation (plane projection + barycentric solve), 100 000 random point/triangle pairs, including 10 % slivers | max abs diff 2.1e-14 |
| KD-tree + covering-bound pruning vs testing every triangle: 4 000 queries on 20 noisy curved sheets with 5 % holes | max abs diff 0.0 vox |
| same comparison in the unit tests, including folds, a per-point loop and a worst-case equilateral lattice | agree to ≤ 1e-6 vox |
| points sampled on the triangles themselves, synthetic sheets | max distance 6.4e-13 vox |
| 100 000 points sampled on the triangles of the public segment `w062` (386 732 vertices, 385 047 quads) | max distance 2.7e-12 vox; ~19 000 points/s (4-core container) |
| meshes with stretched quads (tiers): 1 600 queries vs every triangle | max abs diff ≤ 1e-6 vox; lower bounds never above the truth |
| cell-screened, capped lower bounds: 6 000 queries, including negative coordinates, caps 5 vox to ∞ | never above the truth |
| points 1 vox above a plane meshed at 20 vox | nearest-vertex error up to 13.1 vox; exact method error 0.0 |
| analytic cases: plane heights, V fold (inside, below the crease, wing normal), U fold between layers, hole rim, beyond the edge | exact to 1e-6 vox |
| clearance, parallel planes 12.5 vox apart | upper = 12.5 |
| clearance, edge-to-edge sheets 1.0 vox apart, every vertex ≥ 5.1 vox away | unrefined `[0, 5.1]` (valid); `--refine 20`: upper = 1.000000, lower = 0.18 > 0 (proven not touching) |
| clearance, random curved sheet pairs vs dense sampling of both surfaces | interval always contains the reference |
| clearance next to a 400 vox quad column, sheets 10 vox apart, `--refine 8` | upper = 10.000000; lower ≥ 10 − 2.04 (not loosened by the long quads) |
| speed on public meshes, unrefined (386 732 × 1 602 247 vertices) | 3.2 s for touching segments; 9.6 s for segments 634 vox apart |

## 6. Findings on real data (PHerc0172)

`demo/fetch_pherc0172.sh OUT` downloads the coordinates of every public PHerc0172 segment traced on
volume `20241024131838` (53 segments). `python demo/pairwise.py OUT pairs.jsonl --pred auto_grown`
then computes the certified clearance between each of the 44 manually traced `w0xx` segments and
each of the 9 `auto_grown_20251115002740308_*` segments. The 9 auto-grown segments play the role of
predictions. The full per-pair table and summary are in [`results/`](results/)
(`python demo/summarize.py results/pairs_auto.jsonl 7.91` regenerates `results/RESULTS.md`).

Frozen at publication: **243 of the 396 pairs** (27 of the 44 manual segments × 9), 44.6 min of
compute, median 10.0 s per pair, unrefined. The remaining pairs can be computed with the same
command (it resumes). Summary (`results/RESULTS.md`):

interval width (upper - lower): median 16.29 vox, max 94.90 vox

| margin (vox) |  margin (um) | PASS | FAIL | INCONCLUSIVE |
|---|---|---|---|---|
| 1 | 8 | 228 | 14 | 1 |
| 5 | 40 | 225 | 17 | 1 |
| 20 | 158 | 225 | 18 | 0 |
| 50 | 396 | 225 | 18 | 0 |
| 100 | 791 | 223 | 19 | 1 |

What this shows, as measured:

* **w078 touches or crosses all 9 auto-grown segments**: for each pair, an interior mesh vertex
  (4 incident valid quads) of one segment lies 0.003 to 0.139 vox (0.02 to 1.1 µm) from the other. If w078 were used for training and any auto-grown segment for prediction
  (or the reverse), the regions would overlap. Witness coordinates are in `results/RESULTS.md`.
* **w066 comes within 0.26 to 0.55 vox of 5 auto-grown segments, but only through isolated
  vertices**: valid w066 vertices that belong to no valid quad. The tool counts such vertices as
  point primitives. For these pairs, the lower bound shows that w066's triangulated surface itself
  is no closer than the reported `lower`. This is a mesh-hygiene finding: stray vertices in a
  tifxyz can create apparent contacts.
* At a 20 vox (158 µm) margin, every computed pair is decided: 225 `PASS`, 18 `FAIL`, 0
  `INCONCLUSIVE`, without refinement.
* The unrefined interval width is 16.3 vox (median). It is 94.9 vox for one pair (w088 ×
  auto_grown_4, `[300.0, 394.9]`), where stretched quads dominate; `--refine` narrows it.

## 7. Limitations and known failure modes

* Without `--refine`, the surface-to-surface lower bound can be looser than the truth by up to the
  quad covering radius: about 16 vox on a 20 vox grid (measured median interval width in §6).
  `--refine` narrows it. An exact triangle–triangle bound would remove this gap; it is not
  implemented yet.
* `lower = 0` with a tiny `upper` means "touching or crossing within `upper`". The tool does not
  separately prove that two sheets cross.
* Distances are to the piecewise-linear mesh, not to the physical papyrus sheet. The mesh can be
  wrong; the tool measures the mesh.
* Both meshes must be in one coordinate frame (same volume). The tool does not resample between
  volumes.
* Input coordinates are float32 and the arithmetic is float64. The error figures above are for
  this setting.
* Memory grows with the number of candidate (point, triangle) pairs. `point_dist` works in chunks.
  The whole demo process, holding all 53 PHerc0172 segments in memory (up to 1.6 M vertices each),
  stayed under 3 GB resident.
* A certificate is about geometry only. It says nothing about ink.

## 8. Data and citation

The demo, quick start and real-mesh validation use public PHerc0172 (Scroll 5) segments from the
Vesuvius Challenge open-data bucket (`s3://vesuvius-challenge-open-data/PHerc0172/segments/`,
volume `20241024131838`, 7.91 µm). Segment IDs are listed in `results/pairs_auto.jsonl`; browse them
in the [Data Browser](https://scrollprize.org/data_browser). If you use these data, cite:

> Giorgio Angelotti, Stephen Parsons, Sean Johnson, Elian Rafael Dal Prà, Johannes Rudolph, Paul Tafforeau, Alessandro Mirone, Paul Henderson, Hendrik Schilling, Forrest McDonald, David Josey, Youssef Nader, C. Seth Parker, W. Brent Seales. *Vesuvius Challenge - CT Scans of Herculaneum Papyri*. Vesuvius Challenge.

Link: https://scrollprize.org/data

### Licences

* Code: MIT (see `LICENSE`).
* `results/` holds per-pair numbers derived from Vesuvius Challenge data. They are licensed
  **CC BY-NC 4.0** (see `results/LICENSE-DATA.md`). No CT volumes or meshes are included; the
  scripts download mesh coordinates from the official bucket.

### Credits

* tifxyz format and the VC3D tooling: Vesuvius Challenge, https://github.com/scrollprize/villa.
* SciPy `cKDTree`: P. Virtanen et al. "SciPy 1.0: fundamental algorithms for scientific computing in
  Python." *Nature Methods* 17, 261–272 (2020).
* Point–triangle distance: C. Ericson, *Real-Time Collision Detection*, Morgan Kaufmann, 2005, §5.1.5.
* Related community tool: the July 2026 progress prize to **Josep Carrera** "for a tool that
  detects intersections in tifxyz patches" (Vesuvius Challenge, "$33.5K awarded in July", Substack,
  2026-08-06, https://scrollprize.substack.com/p/335k-awarded-in-july). The post gives no repository
  link, and we have not run or benchmarked that tool. Compared with intersection detection, this
  tool adds:
  * how far apart segments are, as a proven `[lower, upper]` interval, not only whether they touch;
  * exact distances from arbitrary point sets (labels, predictions) to a segment;
  * a train/prediction certificate at a chosen margin, with a witness point when it fails;
  * near-contacts within a margin that do not intersect, including edge-to-edge approaches that
    vertex-based checks overestimate.

## 9. AI assistance

Developed with the help of AI coding agents; all numbers are produced by the scripts and tests in this repository.

## 10. Contributing and issues

Bug reports, failing cases on real segments, and pull requests are welcome through GitHub issues.
To report a wrong distance, include the segment IDs, the command and the JSON output.
