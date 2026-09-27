"""All-pairs certified clearance between tifxyz segments (resumable).

Usage: python demo/pairwise.py MESH_DIR OUT.jsonl [--refine N] [--pred SUBSTR]
With --pred, only (other, pred) pairs are computed, where pred segment names contain SUBSTR.
Each output line: {"a", "b", "lower", "upper", "witness", "seconds"} in voxels of the volume."""
import argparse, itertools, json, os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from tifxyz_dist import Surface, clearance, load_grid

ap = argparse.ArgumentParser()
ap.add_argument('mesh_dir'); ap.add_argument('out'); ap.add_argument('--refine', type=int, default=0)
ap.add_argument('--pred', default=None)
a = ap.parse_args()
names = sorted(d for d in os.listdir(a.mesh_dir) if os.path.exists(os.path.join(a.mesh_dir, d, 'x.tif')))
done = set()
if os.path.exists(a.out):
    done = {(r['a'], r['b']) for r in map(json.loads, open(a.out))}
S = {}
def surf(n):
    if n not in S:
        S[n] = Surface(load_grid(os.path.join(a.mesh_dir, n)))
    return S[n]
# cheap axis-aligned bounding-box gap first: a certified lower bound that skips far pairs
bb = {n: (lambda g: (g.min((0, 1)), g.max((0, 1))))(surf(n).V) for n in names}
with open(a.out, 'a') as f:
    pairs = itertools.combinations(names, 2)
    if a.pred:
        pairs = [(x, y) for x in names if a.pred not in x for y in names if a.pred in y]
    for x, y in pairs:
        if (x, y) in done:
            continue
        t0 = time.time()
        gap = float(((bb[x][0] - bb[y][1]).clip(0) ** 2 + (bb[y][0] - bb[x][1]).clip(0) ** 2).sum() ** 0.5)
        if gap > 50:
            r = {'lower': gap, 'upper': None, 'witness': None, 'bbox_only': True}
        else:
            r = clearance(surf(x), surf(y), a.refine)
        r.update(a=x, b=y, seconds=round(time.time() - t0, 2))
        f.write(json.dumps(r) + '\n'); f.flush()
