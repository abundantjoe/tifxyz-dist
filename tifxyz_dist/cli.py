"""Command line interface: tifxyz-dist {points,clearance,certify}."""
import argparse
import json
import sys

import numpy as np

from .core import Surface
from .io import load_grid, load_points
from . import ops


def _surf(path):
    return Surface(load_grid(path))


def main(argv=None):
    ap = argparse.ArgumentParser(prog='tifxyz-dist', description=__doc__)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('points', help='exact distance from points (.npy / .csv, x y z) to a tifxyz')
    p.add_argument('tifxyz'); p.add_argument('points')
    p.add_argument('--out', help='write per-point distances (.npy); default prints summary only')
    c = sub.add_parser('clearance', help='certified min distance between two tifxyz surfaces')
    c.add_argument('a'); c.add_argument('b'); c.add_argument('--refine', type=int, default=0)
    k = sub.add_parser('certify', help='train/prediction overlap certificate')
    k.add_argument('--train', nargs='+', required=True); k.add_argument('--pred', nargs='+', required=True)
    k.add_argument('--margin', type=float, required=True, help='required clearance, voxels')
    k.add_argument('--refine', type=int, default=0)
    a = ap.parse_args(argv)
    if a.cmd == 'points':
        d = ops.points_to_surface(load_points(a.points), _surf(a.tifxyz))
        if a.out:
            np.save(a.out, d)
        res = {'n': int(d.size), 'min': float(d.min()), 'argmin': int(d.argmin()),
               'median': float(np.median(d)), 'max': float(d.max()), 'exact': True}
    elif a.cmd == 'clearance':
        res = ops.clearance(_surf(a.a), _surf(a.b), a.refine)
    else:
        res = ops.certify_disjoint({t: _surf(t) for t in a.train}, {q: _surf(q) for q in a.pred},
                                   a.margin, a.refine)
    json.dump(res, sys.stdout, indent=1); print()
    return 1 if res.get('verdict') == 'FAIL' else 0


if __name__ == '__main__':
    sys.exit(main())
