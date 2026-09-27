"""Read tifxyz meshes (a directory holding x.tif, y.tif, z.tif and usually meta.json)."""
import json
import os

import numpy as np


def load_grid(path):
    """Return the (H, W, 3) float32 vertex grid of a tifxyz directory.

    Invalid vertices (stored as -1, or non-finite) are set to NaN. Coordinates stay in the
    voxel units of the volume the mesh was traced on; only coordinate arrays are read."""
    import tifffile
    g = np.stack([tifffile.imread(os.path.join(path, f'{a}.tif')) for a in 'xyz'], -1).astype(np.float32)
    g[~(np.isfinite(g).all(-1) & (g >= 0).all(-1))] = np.nan
    return g


def load_meta(path):
    p = os.path.join(path, 'meta.json')
    return json.load(open(p)) if os.path.exists(p) else {}


def save_grid(path, grid, meta=None):
    """Write a (H, W, 3) grid as a tifxyz directory (invalid -> -1). Used for tests and demos."""
    import tifffile
    os.makedirs(path, exist_ok=True)
    g = np.asarray(grid, np.float32).copy()
    g[~(np.isfinite(g).all(-1) & (g >= 0).all(-1))] = -1
    for k, a in enumerate('xyz'):
        tifffile.imwrite(os.path.join(path, f'{a}.tif'), g[..., k])
    json.dump(meta or {'format': 'tifxyz', 'type': 'seg'}, open(os.path.join(path, 'meta.json'), 'w'))


def load_points(path):
    """Points as (n, 3) float64 in x, y, z order: .npy, or .csv/.txt with 3 columns (header allowed)."""
    if path.endswith('.npy'):
        P = np.load(path)
    else:
        P = np.genfromtxt(path, delimiter=',' if path.endswith('.csv') else None, invalid_raise=True)
        P = P[np.isfinite(P).all(-1)] if P.ndim == 2 else P
    return np.asarray(P, np.float64).reshape(-1, 3)
