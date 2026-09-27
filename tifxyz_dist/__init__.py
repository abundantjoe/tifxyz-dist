"""tifxyz-dist: exact, certified distances to and between tifxyz surfaces."""
from .core import Surface, closest_point_dist
from .ops import points_to_surface, set_min, clearance, certify_disjoint
from .io import load_grid, save_grid, load_points

__version__ = '0.1.0'
__all__ = ['Surface', 'closest_point_dist', 'points_to_surface', 'set_min', 'clearance',
           'certify_disjoint', 'load_grid', 'save_grid', 'load_points']
