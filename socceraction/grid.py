import numpy as np
import pandas as pd

from typing import Tuple, Callable

import socceraction.spadl.config as spadlconfig

M: int = 12
N: int = 16

class Grid:
    """Interface defining the expected methods for a custom grid layout"""

    def _get_length(self):
        raise Exception("Not implemented")

    def _get_flat_indexes(self, x: pd.Series, y: pd.Series, use_interpolation: bool = False) -> pd.Series:
        raise Exception("Not implemented")

    def _interpolate(self, z: np.ndarray):
        raise Exception("Not implemented")


class DefaultGrid(Grid):
    """Default layout of a 12 x 16 grid"""

    def __init__(self, l: int = N, w: int = M):
        self.l = l
        self.w = w

    def _get_length(self):
        return self.w * self.l

    def _get_cell_indexes(self, x: pd.Series, y: pd.Series, l: int, w: int) -> Tuple[pd.Series, pd.Series]:
        xmin = 0
        ymin = 0

        xi = (x - xmin) / spadlconfig.field_length * l
        yj = (y - ymin) / spadlconfig.field_width * w
        xi = xi.astype(int).clip(0, l - 1)
        yj = yj.astype(int).clip(0, w - 1)
        return xi, yj

    def _get_flat_indexes(self, x: pd.Series, y: pd.Series, use_interpolation: bool = False) -> pd.Series:

        if not use_interpolation:
            xi, yj = self._get_cell_indexes(x,y, self.l, self.w)
            return self.l * (self.w - 1 - yj) + xi

        else:
            l = int(spadlconfig.field_length * 10)
            w = int(spadlconfig.field_width * 10)
            xi, yj = self._get_cell_indexes(x, y, l, w)
            return l * (w - 1 - yj) + xi


    def interpolator(self, z: np.ndarray, kind: str = "linear") -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        from scipy.interpolate import interp2d  # type: ignore

        cell_length = spadlconfig.field_length / self.l
        cell_width = spadlconfig.field_width / self.w

        x = np.arange(0.0, spadlconfig.field_length, cell_length) + 0.5 * cell_length
        y = np.arange(0.0, spadlconfig.field_width, cell_width) + 0.5 * cell_width

        return interp2d(x=x, y=y, z=z, kind=kind, bounds_error=False)

    def _interpolate(self, z: np.ndarray):
        # Use interpolation to create a
        # more fine-grained 1050 x 680 grid
        interp = self.interpolator(z.reshape((self.w, self.l)))
        l = int(spadlconfig.field_length * 10)
        w = int(spadlconfig.field_width * 10)
        xs = np.linspace(0, spadlconfig.field_length, l)
        ys = np.linspace(0, spadlconfig.field_width, w)
        z_interpolated = interp(xs, ys)
        return z_interpolated.flatten()


class PolarGrid(Grid):
    """Polar grid layout"""

    def __init__(self, l: int = 8, w: int = 11):
        self.l = l
        self.w = w

    def _get_length(self):
        return 2 * (self.l + 1) * self.w

    def _get_cell_indexes(self, x: pd.Series, y: pd.Series, l: int, w: int) -> Tuple[pd.Series, pd.Series]:
        xmin = 0
        ymin = 0

        halfx = (spadlconfig.field_length - xmin) / 2
        halfy = (spadlconfig.field_width - ymin) / 2

        s = ((x - xmin) > halfx).astype(int) 
        r1 = (x - xmin) ** 2 + (y - ymin - halfy) ** 2
        r2 = (spadlconfig.field_length - (x - xmin)) ** 2 + (y - ymin - halfy) ** 2
        ri = ((s == 0) * r1 + (s == 1) * r2) / (halfx ** 2) * l 
        yj = (y - ymin) / spadlconfig.field_width * w
        ri = ri.astype(int).clip(0, l)
        yj = yj.astype(int).clip(0, w - 1)

        return s, ri, yj

    def _get_flat_indexes(self, x: pd.Series, y: pd.Series, use_interpolation: bool = False) -> pd.Series:
        
        s, ri, yj = self._get_cell_indexes(x, y, self.l, self.w)
        return s * (self.l + 1) * self.w + ri * self.w + yj




