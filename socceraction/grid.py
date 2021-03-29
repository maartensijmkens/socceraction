import numpy as np
import pandas as pd

from typing import Tuple

import socceraction.spadl.config as spadlconfig


class Grid:
    """Interface defining the expected methods for a custom grid layout"""

    def get_length(self):
        raise Exception("Not implemented")

    def get_cell(self, x: pd.Series, y: pd.Series) -> pd.Series:
        raise Exception("Not implemented")


class DefaultGrid(Grid):
    """Default layout of a M x N grid"""

    def __init__(self, N: int = 16, M: int = 12):
        self.N = N
        self.M = M

    def get_length(self):
        return self.M * self.N

    def _get_cell_index(self, x: pd.Series, y: pd.Series) -> Tuple[pd.Series, pd.Series]:
        xmin = 0
        ymin = 0
        xi = (x - xmin) / spadlconfig.field_length * self.N
        yj = (y - ymin) / spadlconfig.field_width * self.M
        xi = xi.astype(int).clip(0, self.N - 1)
        yj = yj.astype(int).clip(0, self.M - 1)
        return xi, yj

    def get_cell(self, x: pd.Series, y: pd.Series) -> pd.Series:
        xi, yj = self._get_cell_index(x, y)
        return self.N * (self.M - 1 - yj) + xi


class PolarGrid(Grid):
    """Polar grid layout"""

    def __init__(self, N: int = 8, M: int = 11):
        self.N = N
        self.M = M

    def get_length(self):
        return 2 * (self.N + 1) * self.M

    def _get_cell_index(self, x: pd.Series, y: pd.Series) -> Tuple[pd.Series, pd.Series]:
        xmin = 0
        ymin = 0

        halfx = (spadlconfig.field_length - xmin) / 2
        halfy = (spadlconfig.field_width - ymin) / 2

        s = ((x - xmin) > halfx).astype(int) 
        r1 = (x - xmin) ** 2 + (y - ymin - halfy) ** 2
        r2 = (spadlconfig.field_length - (x - xmin)) ** 2 + (y - ymin - halfy) ** 2
        ri = np.clip(np.sqrt(((s == 0) * r1 + (s == 1) * r2) / (halfx ** 2)) * self.N,0,self.N-1)
        yj = (y - ymin) / spadlconfig.field_width * self.M
        ri = ri.astype(int).clip(0, self.N)
        yj = yj.astype(int).clip(0, self.M - 1)

        return s, ri, yj

    def get_cell(self, x: pd.Series, y: pd.Series) -> pd.Series:
        
        s, ri, yj = self._get_cell_index(x, y)
        return s * (self.N + 1) * self.M + ri * self.M + yj




