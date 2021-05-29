import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from typing import Tuple

import socceraction.spadl.config as spadlconfig


class Grid:
    """Interface defining the expected methods for a custom grid layout"""

    def get_length(self):
        raise Exception("Not implemented")

    def get_cell(self, x: pd.Series, y: pd.Series) -> pd.Series:
        raise Exception("Not implemented")

    def visualize(self):
        L = 105
        W = 68
        xx, yy = np.meshgrid(np.arange(0, L, 1), np.arange(0, W, 1))
        c = np.c_[xx.ravel(), yy.ravel()]
        Z = self.get_cell(pd.Series(c[:,0]), pd.Series(c[:,1]))
        Z = Z.to_numpy().reshape((W,L))
        plt.pcolormesh(xx, yy, Z, shading='nearest')
        # plt.contour(Z, levels=self.get_length(), colors='black', linewidths=1.0)


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


def getPolarCoords(x: pd.Series, y:pd.Series):
        xmin = 0
        ymin = 0

        halfx = (spadlconfig.field_length - xmin) / 2
        halfy = (spadlconfig.field_width - ymin) / 2

        s = ((x - xmin) > halfx).astype(int)
        d0 = np.sqrt((x - xmin) ** 2 + (y - ymin - halfy) ** 2)
        d1 = np.sqrt((spadlconfig.field_length - (x - xmin)) ** 2 + (y - ymin - halfy) ** 2)
        distance = (s == 0) * d0 + (s == 1) * d1
        a0 = np.arctan2((y - ymin - halfy), (x - xmin)) / np.pi * 90
        a1 = np.arctan2((y - ymin - halfy), (spadlconfig.field_length - (x - xmin))) / np.pi * 90
        angle = (s == 0) * a0 + (s == 1) * a1

        return s, distance, angle


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


class ClusteringGrid(Grid):
    """Grid based on kmeans clustering on x and y coordinates"""

    def __init__(self, N: int = 50):
        self.N = N
        self.kmeans = KMeans(n_clusters=N, random_state=0)

    def fit(self, x: pd.Series, y: pd.Series):
        X = np.stack((x.to_numpy(), y.to_numpy()), axis=1)
        self.kmeans.fit(X)
        return self

    def get_length(self):
        return self.N

    def get_cell(self, x: pd.Series, y: pd.Series) -> pd.Series:
        X = np.stack((x.to_numpy(), y.to_numpy()), axis=1)
        return pd.Series(self.kmeans.predict(X), index=x.index)


class PolarClusteringGrid(Grid):
    """Grid based on kmeans clustering on side, distance and angle"""

    def __init__(self, N: int = 50):
        self.N = N
        self.kmeans = KMeans(n_clusters=N, random_state=0)

    def fit(self, x: pd.Series, y: pd.Series):
        side, distance, angle = getPolarCoords(x, y)
        distance /= 50
        angle /= 60
        X = np.stack((side, distance, angle), axis=1)
        self.kmeans.fit(X)
        return self

    def get_length(self):
        return self.N

    def get_cell(self, x: pd.Series, y: pd.Series) -> pd.Series:
        side, distance, angle = getPolarCoords(x, y)
        distance /= 50
        angle /= 60
        X = np.stack((side, distance, angle), axis=1)
        return pd.Series(self.kmeans.predict(X), index=x.index)

