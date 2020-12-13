import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import warnings  # type: ignore

from typing import Tuple, List

import socceraction.spadl.config as spadlconfig
from socceraction.grid import Grid, CartesianGrid


def _count(x: pd.Series, y: pd.Series, grid: Grid) -> np.ndarray:
    """ Count the number of actions occurring in each cell of the grid.

    :param x: The x-coordinates of the actions.
    :param y: The y-coordinates of the actions.
    :return: A matrix, denoting the amount of actions occurring in each cell. The top-left corner is the origin.
    """
    x = x[~np.isnan(x) & ~np.isnan(y)]
    y = y[~np.isnan(x) & ~np.isnan(y)]

    n = grid._get_cell_amount()

    flat_indexes = grid._get_flat_indexes(x, y)
    vc = flat_indexes.value_counts(sort=False)
    vector = np.zeros(n)
    vector[vc.index] = vc
    return vector


def safe_divide(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.divide(a, b, out=np.zeros_like(a), where=b != 0)


def scoring_prob(actions: pd.DataFrame, grid: Grid) -> np.ndarray:
    """ Compute the probability of scoring when taking a shot for each cell.

    :param actions: Actions, in SPADL format.
    :param grid: Grid object.
    :return: A matrix, denoting the probability of scoring for each cell.
    """
    shot_actions = actions[(actions.type_name == "shot")]
    goals = shot_actions[(shot_actions.result_name == "success")]

    shotmatrix = _count(shot_actions.start_x, shot_actions.start_y, grid)
    goalmatrix = _count(goals.start_x, goals.start_y, grid)
    return safe_divide(goalmatrix, shotmatrix)


def get_move_actions(actions: pd.DataFrame) -> pd.DataFrame:
    return actions[
        (actions.type_name == "pass")
        | (actions.type_name == "dribble")
        | (actions.type_name == "cross")
    ]

def get_defensive_actions(actions: pd.DataFrame) -> pd.DataFrame:
    return actions[
        (actions.type_name == "interception")
        | (actions.type_name == "tackle")
    ]

def get_successful(actions: pd.DataFrame) -> pd.DataFrame:
    return actions[actions.result_name == "success"]

def get_failed(actions: pd.DataFrame) -> pd.DataFrame:
    return actions[actions.result_name == "fail"]


def action_prob(actions: pd.DataFrame, grid: Grid) -> Tuple[np.ndarray, np.ndarray]:
    """ Compute the probability of taking an action in each cell of the grid. The options are: shooting or moving.

    :param actions: Actions, in SPADL format.
    :param grid: Grid object.
    :return: 2 matrices, denoting for each cell the probability of choosing to shoot
    and the probability of choosing to move.
    """
    move_actions = get_move_actions(actions)
    shot_actions = actions[(actions.type_name == "shot")]

    movematrix = _count(move_actions.start_x, move_actions.start_y, grid)
    shotmatrix = _count(shot_actions.start_x, shot_actions.start_y, grid)
    totalmatrix = movematrix + shotmatrix

    return safe_divide(shotmatrix, totalmatrix), safe_divide(movematrix, totalmatrix)


def move_transition_matrix(actions: pd.DataFrame, grid: Grid) -> Tuple[np.ndarray, np.ndarray]:
    """ Compute the move transition matrix from the given actions.

    This is, when a player chooses to move, the probability that he will
    end up in each of the other cells of the grid successfully.

    :param actions: Actions, in SPADL format.
    :param grid: Grid object.
    :return: The transition matrix.
    """
    move_actions = get_move_actions(actions)
    n = grid._get_cell_amount()

    X = pd.DataFrame()
    X["start_cell"] = grid._get_flat_indexes(move_actions.start_x, move_actions.start_y)
    X["end_cell"] = grid._get_flat_indexes(move_actions.end_x, move_actions.end_y)
    X["result_name"] = move_actions.result_name

    vc = X.start_cell.value_counts(sort=False)
    start_counts = np.zeros(n)
    start_counts[vc.index] = vc

    transition_matrix = np.zeros((n, n))

    for i in range(0, n):
        vc2 = X[
            ((X.start_cell == i) & (X.result_name == "success"))
        ].end_cell.value_counts(sort=False)
        transition_matrix[i, vc2.index] = vc2 / start_counts[i]

    return transition_matrix


class ExpectedThreat:
    """An implementation of Karun Singh's Expected Threat model (https://karun.in/blog/expected-threat.html)."""

    def __init__(self, grid: Grid = CartesianGrid(), use_interpolation: bool = True, eps: float = 1e-5):
        self.grid = grid
        self.n = grid._get_cell_amount()
        self.eps = eps
        self.use_interpolation = use_interpolation
        self.heatmaps: List[np.ndarray] = []
        self.xT: np.ndarray = np.zeros(self.n)
        self.scoring_prob_matrix: np.ndarray = np.zeros(self.n)
        self.shot_prob_matrix: np.ndarray = np.zeros(self.n)
        self.move_prob_matrix: np.ndarray = np.zeros(self.n)
        self.transition_matrix: np.ndarray = np.zeros((self.n, self.n))

    def __solve(
        self,
        p_scoring: np.ndarray,
        p_shot: np.ndarray,
        p_move: np.ndarray,
        transition_matrix: np.ndarray,
    ) -> None:
        """Solves the expected threat equation with dynamic programming.

        :param p_scoring (matrix, shape(M, N)): Probability of scoring at each grid cell, when shooting from that cell.
        :param p_shot (matrix, shape(M,N)): For each grid cell, the probability of choosing to shoot from there.
        :param p_move (matrix, shape(M,N)): For each grid cell, the probability of choosing to move from there.
        :param transition_matrix (matrix, shape(M*N,M*N)): When moving, the probability of moving to each of the other zones.
        """
        gs = p_scoring * p_shot
        diff = 1
        it = 0
        self.heatmaps.append(self.xT.copy())

        while np.any(diff > self.eps):
            total_payoff = np.zeros(self.n)

            for c in range(0, self.n):
                for q in range(0, self.n):
                    total_payoff[c] += transition_matrix[c, q] * self.xT[q]

            newxT = gs + (p_move * total_payoff)
            diff = newxT - self.xT
            self.xT = newxT
            self.heatmaps.append(self.xT.copy())
            it += 1

        if self.use_interpolation:
            self.xT = self.grid._interpolate(self.xT)

        print("# iterations: ", it)

    def fit(self, actions: pd.DataFrame):
        """ Fits the xT model with the given actions.

        :param actions: Actions, in SPADL format.
        """
        self.scoring_prob_matrix = scoring_prob(actions, self.grid)
        self.shot_prob_matrix, self.move_prob_matrix = action_prob(actions, self.grid)
        self.transition_matrix = move_transition_matrix(actions, self.grid)
        self.__solve(
            self.scoring_prob_matrix,
            self.shot_prob_matrix,
            self.move_prob_matrix,
            self.transition_matrix,
        )
        return self

    def get_oppc(self, x: pd.Series, y: pd.Series):
        opp_x = spadlconfig.field_length - x
        opp_y = spadlconfig.field_width - y
        return self.grid._get_flat_indexes(opp_x, opp_y, self.use_interpolation)

    def predict(
        self, actions: pd.DataFrame, xP = None
    ) -> np.ndarray:
        """ Predicts the xT values for the given actions.

        :param actions: Actions, in SPADL format.
        :param use_interpolation: Indicates whether to use bilinear interpolation when inferring xT values.
        :return: Each action, including its xT value.
        """

        mov_actions = get_move_actions(actions)
        succ_mov_actions = get_successful(mov_actions)
        fail_mov_actions = get_failed(mov_actions)
        def_actions = get_defensive_actions(actions)
        succ_def_actions = get_successful(def_actions)

        xT = pd.Series(np.zeros(actions.index.size), index=actions.index)
        xT.update(self.predict_successful_move_actions(succ_mov_actions))
        xT.update(self.predict_failed_move_actions(fail_mov_actions))
        xT.update(self.predict_successful_def_actions(succ_def_actions))

        return xT

    
    def predict_successful_move_actions(self, actions: pd.DataFrame, xP = None) -> np.ndarray:

        if xP is None:
            xP = np.ones(len(actions))

        startc = self.grid._get_flat_indexes(actions.start_x, actions.start_y, self.use_interpolation)
        endc = self.grid._get_flat_indexes(actions.end_x, actions.end_y, self.use_interpolation)  

        xT_start, xT_end = self.xT[startc], self.xT[endc]    

        return pd.Series(xP * (xT_end - xT_start).clip(0), index=actions.index)


    def predict_failed_move_actions(self, actions: pd.DataFrame, xP = None) -> np.ndarray:

        if xP is None:
            xP = np.ones(len(actions))

        startc = self.grid._get_flat_indexes(actions.start_x, actions.start_y, self.use_interpolation)
        opp_endc = self.get_oppc(actions.end_x, actions.end_y) 

        xT_start, xT_end = self.xT[startc], -self.xT[opp_endc]    

        return pd.Series(xP * (xT_end - 0), index=actions.index)


    def predict_successful_def_actions(self, actions: pd.DataFrame, xP = None) -> np.ndarray:

        if xP is None:
            xP = np.ones(len(actions))

        opp_startc = self.get_oppc(actions.start_x, actions.start_y) 
        endc = self.grid._get_flat_indexes(actions.end_x, actions.end_y, self.use_interpolation)

        xT_start, xT_end = -self.xT[opp_startc], self.xT[endc]

        return pd.Series(xP * (xT_end - xT_start), index=actions.index)

