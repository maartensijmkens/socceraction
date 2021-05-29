import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import warnings  # type: ignore

from typing import Tuple, List

import socceraction.spadl.config as spadlconfig
from socceraction.grid import Grid, DefaultGrid
from socceraction.timeframe import TimeFrame, DefaultTimeFrame
from socceraction.scorediff import ClippedScore
from socceraction.xpoints import ExpectedPoints

from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import interp2d


def _count(actions: pd.DataFrame, T: int, D: int, N: int) -> np.ndarray:
    """ Count the number of actions occurring in each cell of the grid.

    :param actions: preprocessed actions.
    :param T: number of timeframes the game is divided in
    :param N: number of cells the field is divided in.
    :return: A matrix, denoting the amount of actions occurring in each cell. The top-left corner is the origin.
    """
    counts = np.zeros((T, D, N))
    vc = actions.groupby(['timeframe', 'scorediff', 'start_cell']).size()
    i = tuple(zip(*vc.index))
    counts[i] = vc
    return counts


def safe_divide(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.divide(a, b, out=np.zeros_like(a), where=b != 0)


def get_shot_actions(actions: pd.DataFrame):
    return actions[(actions.type_name == "shot")]


def get_move_actions(actions: pd.DataFrame) -> pd.DataFrame:
    return actions[actions.type_name.isin(["pass", "cross", "dribble"])]


def get_defensive_actions(actions: pd.DataFrame) -> pd.DataFrame:
    return actions[actions.type_name.isin(["interception", "tackle"])]


def get_successful(actions: pd.DataFrame) -> pd.DataFrame:
    return actions[actions.result_name == "success"]


def get_failed(actions: pd.DataFrame) -> pd.DataFrame:
    return actions[actions.result_name == "fail"]


def scoring_prob(actions: pd.DataFrame, T: int, D: int, N: int) -> np.ndarray:
    """ Compute the probability of scoring when taking a shot for each cell.

    :param actions: preprocessed actions.
    :param T: number of timeframes the game is divided in
    :param N: number of cells the field is divided in.
    :return: A matrix, denoting the probability of scoring for each cell.
    """
    shots = get_shot_actions(actions)
    goals = get_successful(shots)

    shotmatrix = _count(shots, T, D, N)
    goalmatrix = _count(goals, T, D, N)

    return safe_divide(goalmatrix, shotmatrix)


def action_prob(actions: pd.DataFrame, T: int, D: int, N: int) -> Tuple[np.ndarray, np.ndarray]:
    """ Compute the probability of taking an action in each cell of the grid. The options are: shooting or moving.

    :param actions: preprocessed actions.
    :param T: number of timeframes the game is divided in
    :param N: number of cells the field is divided in.
    :return: 2 matrices, denoting for each cell the probability of choosing to shoot
    and the probability of choosing to move.
    """
    move_actions = get_move_actions(actions)
    shots = get_shot_actions(actions)

    movematrix = _count(move_actions, T, D, N)
    shotmatrix = _count(shots, T, D, N)
    totalmatrix = movematrix + shotmatrix

    return safe_divide(shotmatrix, totalmatrix), safe_divide(movematrix, totalmatrix)


def move_transition_matrix(actions: pd.DataFrame, T: int, D: int, N: int) -> Tuple[np.ndarray, np.ndarray]:
    """ Compute the move transition matrix from the given actions.

    This is, when a player chooses to move, the probability that he will
    end up in each of the other cells of the grid successfully.

    :param actions: preprocessed actions.
    :param T: number of timeframes the game is divided in
    :param grid: Grid object that divides the field in cells.
    :return: The transition matrix.
    """
    move_actions = get_move_actions(actions)

    # count the number of actions with the same timeframe and start_cell
    start_counts = np.zeros((T, D, N))
    vc = move_actions.groupby(['timeframe', 'scorediff', 'start_cell']).size()
    i = tuple(zip(*vc.index))
    start_counts[i] = vc

    # count the number of succesful actions with the same timeframe, start_cell and end_cell
    success_transition_counts = np.zeros((T, D, N, N))
    vc = get_successful(move_actions).groupby(['timeframe', 'scorediff', 'start_cell', 'end_cell']).size()
    i = tuple(zip(*vc.index))
    success_transition_counts[i] = vc

    # count the number of failed actions with the same timeframe, start_cell and end_opp_cell
    fail_transition_counts = np.zeros((T, D, N, N))
    vc = get_failed(move_actions).groupby(['timeframe', 'scorediff', 'start_cell', 'end_opp_cell']).size()
    i = tuple(zip(*vc.index))
    fail_transition_counts[i] = vc

    # calculate the probabilities
    return (
        safe_divide(success_transition_counts, start_counts[:,:,:,None]), 
        safe_divide(fail_transition_counts, start_counts[:,:,:,None])
    )


class ExpectedThreat:
    """An implementation of Karun Singh's Expected Threat model (https://karun.in/blog/expected-threat.html)."""

    def __init__(
            self, 
            grid: Grid = DefaultGrid(), 
            timeframe: TimeFrame = DefaultTimeFrame(), 
            scorediff: ClippedScore = ClippedScore(D = 0),
            use_interpolation: bool = False, 
            use_xRisk: bool = False,
            use_xP: bool = False, 
            eps: float = 1e-5
        ):
        self.timeframe = timeframe
        self.T = timeframe.get_length()
        self.scorediff = scorediff
        self.D = scorediff.get_length()
        self.grid = grid
        self.N = grid.get_length()
        self.use_interpolation = use_interpolation
        self.use_xRisk = use_xRisk
        self.use_xP = use_xP 
        self.eps = eps
        self.heatmaps: List[np.ndarray] = []
        self.xT: np.ndarray = np.zeros((self.T, self.D, self.N))
        self.scoring_prob_matrix: np.ndarray = np.zeros((self.T, self.D, self.N))
        self.shot_prob_matrix: np.ndarray = np.zeros((self.T, self.D, self.N))
        self.move_prob_matrix: np.ndarray = np.zeros((self.T, self.D, self.N))
        self.transition_matrix: np.ndarray = np.zeros((self.T, self.D, self.N, self.N))

    def __solve(
        self,
        p_scoring: np.ndarray,
        p_shot: np.ndarray,
        p_move: np.ndarray,
        success_transition_matrix: np.ndarray,
        fail_transition_matrix: np.ndarray,
    ) -> None:
        """Solves the expected threat equation with dynamic programming.

        :param p_scoring (matrix, shape(M, N)): Probability of scoring at each grid cell, when shooting from that cell.
        :param p_shot (matrix, shape(M, N)): For each grid cell, the probability of choosing to shoot from there.
        :param p_move (matrix, shape(M, N)): For each grid cell, the probability of choosing to move from there.
        :param transition_matrix (matrix, shape(M*N, M*N)): When moving, the probability of moving to each of the other zones.
        """
        gs = p_scoring * p_shot
        diff = 1
        it = 0
        self.heatmaps.append(self.xT.copy())

        while np.any(diff > self.eps):
            total_payoff = np.zeros((self.T, self.D, self.N))

            for t in range(self.T):
                for d in range(self.D):
                    for c in range(self.N):
                        for q in range(self.N):
                            total_payoff[t, d, c] += success_transition_matrix[t, d, c, q] * self.xT[t, d, q]
                            if self.use_xRisk:
                                total_payoff[t, d, c] -= fail_transition_matrix[t, d, c, q] * self.xT[t, -d, q]

            newxT = gs + (p_move * total_payoff)
            diff = newxT - self.xT
            self.xT = newxT
            self.heatmaps.append(self.xT.copy())
            it += 1

        print("# iterations: ", it)

    def preprocess(self, actions: pd.DataFrame):
        """ Extract relevant features (cell_index, timeframe, goal_difference, type, result) for every action.

        :param actions: Actions, in SPADL format.
        """

        df = pd.DataFrame(index = actions.index)
        df['start_x'] = actions['start_x']
        df['start_y'] = actions['start_y']
        df['end_x'] = actions['end_x']
        df['end_y'] = actions['end_y']
        df['start_opp_x'] = spadlconfig.field_length - actions['start_x']
        df['start_opp_y'] = spadlconfig.field_width - actions['start_y']
        df['end_opp_x'] = spadlconfig.field_length - actions['end_x']
        df['end_opp_y'] = spadlconfig.field_width - actions['end_y']  
        df['start_cell'] = self.grid.get_cell(df['start_x'], df['start_y'])
        df['end_cell'] = self.grid.get_cell(df['end_x'], df['end_y'])
        df['start_opp_cell'] = self.grid.get_cell(df['start_opp_x'], df['start_opp_y'])
        df['end_opp_cell'] = self.grid.get_cell(df['end_opp_x'], df['end_opp_y'])
        df['timeframe'] = self.timeframe.get_timeframe(actions)
        df['scorediff'] = self.scorediff.get_diff(actions['score'])
        df['type_name'] = actions['type_name']
        df['result_name'] = actions['result_name']

        return df[df['timeframe'] >= 0]


    def fit(self, actions: pd.DataFrame):
        """ Fits the xT model with the given actions.

        :param actions: Actions, in SPADL format.
        """
        actions = self.preprocess(actions)

        self.scoring_prob_matrix = scoring_prob(actions, self.T, self.D, self.N)
        self.shot_prob_matrix, self.move_prob_matrix = action_prob(actions, self.T, self.D, self.N)
        self.success_transition_matrix, self.fail_transition_matrix = move_transition_matrix(actions, self.T, self.D, self.N)
        self.__solve(
            self.scoring_prob_matrix,
            self.shot_prob_matrix,
            self.move_prob_matrix,
            self.success_transition_matrix,
            self.fail_transition_matrix
        )

        if self.use_interpolation:
            points = actions.groupby(['timeframe', 'scorediff', 'start_cell']).mean()
            i = tuple(zip(*points.index.to_list()))
            x = self.interpolator_features(points.reset_index(), ['start_x', 'start_y'])
            self.cell_centers = x
            z = self.xT[i]
            self.interpolator = LinearNDInterpolator(x, z)

        return self


    def predict(self, actions: pd.DataFrame) -> np.ndarray:
        """ Predicts the xT values for the given actions.

        :param actions: Actions, in SPADL format.
        :return: Each action's xT value.
        """        
        mov_actions = get_move_actions(actions)
        succ_mov_actions = get_successful(mov_actions)
        def_actions = get_defensive_actions(actions)
        succ_def_actions = get_successful(def_actions)

        empty = np.empty(actions.index.size)
        empty[:] = np.nan
        xT = pd.Series(empty, index=actions.index)
        xT.update(self.predict_successful_move_actions(succ_mov_actions))
        xT.update(self.predict_successful_def_actions(succ_def_actions))
        return xT

    
    def predict_successful_move_actions(self, actions: pd.DataFrame) -> np.ndarray:
        """ Predicts the xT values for the given actions.

        :param actions: Successful move actions, in SPADL format.
        :return: Each action, including its xT value.
        """
        actions = self.preprocess(actions)

        # if self.use_xP:
        #     t = actions.apply(self.expectedPoints.get_timeframe, axis=1)
        #     d = actions.apply(self.expectedPoints.get_score_diff, axis=1)
        #     xP_start = self.expectedPoints.predict(t, d)
        #     xP_end = self.expectedPoints.predict(t, d + 1)
        #     xP = xP_end - xP_start 

        if not self.use_interpolation:
            xT_start = self.xT[actions.timeframe, actions.scorediff, actions.start_cell]
            xT_end = self.xT[actions.timeframe, actions.scorediff, actions.end_cell]

        return pd.Series(xT_end - xT_start, index=actions.index)


    def predict_failed_move_actions(self, actions: pd.DataFrame) -> np.ndarray:
        """ Predicts the xT values for the given actions.

        :param actions: Failed move actions, in SPADL format.
        :return: Each action, including its xT value.
        """
        actions = self.preprocess(actions)

        if not self.use_interpolation:
            xT_start = self.xT[actions.timeframe, actions.scorediff, actions.start_cell]
            xT_end = -self.xT[actions.timeframe, actions.scorediff, actions.end_opp_cell] 

        return pd.Series(xT_end - xT_start, index=actions.index)


    def predict_successful_def_actions(self, actions: pd.DataFrame) -> np.ndarray:
        """ Predicts the xT values for the given actions.

        :param actions: Successful defensive actions, in SPADL format.
        :return: Each action, including its xT value.
        """
        actions = self.preprocess(actions)

        if not self.use_interpolation:
            xT_start = -self.xT[actions.timeframe, actions.scorediff, actions.start_opp_cell]
            xT_end = self.xT[actions.timeframe, actions.scorediff, actions.end_cell]

        return pd.Series(xT_end - xT_start, index=actions.index)

