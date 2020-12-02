import numpy as np
import pandas as pd
from tqdm import tqdm


def get_diff(movactions: pd.DataFrame, actions: pd.DataFrame):

    indices = []
    scores = []

    for game_id, game_actions in tqdm(movactions.groupby('game_id')):
        
        game_goals = actions[
            (actions.game_id == game_id) &
            ((actions.type_name == "shot") | 
            (actions.type_name == "shot_penalty") | 
            (actions.type_name == "shot_freekick")) &
            (actions.result_name == "success") & 
            (actions.period_id < 5)
        ]

        game_owngoals = actions[
            (actions.game_id == game_id) &
            (actions.result_name == "owngoal")
        ]

        for i, action in game_actions.iterrows():

            goals = game_goals[
                (game_goals.period_id < action['period_id']) |
                ((game_goals.period_id == action['period_id']) &
                (game_goals.time_seconds < action['time_seconds']))
            ]

            owngoals = game_owngoals[
                (game_owngoals.period_id < action['period_id']) |
                ((game_owngoals.period_id == action['period_id']) &
                (game_owngoals.time_seconds < action['time_seconds']))
            ]            

            score = len(goals[goals.team_id == action['team_id']]) - len(goals[goals.team_id != action['team_id']]) - len(owngoals[owngoals.team_id == action['team_id']]) + len(owngoals[owngoals.team_id != action['team_id']])
            indices.append(i)
            scores.append(score)
    
    return np.array(scores)



def get_timeframe(action):

    period_id = action['period_id']
    time_seconds = action['time_seconds']

    if period_id <= 2:
        t = 3*(period_id-1) + np.clip(time_seconds // (15*60), 0, 2)
    else:
        t = 6 + (period_id-3)
    return int(t)


def get_points(home_score, away_score):
    if home_score > away_score:
        return (3, 0)
    if home_score == away_score:
        return (1, 1)
    else:
        return (0, 3)


class ExpectedPoints:

    def __init__(self, d: int = 2):
        self.d = d

    def fit(self, actions: pd.DataFrame, games: pd.DataFrame):

        shots = actions[(actions.type_name == "shot") | (actions.type_name == "shot_penalty") | (actions.type_name == "shot_freekick")]
        goals = shots[(shots.result_name == "success") & (shots.period_id < 5)]
        owngoals = actions[actions.result_name == "owngoal"]

        points_matrix = np.zeros((8, 2*self.d+1))
        count_matrix = np.zeros((8, 2*self.d+1))

        for _, game in games.iterrows():

            game_id = game['game_id']
            home_team_id = game['home_team_id']
            away_team_id = game['away_team_id']
            home_score = game['home_score']
            away_score = game['away_score']

            home_points, away_points = get_points(home_score, away_score)

            game_goals = goals[goals.game_id == game_id]
            game_owngoals = owngoals[owngoals.game_id == game_id]

            home_goals = game_goals[game_goals.team_id == home_team_id]
            away_owngoals = game_owngoals[game_owngoals.team_id == away_team_id]
            away_goals = game_goals[game_goals.team_id == away_team_id]
            home_owngoals = game_owngoals[game_owngoals.team_id == home_team_id]

            for t in range(8):
                
                hg = len(home_goals[(home_goals.apply(get_timeframe, axis=1) <= t)])
                ao = len(away_owngoals[(away_owngoals.apply(get_timeframe, axis=1) <= t)])
                ag = len(away_goals[(away_goals.apply(get_timeframe, axis=1) <= t)])
                ho = len(home_owngoals[(home_owngoals.apply(get_timeframe, axis=1) <= t)])  

                home_diff = np.clip(hg + ao - ag - ho, -self.d, self.d)
                away_diff = -home_diff

                points_matrix[t, home_diff+self.d] += home_points
                points_matrix[t, away_diff+self.d] += away_score
                count_matrix[t, home_diff+self.d] += 1
                count_matrix[t, away_diff+self.d] += 1

        out = np.empty_like(points_matrix)
        out[:] = np.nan        
        self.xpoints = np.divide(points_matrix, count_matrix, out = out, where = (count_matrix != 0))


        # replace nans with closest value with the same score diff
        for t,row in enumerate(self.xpoints):
            for s,cell in enumerate(row):
                if np.isnan(cell):
                    d = np.argwhere(np.logical_not(np.isnan(self.xpoints[:,s]))).flatten()
                    f = np.argmin(d-t)
                    self.xpoints[t,s] = self.xpoints[d[f],s]


    def predict(self, actions, diff, new_diff):
        
        t = actions.apply(get_timeframe, axis=1).to_numpy()
        diff = np.clip(diff, -self.d, self.d) + self.d
        new_diff = np.clip(new_diff, -self.d, self.d) + self.d

        return self.xpoints[t,new_diff] - self.xpoints[t,diff]
