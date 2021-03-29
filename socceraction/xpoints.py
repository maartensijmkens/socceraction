import numpy as np
import pandas as pd
from tqdm import tqdm


class ExpectedPoints:

    def __init__(self, D: int = 2, T = 8):
        self.D = D
        self.xwin = np.full((T, D + 1), np.nan)
        self.xdraw = np.full((T, D + 1), np.nan)
 

    def fit(self, actions: pd.DataFrame, games: pd.DataFrame):

        self.score_progression = self.get_score_progression(actions, games)
        sp = self.score_progression

        win_matrix = np.zeros_like(self.xwin)
        draw_matrix = np.zeros_like(self.xdraw)
        count_matrix = np.zeros_like(self.xwin)

        for i, game in games.iterrows():

            game_id = game['game_id']
            home_score = game['home_score']
            away_score = game['away_score']

            diff = home_score - away_score

            for t in range(8):

                cur_score = sp[(sp.game_id == game_id) & (sp.apply(self.get_timeframe, axis=1) <= t)].tail(1)

                cur_home_score, cur_away_score = \
                    cur_score.iloc[0][['home_score','away_score']]

                cur_diff = int(cur_home_score - cur_away_score)
                d = np.clip(abs(cur_diff), 0, self.D)

                if diff * cur_diff > 0:
                    win_matrix[t, d] += 2

                if diff == 0:
                    draw_matrix[t, d] += 2

                if cur_diff == 0 and diff != 0:
                    win_matrix[t, d] += 1

                count_matrix[t, d] += 2

        np.divide(win_matrix, count_matrix, out = self.xwin, where = (count_matrix != 0))
        np.divide(draw_matrix, count_matrix, out = self.xdraw, where = (count_matrix != 0))

        self.fill_gaps(self.xwin)
        self.fill_gaps(self.xdraw)       


    def fill_gaps(self, matrix):
        # replace nans with closest value with the same score diff
        for t,row in enumerate(matrix):
            for s,cell in enumerate(row):
                if np.isnan(cell):
                    d = np.argwhere(np.logical_not(np.isnan(matrix[:,s]))).flatten()
                    f = np.argmin(d-t)
                    matrix[t,s] = matrix[d[f],s]


    def predict(self, t: int, diff: int):
        """ Predict the expected points (xP) for the given score difference and time """

        d = np.clip(abs(diff), 0, self.D)
        a = 3 * self.xwin[t,d] + self.xdraw[t,d]
        b = 3 * (1 - self.xwin[t,d] - self.xdraw[t,d]) + self.xdraw[t,d]

        return np.where(diff >= 0, a, b)

    
    def get_timeframe(self, action):

        period_id = action['period_id']
        time_seconds = action['time_seconds']

        if period_id <= 2:
            t = 3*(period_id-1) + np.clip(time_seconds // (15*60), 0, 2)
        else:
            t = 6 + (period_id-3)
        return int(t)


    def get_score_diff(self, action: pd.DataFrame):

        sp = self.score_progression

        game_id, team_id, period_id, time_seconds = \
            action[['game_id', 'team_id', 'period_id', 'time_seconds']]

        current_score = sp[
            (sp.game_id == game_id) & 
            (((sp.period_id == period_id) & (sp.time_seconds <= time_seconds))
            | (sp.period_id < period_id))
        ].tail(1)

        home_team_id, away_team_id, home_score, away_score = \
            current_score.iloc[0][['home_team_id','away_team_id','home_score','away_score']]

        assert(team_id in [home_team_id, away_team_id])

        diff = int(home_score - away_score)

        if home_team_id == team_id:
            return diff
        elif away_team_id == team_id:
            return -diff


    def get_score_progression(self, actions: pd.DataFrame, games: pd.DataFrame):

        score_progression = []
        cols = ['game_id', 'home_team_id', 'away_team_id', 'period_id', 'time_seconds', 'home_score', 'away_score']

        for game_id, game_actions in actions.groupby(['game_id']):

            game = games[games.game_id == game_id]
            home_team = game.iloc[0]['home_team_id']
            away_team = game.iloc[0]['away_team_id']
            home_score = game.iloc[0]['home_score']
            away_score = game.iloc[0]['away_score']

            shots = game_actions[
                (game_actions.type_name == "shot") 
                | (game_actions.type_name == "shot_penalty") 
                | (game_actions.type_name == "shot_freekick")
            ]
            goals = shots[(shots.result_name == "success") & (shots.period_id < 5)]
            owngoals = game_actions[game_actions.result_name == "owngoal"]
            allgoals = pd.concat([goals, owngoals]).sort_values(["period_id", "time_seconds"])

            cur_score = [0,0]
            teams = (home_team, away_team)

            score_progression.append([game_id, home_team, away_team, 1, 0] + cur_score)

            for _, goal in allgoals.iterrows():

                if (goal['result_name'] == "success"):
                    t = teams.index(goal['team_id'])
                if (goal['result_name'] == "owngoal"):
                    t = (teams.index(goal['team_id']) + 1) % 2
                cur_score[t] += 1
                score_progression.append([game_id, home_team, away_team, goal['period_id'], goal['time_seconds']] + cur_score)

            assert (cur_score[0] == home_score and cur_score[1] == away_score)

        return pd.DataFrame(score_progression, columns=cols).sort_values(["game_id", "period_id", "time_seconds"])