import pandas as pd
import numpy as np
from tqdm import tqdm


class ScoreProgression:

    def _get_difference(self, action: pd.DataFrame):

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

    def get_difference(self, actions: pd.DataFrame):
        tqdm.pandas()
        return actions.progress_apply(self._get_difference, axis=1) 

    def fit(self, actions: pd.DataFrame, games: pd.DataFrame):

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

        self.score_progression = \
            pd.DataFrame(score_progression, columns=cols).sort_values(["game_id", "period_id", "time_seconds"])

        return self


class ClippedScore: 

    def __init__(self, D: int, sp: ScoreProgression = None):
        self.D = D
        self.sp = sp

    def get_length(self):
        return 2 * self.D + 1

    def get_diff(self, scores: pd.Series):
        return np.clip(scores, -self.D, self.D) + self.D
