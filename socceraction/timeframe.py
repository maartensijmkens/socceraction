import numpy as np
import pandas as pd


class TimeFrame:
    "Interface defining the expected methods for a custom timeframe division"

    def get_timeframe(self, actions: pd.DataFrame):
        raise Exception("Not implemented")

    def get_length(self):
        raise Exception("Not implemented")
    

class DefaultTimeFrame(TimeFrame):
    "Default timeframe: considers the whole game as one"

    def get_timeframe(self, actions: pd.DataFrame):
        return np.zeros(len(actions.index), dtype=int)

    def get_length(self):
        return 1


class QuarterTimeFrame(TimeFrame):
    """Divides the game into timeframes of 15 minutes"""

    def _get_timeframe(self, action):

        period_id = action['period_id']
        time_seconds = action['time_seconds']

        if period_id <= 2:
            t = 3*(period_id-1) + np.clip(time_seconds // (15*60), 0, 2)
        else:
            t = 6 + (period_id-3)
        return int(t)

    def get_timeframe(self, actions: pd.DataFrame):
        return actions.apply(self._get_timeframe, axis=1) 

    def get_length(self):
        return 8