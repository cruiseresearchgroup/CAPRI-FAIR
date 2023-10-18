"""
Reranking module contains methods that take in a base prediction of
recommended POIs and re-orders them according to some method or criterion.

These methods incorporate fairness into the rankings, whether that is user or
item fairness.
"""

from random import shuffle
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm
from Models.utils import normalize
from utils import logger, textToOperator
from config import USGDict, topK, listLimit, outputsDir


def item_exposure_ranking(
        predictions: dict,
        k: int,
        userCheckinCounts: pd.DataFrame,
        poiCheckinCounts: pd.DataFrame,
        scalingFactor: int,
        predictionScores: dict):
    """
    Try and prioritize low-exposure POIs up the recommendation list.
    We can find a scaling factor for POIs based on 
    """
    # compute item exposure in the results
    poi_exposures = Counter([p for uid, lids in predictions.items() for p in lids])
    poi_exposures = pd.DataFrame(poi_exposures.items(), columns=['poi_id', 'exposure']).set_index('poi_id').sort_values(by='exposure', ascending=False)
    poi_exposures['exposure_stzd'] = ((poi_exposures['exposure'] - poi_exposures['exposure'].mean()) / poi_exposures['exposure'].std()) * -1
    poi_exposures['exposure_stzd'] = np.exp(poi_exposures['exposure_stzd'] / scalingFactor)
    exposure_scaling = {pid: c[0] for pid, c in poi_exposures[['exposure_stzd']].iterrows()}

    # get set of explore users
    explore_uids = userCheckinCounts[~userCheckinCounts['repeat_user']].index
    for uid in explore_uids:
        _results = [(p, o * exposure_scaling[p]) for p, o in zip(predictions[uid], predictionScores[uid])]
        _results = sorted(_results, key=lambda x: x[1])
        predictions[uid] = [x[0] for x in _results][:k]

    return predictions