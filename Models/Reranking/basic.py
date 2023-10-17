"""
Reranking module contains methods that take in a base prediction of
recommended POIs and re-orders them according to some method or criterion.

These methods are baseline, trivial methods.
"""

from random import shuffle

import numpy as np
import pandas as pd
from tqdm import tqdm
from Models.utils import normalize
from Models.parallel_utils import run_parallel, CHUNK_SIZE
from utils import logger, textToOperator
from config import USGDict, topK, listLimit, outputsDir


def topk_ranking(predictions: dict):
    """
    Simply perform no reordering; that is, pick the top K items in each list.
    """
    for uid in predictions.keys():
        predictions[uid] = predictions[uid][:topK]

    return predictions


def random_ranking(predictions: dict):
    """
    Randomly rank the preselected items
    """
    for uid in predictions.keys():
        shuffle(predictions[uid])
        predictions[uid] = predictions[uid][:topK]

    return predictions