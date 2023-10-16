import math

import numpy as np
from tqdm import tqdm
from config import GeoSoCaDict
from Models.utils import loadModel, saveModel, CHUNK_SIZE
from Models.parallel_utils import run_parallel
from Models.GeoSoCa.lib.AdaptiveKernelDensityEstimation import (
    AdaptiveKernelDensityEstimation,
    adaptive_kde_predict
)

modelName = 'GeoSoCa'


def geographicalCalculations(datasetName: str, users: dict, pois: dict, poiCoos: dict, trainingMatrix, groundTruth):
    """
    This function calculates the geographical parameters of the dataset

    Parameters
    ----------
    datasetName : str
        The name of the dataset
    users : dict
        The users of the dataset
    pois : dict
        The pois of the dataset
    poiCoos : dict 
        The poi coordinates of the dataset
    groundTruth : dict
        The ground truth of the dataset
    trainingMatrix : dict
        The training matrix of the dataset
    groundTruth : dict
        The ground truth of the dataset

    Returns
    -------
    AKDEScores : dict
        The AKDE scores of the dataset
    """
    # Initializing parameters
    userCount = users['count']
    alpha = GeoSoCaDict['alpha']
    logDuration = 1 if userCount < 20 else 10
    AKDEScores = np.zeros((userCount, pois['count']))
    # Checking for existing model
    print("Preparing Adaptive Kernel Density Estimation matrix ...")
    loadedModel = loadModel(modelName, datasetName,
                            f'AKDE_{userCount}User')
    if loadedModel == []:  # It should be created
        # Creating object to AKDE Class
        AKDE = AdaptiveKernelDensityEstimation(alpha)
        # Calculating AKDE scores
        # TODO: We may be able to load the model from disk
        AKDE.precomputeKernelParameters(trainingMatrix, poiCoos)

        print("Now, predicting the model for each user ...")
        uids = (uid for uid in users['list'] if uid in groundTruth)
        args = [(id(AKDE), uid) for uid in uids]

        with np.errstate(under='ignore'):
            results = run_parallel(adaptive_kde_predict, args, CHUNK_SIZE)

        print("Writing the result...")
        for uid, lidScores in tqdm(zip(uids, results)):
            np.copyto(AKDEScores[uid, :], lidScores)

        saveModel(AKDEScores, modelName, datasetName,
                  f'AKDE_{userCount}User')
    else:  # It should be loaded
        AKDEScores = loadedModel
    # Returning the scores
    return AKDEScores
