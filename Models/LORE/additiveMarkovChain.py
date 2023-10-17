import numpy as np
from tqdm import tqdm
from utils import logger
from config import LoreDict
from Models.utils import loadModel, saveModel
from Models.parallel_utils import run_parallel, CHUNK_SIZE
from Models.LORE.lib.AdditiveMarkovChain import (
    AdditiveMarkovChain,
    additivemarkovchain_predict
)

modelName = 'LORE'


def additiveMarkovChainCalculations(datasetName: str, users: dict, pois: dict, sortedTrainingCheckins, groundTruth):
    """
    This function calculates the additive markov chain features of the dataset.

    Parameters
    ----------
    datasetName : str
        The name of the dataset
    users : dict
        The users of the dataset
    pois : dict
        The pois of the dataset
    sortedTrainingCheckins : dict
        The sorted training checkins of the dataset
    groundTruth : dict
        The ground truth of the dataset

    Returns
    -------
    KDEScores : dict
        The KDE scores of the dataset
    """
    # Initializing parameters
    userCount = users['count']
    alpha, deltaT = LoreDict['alpha'], LoreDict['deltaT']
    AMCScores = np.zeros((users['count'], pois['count']))
    # Checking for existing model
    logger('Preparing Additive Markov Chain matrix ...')
    loadedModel = loadModel(modelName, datasetName, f'AMC_{userCount}User')
    if loadedModel == []:  # It should be created
        # Creating object to AMC Class
        AMC = AdditiveMarkovChain(deltaT, alpha)
        # Calculating AMC scores
        # TODO: We may be able to load the model from disk
        AMC.buildLocationToLocationTransitionGraph(sortedTrainingCheckins)

        print("Now, predicting the model for each user ...")
        uids = (uid for uid in users['list'] if uid in groundTruth)
        args = [(id(AMC), uid, pois['count']) for uid in uids]

        with np.errstate(under='ignore'):
            results = run_parallel(additivemarkovchain_predict, args, CHUNK_SIZE)

        print("Writing the result...")
        for uid, lidScores in tqdm(zip(uids, results)):
            np.copyto(AMCScores[uid, :], lidScores)

        saveModel(AMCScores, modelName, datasetName,
                  f'AMC_{userCount}User')
    else:  # It should be loaded
        AMCScores = loadedModel
    # Returning the scores
    return AMCScores
