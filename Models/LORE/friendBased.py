import numpy as np
from tqdm import tqdm
from utils import logger
from Models.utils import loadModel, saveModel
from Models.parallel_utils import run_parallel, CHUNK_SIZE
from Models.LORE.lib.FriendBasedCF import FriendBasedCF, friend_based_cf_predict

modelName = 'LORE'


def friendBasedCalculations(datasetName: str, users: dict, pois: dict, socialRelations, poiCoos, sparseTrainingMatrix, groundTruth):
    """
    This function calculates the friend-based features of the dataset.

    Parameters
    ----------
    datasetName : str
        The name of the dataset
    users : dict
        The users of the dataset
    pois : dict
        The pois of the dataset
    socialRelations : dict 
        The social relations of the dataset
    poiCoos : dict
        The poi coordinates of the dataset
    sparseTrainingMatrix : dict
        The sparse training matrix of the dataset
    groundTruth : dict
        The ground truth of the dataset

    Returns
    -------
    FCFScores : dict
        The FCF scores of the dataset
    """
    # Initializing parameters
    userCount = users['count']
    logDuration = 1 if userCount < 20 else 10
    FCFScores = np.zeros((users['count'], pois['count']))
    # Checking for existing model
    logger('Preparing Friend-based CF matrix ...')
    loadedModel = loadModel(modelName, datasetName, f'FCF_{userCount}User')
    if loadedModel == []:  # It should be created
        # Creating object to FCF Class
        FCF = FriendBasedCF()
        # Calculating FCF scores
        # TODO: We may be able to load the model from disk
        FCF.friendsSimilarityCalculation(
            socialRelations, poiCoos, sparseTrainingMatrix)

        print("Now, predicting the model for each user ...")
        uids = [uid for uid in users['list'] if uid in groundTruth]
        args = [(id(FCF), uid) for uid in uids]

        with np.errstate(under='ignore'):
            results = run_parallel(friend_based_cf_predict, args, CHUNK_SIZE)

        print("Writing the result...")
        for uid, lidScores in tqdm(zip(uids, results)):
            np.copyto(FCFScores[uid, :], lidScores)

        saveModel(FCFScores, modelName, datasetName,
                  f'FCF_{userCount}User')
    else:  # It should be loaded
        FCFScores = loadedModel
    # Returning the scores
    return FCFScores
