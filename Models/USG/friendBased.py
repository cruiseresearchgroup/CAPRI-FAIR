import numpy as np
from tqdm import tqdm
from utils import logger
from config import USGDict
from Models.utils import loadModel, saveModel
from Models.parallel_utils import run_parallel, CHUNK_SIZE
from Models.USG.lib.FriendBasedCF import FriendBasedCF, friend_based_cf_predict

modelName = 'USG'


def friendBasedCalculations(datasetName: str, users: dict, pois: dict, socialRelations, trainingMatrix, groundTruth):
    # Initializing parameters
    userCount = users['count']
    eta = USGDict['eta']
    SScores = np.zeros((users['count'], pois['count']))
    # Checking for existing model
    logger('Preparing Friend-based CF matrix ...')
    loadedModel = loadModel(modelName, datasetName, f'S_{userCount}User')
    if loadedModel == []:  # It should be created
        # Creating object to S Class
        S = FriendBasedCF(eta)
        # Calculating S scores
        # TODO: We may be able to load the model from disk
        S.friendsSimilarityCalculation(socialRelations, trainingMatrix)

        print("Now, predicting the model for each user ...")
        uids = (uid for uid in users['list'] if uid in groundTruth)
        args = [(id(S), uid) for uid in uids]

        with np.errstate(under='ignore'):
            results = run_parallel(friend_based_cf_predict, args, CHUNK_SIZE)

        print("Writing the result...")
        for uid, lidScores in tqdm(zip(uids, results)):
            np.copyto(SScores[uid, :], lidScores)

        saveModel(SScores, modelName, datasetName, f'S_{userCount}User')
    else:  # It should be loaded
        SScores = loadedModel
    # Returning the scores
    return SScores
