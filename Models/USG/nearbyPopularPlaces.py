import numpy as np
from tqdm import tqdm
from utils import logger
from Models.USG.lib.ItemExposurePowerLaw import ItemExposurePowerLaw
from Models.USG.lib.NearbyPopularPlaces import NearbyPopularPlaces, nearby_predict
from Models.utils import loadModel, saveModel
from Models.parallel_utils import run_parallel, CHUNK_SIZE

modelName = 'USG'


def NearbyPopularPlacesCalculations(datasetName: str, users: dict, pois: dict, trainingMatrix, poiCoos, poiCheckinCounts, activeUsers, groundTruth):
    # Initializing parameters
    userCount = users['count']
    NScores = np.zeros((users['count'], pois['count']))
    # Checking for existing model
    logger('Preparing Power Law matrix ...')
    loadedModel = loadModel(modelName, datasetName, f'N_{userCount}User')
    if loadedModel == []:  # It should be created
        N = NearbyPopularPlaces()
        N.calculatePopularities(trainingMatrix, poiCoos, poiCheckinCounts, activeUsers)

        print("Now, predicting the model for each user ...")
        uids = [uid for uid in users['list'] if uid in groundTruth]
        args = [(id(N), uid) for uid in uids]

        # with np.errstate(under='ignore'):
        results = run_parallel(nearby_predict, args, CHUNK_SIZE)

        print("Writing the result...")
        for uid, lidScores in tqdm(zip(uids, results)):
            np.copyto(NScores[uid, :], lidScores)

        saveModel(NScores, modelName, datasetName, f'N_{userCount}User')
    else:  # It should be loaded
        NScores = loadedModel
    # Returning the scores
    return NScores
