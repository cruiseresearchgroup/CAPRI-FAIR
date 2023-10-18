import numpy as np
from tqdm import tqdm
from utils import logger
from Models.USG.lib.ItemExposurePowerLaw import ItemExposurePowerLaw
from Models.utils import loadModel, saveModel

modelName = 'USG'


def ItemExposureCalculations(datasetName: str, users: dict, pois: dict, poiCheckinCounts, groundTruth):
    # Initializing parameters
    userCount = users['count']
    IScores = np.zeros((users['count'], pois['count']))
    # Checking for existing model
    logger('Preparing Power Law matrix ...')
    loadedModel = loadModel(modelName, datasetName, f'I_{userCount}User')
    if loadedModel == []:  # It should be created
        I = ItemExposurePowerLaw()
        I.fitExposureDistribution(poiCheckinCounts)
        predicted = I.predict(pois['count']).T
        np.copyto(IScores, predicted)  # Should work because of broadcasting
        saveModel(IScores, modelName, datasetName, f'I_{userCount}User')
    else:  # It should be loaded
        IScores = loadedModel
    # Returning the scores
    return IScores
