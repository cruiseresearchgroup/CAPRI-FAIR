import numpy as np
from tqdm import tqdm
from utils import logger
from Models.utils import loadModel, saveModel
from Models.parallel_utils import run_parallel, CHUNK_SIZE
from Models.USG.lib.PowerLaw import PowerLaw, power_law_predict

modelName = 'USG'


def powerLawCalculations(datasetName: str, users: dict, pois: dict, trainingMatrix, poiCoos, groundTruth):
    # Initializing parameters
    userCount = users['count']
    GScores = np.zeros((users['count'], pois['count']))
    # Checking for existing model
    logger('Preparing Power Law matrix ...')
    loadedModel = loadModel(modelName, datasetName, f'G_{userCount}User')
    if loadedModel == []:  # It should be created
        # Creating object to G Class
        G = PowerLaw()
        # Calculating G scores
        # TODO: We may be able to load the model from disk
        G.fitDistanceDistribution(trainingMatrix, poiCoos)

        print("Now, predicting the model for each user ...")
        uids = [uid for uid in users['list'] if uid in groundTruth]
        args = [(id(G), uid) for uid in uids]

        # with np.errstate(under='ignore'):
        results = run_parallel(power_law_predict, args, CHUNK_SIZE)

        print("Writing the result...")
        for uid, lidScores in tqdm(zip(uids, results)):
            np.copyto(GScores[uid, :], lidScores)

        saveModel(GScores, modelName, datasetName, f'G_{userCount}User')
    else:  # It should be loaded
        GScores = loadedModel
    # Returning the scores
    return GScores
