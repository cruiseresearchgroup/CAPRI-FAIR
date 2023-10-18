import numpy as np
from tqdm import tqdm
from utils import logger
from Models.utils import loadModel, saveModel
from Models.parallel_utils import run_parallel, CHUNK_SIZE
from Models.LORE.lib.KernelDensityEstimation import KernelDensityEstimation, kde_predict


modelName = 'LORE'


def kernelDensityEstimationCalculations(datasetName: str, users: dict, pois: dict, poiCoos, sparseTrainingMatrix, groundTruth):
    """
    This function calculates the kernel density estimation features of the dataset.

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
    sparseTrainingMatrix : dict
        The sparse training matrix of the dataset
    groundTruth : dict
        The ground truth of the dataset

    Returns
    -------
    KDEScores : dict
        The KDE scores of the dataset
    """
    # Initializing parameters
    userCount = users['count']
    KDEScores = np.zeros((users['count'], pois['count']))
    # Checking for existing model
    logger('Preparing Kernel Density Estimation matrix ...')
    loadedModel = loadModel(modelName, datasetName, f'KDE_{userCount}User')
    if loadedModel == []:  # It should be created
        # Creating object to KDE Class
        KDE = KernelDensityEstimation()
        # Calculating KDE scores
        # TODO: We may be able to load the model from disk
        KDE.precomputeKernelParameters(sparseTrainingMatrix, poiCoos)

        print("Now, predicting the model for each user ...")
        uids = [uid for uid in users['list'] if uid in groundTruth]
        args = [(id(KDE), uid) for uid in uids]

        with np.errstate(under='ignore'):
            results = run_parallel(kde_predict, args, CHUNK_SIZE)

        print("Writing the result...")
        for uid, lidScores in tqdm(zip(uids, results)):
            np.copyto(KDEScores[uid, :], lidScores)

        saveModel(KDEScores, modelName, datasetName,
                  f'KDE_{userCount}User')
    else:  # It should be loaded
        KDEScores = loadedModel
    # Returning the scores
    return KDEScores
