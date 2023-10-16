import os
import time
from collections import defaultdict, Counter
from itertools import islice

import pandas as pd
import numpy as np
import scipy.sparse as sparse

from utils import logger

CHUNK_SIZE = 32


def batched(iterable, n):
    "Batch data into tuples of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    while (batch := tuple(islice(iterable, n))):
        yield batch


def readSparseTrainingData(trainFile: str, numberOfUsers: int, numberOfPoI: int):
    """
    Reads the training data from the file and returns a sparse matrix.

    Parameters
    ----------
    trainFile : str
        The path to the training data file.
    numberOfUsers : int
        The number of users in the dataset.
    numberOfPoI : int
        The number of points of interest in the dataset.

    Returns
    -------
    sparseTrainingMatrix : sparse matrix
        The sparse matrix representation of the training data.
    """
    print('Reading sparse training data...')
    trainingData = open(trainFile, 'r').readlines()
    sparseTrainingMatrix = sparse.dok_matrix((numberOfUsers, numberOfPoI))
    trainingTuples = set()
    for dataInstance in trainingData:
        uid, lid, freq = dataInstance.strip().split()
        uid, lid, freq = int(uid), int(lid), int(freq)
        sparseTrainingMatrix[uid, lid] = freq
        trainingTuples.add((uid, lid))
    return sparseTrainingMatrix, trainingTuples


def readTrainingData(trainFile: str, numberOfUsers: int, numberOfPoI: int, withFrequency: bool = False):
    """
    Reads the training data from the file and returns 3 things: a check-in matrix
    and 2 dataframes denoting the user and item checkin counts. The 2 dataframes
    will be used in computing global fairness metrics.

    Parameters
    ----------
    trainFile : str
        The path to the training data file.
    numberOfUsers : int
        The number of users in the dataset.
    numberOfPoI : int
        The number of points of interest in the dataset.
    withFrequency : bool
        True for GeoSoCa, False for USG

    Returns
    -------
    trainingMatrix : ndarray
        The ndarray representation of the training data.
    userCheckinCounts : pd.DataFrame
        The dataframe of user checkin counts, as well as the ratio of repeat
        visits and whether or not they are distinguished as repeat or explore
        users.
    poiCheckinCounts : pd.DataFrame
        The dataframe of item/POI checkin counts, as well as the distinction
        between short-head (popular) and long-tail (unpopular) POIs.
    """
    print('Reading training data...')
    trainingData = open(trainFile, 'r').readlines()
    trainingMatrix = np.zeros((numberOfUsers, numberOfPoI))
    trainCheckinsByUser = defaultdict(list)
    trainCheckinsByPOI = defaultdict(list)

    # TODO: we may replace this condition with a more compact one
    # e.g. value = freq if withFrequency == True else 1.0
    for dataInstance in trainingData:
        uid, lid, freq = dataInstance.strip().split()
        uid, lid, freq = int(uid), int(lid), int(freq)
        trainCheckinsByUser[uid].extend([lid for _ in range(freq)])
        trainCheckinsByPOI[lid].extend([uid for _ in range(freq)])
        trainingMatrix[uid, lid] = (freq if withFrequency else 1)

    userCheckinCounts = [
        (user, len(pois), len(set(pois)))
        for user, pois in dict(trainCheckinsByUser).items()
    ]
    userCheckinCounts = pd.DataFrame(
        userCheckinCounts,
        columns=['user_id', 'checkins', 'unique_checkins']
    ).set_index('user_id')
    userCheckinCounts['repeat_ratio'] = (
        (userCheckinCounts['checkins'] - userCheckinCounts['unique_checkins'])
        / userCheckinCounts['checkins'])
    med = userCheckinCounts['repeat_ratio'].median()
    userCheckinCounts['repeat_user'] = (
        userCheckinCounts['repeat_ratio'] >= med)

    print(f"Setting threshold to median repeating POI ratio of {med:.1f}")

    poiCheckinCounts = [
        (poi, len(users))
        for poi, users in dict(trainCheckinsByPOI).items()
    ]
    poiCheckinCounts = pd.DataFrame(
        poiCheckinCounts, columns=['poi_id', 'checkins']
    ).set_index('poi_id').sort_values(by='checkins', ascending=False)
    med = poiCheckinCounts['checkins'].quantile([0.8]).values[0]
    poiCheckinCounts['short_head'] = (poiCheckinCounts['checkins'] >= med)

    print(f"Setting threshold to 80th prc. of POI visits of {med:.1f}")

    return trainingMatrix, userCheckinCounts, poiCheckinCounts


def readTrainingCheckins(checkinFile: str, sparseTrainingMatrix):
    """
    Reads the training checkins from the file and returns a dictionary.

    Parameters
    ----------
    checkinFile : str
        The path to the training checkins file.
    sparseTrainingMatrix : sparse matrix
        The sparse matrix representation of the training data.

    Returns
    -------
    trainingCheckins : dict
        The dictionary representation of the training checkins.
    """
    print('Reading training checkins...')
    checkinData = open(checkinFile, 'r').readlines()
    trainingCheckins = defaultdict(list)
    for dataInstance in checkinData:
        uid, lid, ctime = dataInstance.strip().split()
        uid, lid, ctime = int(uid), int(lid), float(ctime)
        if not sparseTrainingMatrix[uid, lid] == 0:
            trainingCheckins[uid].append([lid, ctime])
    return trainingCheckins


# appendType: 'list' for LORE, 2) 'dictionary' for USG, 3) 'ndarray' for GeoSoCa
# numberOfUsers is only needed for GeoSoCa, others should get None
def readFriendData(socialFile, appendType, numberOfUsers):
    print('Reading friendship checkins...')
    socialData = open(socialFile, 'r').readlines()
    # TODO: we may replace this condition with a more compact one
    if appendType == 'list':  # LORE
        socialRelations = []
        for dataInstance in socialData:
            uid1, uid2 = dataInstance.strip().split()
            uid1, uid2 = int(uid1), int(uid2)
            socialRelations.append([uid1, uid2])
        return socialRelations
    elif appendType == 'ndarray':  # GeoSoCa
        # GeoSoCa needs numberOfUsers
        socialRelations = np.zeros((numberOfUsers, numberOfUsers))
        for dataInstance in socialData:
            uid1, uid2 = dataInstance.strip().split()
            uid1, uid2 = int(uid1), int(uid2)
            socialRelations[uid1, uid2] = 1.0
            socialRelations[uid2, uid1] = 1.0
        return socialRelations
    else:  # USG
        socialRelations = defaultdict(list)
        for dataInstance in socialData:
            uid1, uid2 = dataInstance.strip().split()
            uid1, uid2 = int(uid1), int(uid2)
            socialRelations[uid1].append(uid2)
            socialRelations[uid2].append(uid1)
        return socialRelations


def readTestData(testFile: str):
    """
    Reads the test data from the file and returns a dictionary.

    Parameters
    ----------
    testFile : str
        The path to the test data file.

    Returns
    -------
    groundTruth : dict
        The dictionary representation of the test data.
    """
    print('Reading test data...')
    groundTruth = defaultdict(set)
    truthData = open(testFile, 'r').readlines()
    for dataInstance in truthData:
        uid, lid, _ = dataInstance.strip().split()
        uid, lid = int(uid), int(lid)
        groundTruth[uid].add(lid)
    return groundTruth


def readPoiCoos(poiFile: str):
    """
    Reads the POI coordinates from the file and returns a dictionary.

    Parameters
    ----------
    poiFile : str
        The path to the POI coordinates file.

    Returns
    -------
    poiCoos : dict
        The dictionary representation of the POI coordinates.
    """
    print('Reading PoI coordinates...')
    poiCoos = {}
    poiData = open(poiFile, 'r').readlines()
    for dataInstance in poiData:
        lid, lat, lng = dataInstance.strip().split()
        lid, lat, lng = int(lid), float(lat), float(lng)
        poiCoos[lid] = (lat, lng)
    return poiCoos


def readCategoryData(categoryFile, numberOfCategories, numberOfPoI):
    """
    Reads the category data from the file and returns a dictionary.
    """
    print('Reading Categories data...')
    categoryData = open(categoryFile, 'r').readlines()
    poiCategoryMatrix = np.zeros((numberOfPoI, numberOfCategories))
    for dataInstance in categoryData:
        lid, cid = dataInstance.strip().split()
        lid, cid = int(lid), int(cid)
        poiCategoryMatrix[lid, cid] = 1.0
    return poiCategoryMatrix


def normalize(scores):
    maxScore = max(scores)
    if not maxScore == 0:
        scores = [s / maxScore for s in scores]
    return scores


def loadModel(modelName: str, datasetName: str, moduleName: str):
    """
    Loads the model from the file.

    Parameters
    ----------
    modelName : str
        The name of the model.
    datasetName : str
        The name of the dataset.
    moduleName : str
        The name of the module.

    Returns
    -------
    model : object
        The model object.
    """
    fileName = f'{modelName}_{datasetName}_{moduleName}.npy'
    logger(f"Looking for {fileName} in previously saved models ...")
    path = os.path.abspath(f'./Models/{modelName}/savedModels/{fileName}')
    fileExists = os.path.exists(path)
    if fileExists == True:
        content = np.load(path)
        logger(f"Model {fileName} loaded from previously execution results!")
        return content
    else:
        logger(
            f"Model {fileName} doesn't exist! It should be created!", 'warn')
        return []


def saveModel(content, modelName: str, datasetName: str, moduleName: str):
    """
    Saves the model to the file.

    Parameters
    ----------
    content : any
        The content to be saved.
    modelName : str
        The name of the model.
    datasetName : str
        The name of the dataset.
    moduleName : str
        The name of the module.
    """
    startTime = time.time()
    fileName = f'{modelName}_{datasetName}_{moduleName}.npy'
    logger(f"Saving model {fileName} ...")
    path = os.path.abspath(f'./Models/{modelName}/savedModels/{fileName}')
    fileExists = os.path.exists(path)
    if fileExists == False:
        open(path, 'w+')
    np.save(path, content)
    elapsedTime = '{:.2f}'.format(time.time() - startTime)
    logger(f"Model saved in {path} (took {elapsedTime} seconds)")
