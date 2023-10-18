import ctypes
import numpy as np
import pandas as pd
from tqdm import tqdm
from Models.utils import normalize
from Models.parallel_utils import run_parallel, CHUNK_SIZE
from utils import logger, textToOperator
from config import USGDict, topK, listLimit, outputsDir


# Parallel score calculators

def parallelScoreCalculatorUSG(userId, evalParamsId, modelParamsId, listLimit):
    # Extracting the list of parameters
    evalParams = ctypes.cast(evalParamsId, ctypes.py_object).value
    modelParams = ctypes.cast(modelParamsId, ctypes.py_object).value

    fusion, poiList, trainingMatrix, fusionWeights = evalParams['fusion'], evalParams['poiList'], evalParams['trainingMatrix'], evalParams['fusionWeights']
    alpha, beta = USGDict['alpha'], USGDict['beta']
    UScores, SScores, GScores, IScores = modelParams['U'], modelParams['S'], modelParams['G'], modelParams['I']

    UScoresNormal = normalize([UScores[userId, lid]
                               if trainingMatrix[userId, lid] == 0 else -1
                               for lid in poiList])
    SScoresNormal = normalize([SScores[userId, lid]
                               if trainingMatrix[userId, lid] == 0 else -1
                               for lid in poiList])
    GScoresNormal = normalize([GScores[userId, lid]
                               if trainingMatrix[userId, lid] == 0 else -1
                               for lid in poiList])
    IScoresNormal = normalize([IScores[userId, lid]
                               if trainingMatrix[userId, lid] == 0 else -1
                               for lid in poiList])
    UScoresNormal, SScoresNormal, GScoresNormal, IScoresNormal = (
        np.array(UScoresNormal), np.array(SScoresNormal),
        np.array(GScoresNormal), np.array(IScoresNormal)
    )

    overallScores = np.array(
        textToOperator(
            fusion,
            [(1.0 - alpha - beta) * UScoresNormal,
             alpha * SScoresNormal,
             beta * GScoresNormal,
             0.5 * IScoresNormal],
            fusionWeights
        )
    )
    argSorted = overallScores.argsort()
    predicted = list(reversed(argSorted))[:listLimit]
    scores = list(reversed(overallScores[argSorted]))[:listLimit]
    return list(zip(predicted, scores))


def parallelScoreCalculatorGeoSoCa(userId, evalParamsId, modelParamsId, listLimit):
    # Extracting the list of parameters
    evalParams = ctypes.cast(evalParamsId, ctypes.py_object).value
    modelParams = ctypes.cast(modelParamsId, ctypes.py_object).value

    fusion, poiList, trainingMatrix, fusionWeights = evalParams['fusion'], evalParams['poiList'], evalParams['trainingMatrix'], evalParams['fusionWeights']
    AKDEScores, SCScores, CCScores = modelParams['AKDE'], modelParams['SC'], modelParams['CC']

    # Check if Category is skipped
    overallScores = np.array([
        textToOperator(
            fusion,
            [AKDEScores[userId, lid], SCScores[userId, lid], CCScores[userId, lid]]
                if not (CCScores is None)
                else [AKDEScores[userId, lid], SCScores[userId, lid]],
            fusionWeights
        )
        if trainingMatrix[userId, lid] == 0 else -1
        for lid in poiList
    ])
    argSorted = overallScores.argsort()
    predicted = list(reversed(argSorted))[:listLimit]
    scores = list(reversed(overallScores[argSorted]))[:listLimit]
    return list(zip(predicted, scores))


def parallelScoreCalculatorLORE(userId, evalParamsId, modelParamsId, listLimit):
    # Extracting the list of parameters
    evalParams = ctypes.cast(evalParamsId, ctypes.py_object).value
    modelParams = ctypes.cast(modelParamsId, ctypes.py_object).value

    fusion, poiList, trainingMatrix, fusionWeights = evalParams['fusion'], evalParams['poiList'], evalParams['trainingMatrix'], evalParams['fusionWeights']
    KDEScores, FCFScores, AMCScores = modelParams['KDE'], modelParams['FCF'], modelParams['AMC']

    overallScores = np.array([
        textToOperator(
            fusion,
            [KDEScores[userId, lid], FCFScores[userId, lid], AMCScores[userId, lid]],
            fusionWeights
        )
        if (userId, lid) not in trainingMatrix else -1
        for lid in poiList
    ])
    argSorted = overallScores.argsort()
    predicted = list(reversed(argSorted))[:listLimit]
    scores = list(reversed(overallScores[argSorted]))[:listLimit]
    return list(zip(predicted, scores))


PARALLEL_FUNC_MAP = {
    'USG': parallelScoreCalculatorUSG,
    'GeoSoCa': parallelScoreCalculatorGeoSoCa,
    'LORE': parallelScoreCalculatorLORE,
}


def calculateScores(modelName: str, evalParams: dict, modelParams: dict,
                    listLimit: int):
    """
    Calculate the predictions dictionary (parallel computation).
    """

    usersList, groundTruth = evalParams['usersList'], evalParams['groundTruth']
    usersInGroundTruth = [u for u in usersList if u in groundTruth]
    args = [(uid, id(evalParams), id(modelParams), listLimit) for uid in usersInGroundTruth]
    results = run_parallel(PARALLEL_FUNC_MAP[modelName], args, CHUNK_SIZE)
    predictions = {
        uid: [p[0] for p in preds]
        for uid, preds in zip(usersInGroundTruth, results)
    }
    scores = {
        uid: [s[1] for s in scores]
        for uid, scores in zip(usersInGroundTruth, results)
    }

    return predictions, scores
