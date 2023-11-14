import ctypes
import numpy as np
import pandas as pd
from tqdm import tqdm
from Models.utils import normalize, normalize_np
from Models.parallel_utils import run_parallel, CHUNK_SIZE
from utils import logger, textToOperator
from config import USGDict, listLimit, outputsDir


# Parallel score calculators

def parallelScoreCalculatorUSG(userId, evalParamsId, modelParamsId, listLimit):
    # Extracting the list of parameters
    evalParams = ctypes.cast(evalParamsId, ctypes.py_object).value
    modelParams = ctypes.cast(modelParamsId, ctypes.py_object).value

    fusion, poiList, trainingMatrix, fusionWeights = evalParams['fusion'], evalParams['poiList'], evalParams['trainingMatrix'], evalParams['fusionWeights']
    alpha, beta = USGDict['alpha'], USGDict['beta']
    provider_coef = evalParams['fairnessWeights']['provider']
    consumer_coef = evalParams['fairnessWeights']['consumer']
    UScores, SScores, GScores = modelParams['U'], modelParams['S'], modelParams['G']

    UScoresNormal = normalize([UScores[userId, lid]
                               if trainingMatrix[userId, lid] == 0 else -1
                               for lid in poiList])
    SScoresNormal = normalize([SScores[userId, lid]
                               if trainingMatrix[userId, lid] == 0 else -1
                               for lid in poiList])
    GScoresNormal = normalize([GScores[userId, lid]
                               if trainingMatrix[userId, lid] == 0 else -1
                               for lid in poiList])

    UScoresNormal, SScoresNormal, GScoresNormal = (
        np.array(UScoresNormal), np.array(SScoresNormal), np.array(GScoresNormal)
    )

    operands = [
        (1.0 - alpha - beta) * UScoresNormal,
        alpha * SScoresNormal,
        beta * GScoresNormal
    ]
    operandWeights = fusionWeights[:3]

    if evalParams['fairness'] in ('Provider', 'Both'):
        IScores = modelParams['I']
        IScoresNormal = normalize([IScores[userId, lid]
                                   if trainingMatrix[userId, lid] == 0 else -1
                                   for lid in poiList])
        IScoresNormal = np.array(IScoresNormal)
        operands.append(provider_coef * IScoresNormal)
        operandWeights.append(provider_coef)

    if evalParams['fairness'] in ('Consumer', 'Both'):
        NScores = modelParams['N']
        NScoresNormal = normalize([NScores[userId, lid]
                                   if trainingMatrix[userId, lid] == 0 else -1
                                   for lid in poiList])
        NScoresNormal = np.array(NScoresNormal)
        operands.append(consumer_coef * NScoresNormal)
        operandWeights.append(consumer_coef)

    overallScores = np.array(textToOperator(fusion, operands, operandWeights))
    argSorted = overallScores.argsort()
    predicted = list(reversed(argSorted))[:listLimit]
    scores = list(reversed(overallScores[argSorted]))[:listLimit]
    return list(zip(predicted, scores))


def parallelScoreCalculatorGeoSoCa(userId, evalParamsId, modelParamsId, listLimit):
    # Extracting the list of parameters
    evalParams = ctypes.cast(evalParamsId, ctypes.py_object).value
    modelParams = ctypes.cast(modelParamsId, ctypes.py_object).value

    fusion, poiList, trainingMatrix, fusionWeights = evalParams['fusion'], evalParams['poiList'], evalParams['trainingMatrix'], evalParams['fusionWeights']
    provider_coef = evalParams['fairnessWeights']['provider']
    consumer_coef = evalParams['fairnessWeights']['consumer']
    AKDEScores, SCScores, CCScores = modelParams['AKDE'], modelParams['SC'], modelParams['CC']

    # Check if Category is skipped
    operands = [
        normalize_np(np.array([
            AKDEScores[userId, lid]
            if trainingMatrix[userId, lid] == 0 else -1
            for lid in poiList
        ])),
        normalize_np(np.array([
            SCScores[userId, lid]
            if trainingMatrix[userId, lid] == 0 else -1
            for lid in poiList
        ]))
    ]
    operandWeights = fusionWeights[:2]

    if not (CCScores is None):
        operands.append(
            normalize_np(np.array([
                CCScores[userId, lid]
                if trainingMatrix[userId, lid] == 0 else -1
                for lid in poiList
            ]))
        )
        operandWeights = fusionWeights[:3]

    for i in range(len(operands)):
        operands[i] = operands[i] / len(operands)

    if evalParams['fairness'] in ('Provider', 'Both'):
        IScores = modelParams['I']
        IScoresNormal = normalize([IScores[userId, lid]
                                   if trainingMatrix[userId, lid] == 0 else -1
                                   for lid in poiList])
        IScoresNormal = np.array(IScoresNormal)
        operands.append(provider_coef * IScoresNormal)
        operandWeights.append(provider_coef)

    if evalParams['fairness'] in ('Consumer', 'Both'):
        NScores = modelParams['N']
        NScoresNormal = normalize([NScores[userId, lid]
                                   if trainingMatrix[userId, lid] == 0 else -1
                                   for lid in poiList])
        NScoresNormal = np.array(NScoresNormal)
        operands.append(consumer_coef * NScoresNormal)
        operandWeights.append(consumer_coef)

    overallScores = np.array(textToOperator(fusion, operands, operandWeights))
    argSorted = overallScores.argsort()
    predicted = list(reversed(argSorted))[:listLimit]
    scores = list(reversed(overallScores[argSorted]))[:listLimit]
    return list(zip(predicted, scores))


def parallelScoreCalculatorLORE(userId, evalParamsId, modelParamsId, listLimit):
    # Extracting the list of parameters
    evalParams = ctypes.cast(evalParamsId, ctypes.py_object).value
    modelParams = ctypes.cast(modelParamsId, ctypes.py_object).value

    fusion, poiList, trainingMatrix, fusionWeights = evalParams['fusion'], evalParams['poiList'], evalParams['trainingMatrix'], evalParams['fusionWeights']
    provider_coef = evalParams['fairnessWeights']['provider']
    consumer_coef = evalParams['fairnessWeights']['consumer']
    KDEScores, FCFScores, AMCScores = modelParams['KDE'], modelParams['FCF'], modelParams['AMC']

    operands = [
        normalize_np(np.array([
            KDEScores[userId, lid]
            if (userId, lid) not in trainingMatrix else -1
            for lid in poiList
        ])) / 3,
        normalize_np(np.array([
            FCFScores[userId, lid]
            if (userId, lid) not in trainingMatrix else -1
            for lid in poiList
        ])) / 3,
        normalize_np(np.array([
            AMCScores[userId, lid]
            if (userId, lid) not in trainingMatrix else -1
            for lid in poiList
        ])) / 3
    ]
    operandWeights = fusionWeights[:3]

    if evalParams['fairness'] in ('Provider', 'Both'):
        IScores = modelParams['I']
        IScoresNormal = normalize([IScores[userId, lid]
                                   if (userId, lid) not in trainingMatrix else -1
                                   for lid in poiList])
        IScoresNormal = np.array(IScoresNormal)
        operands.append(provider_coef * IScoresNormal)
        operandWeights.append(provider_coef)

    if evalParams['fairness'] in ('Consumer', 'Both'):
        NScores = modelParams['N']
        NScoresNormal = normalize([NScores[userId, lid]
                                   if (userId, lid) not in trainingMatrix else -1
                                   for lid in poiList])
        NScoresNormal = np.array(NScoresNormal)
        operands.append(consumer_coef * NScoresNormal)
        operandWeights.append(consumer_coef)

    overallScores = np.array(textToOperator(fusion, operands, operandWeights))
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
