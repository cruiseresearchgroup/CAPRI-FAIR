import ctypes
import numpy as np
import pandas as pd
from tqdm import tqdm
from Models.utils import normalize
from Models.parallel_utils import run_parallel, CHUNK_SIZE
from utils import logger, textToOperator
from config import USGDict, listLimit, outputsDir
from Evaluations.metrics.accuracy import precisionk, recallk, ndcgk, mapk, hitRatio
from Evaluations.metrics.fairness import (
    gceGlobalUserFairness,
    gceGlobalItemFairness,
    accuracyMetricByUserGroup,
    exposureMetricByItemGroup
)
from Evaluations.metrics.spatiotemporal import medianDistance


def overallScoreCalculator(modelName: str, userId, evalParams, modelParams):
    """
    Calculate the overall score of the model based on the given parameters

    Parameters
    ----------
    modelName : str
        Name of the model to be evaluated
    userId : int
        User ID
    evalParams : dict
        Dictionary of evaluation parameters
    modelParams : dict
        Dictionary of model parameters

    Returns
    -------
    overallScores : numpy.ndarray
        Array of overall scores
    """
    # Extracting the list of parameters
    fusion, poiList, trainingMatrix = evalParams['fusion'], evalParams['poiList'], evalParams['trainingMatrix']
    # Checking for proper model
    if (modelName == 'GeoSoCa'):
        AKDEScores, SCScores, CCScores = modelParams['AKDE'], modelParams['SC'], modelParams['CC']
        # Check if Category is skipped
        overallScores = [
            textToOperator(
                fusion,
                [AKDEScores[userId, lid], SCScores[userId, lid], CCScores[userId, lid]]
                    if not (CCScores is None)
                    else [AKDEScores[userId, lid], SCScores[userId, lid]]
            )
            if trainingMatrix[userId, lid] == 0 else -1
            for lid in poiList
        ]
    elif (modelName == 'LORE'):
        KDEScores, FCFScores, AMCScores = modelParams['KDE'], modelParams['FCF'], modelParams['AMC']
        overallScores = [textToOperator(fusion, [KDEScores[userId, lid], FCFScores[userId, lid], AMCScores[userId, lid]])
                         if (userId, lid) not in trainingMatrix else -1
                         for lid in poiList]
    elif (modelName == 'USG'):
        alpha, beta = USGDict['alpha'], USGDict['beta']
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
        UScoresNormal, SScoresNormal, GScoresNormal = np.array(
            UScoresNormal), np.array(SScoresNormal), np.array(GScoresNormal)
        overallScores = textToOperator(
            fusion, [(1.0 - alpha - beta) * UScoresNormal, alpha * SScoresNormal, beta * GScoresNormal])
    return np.array(overallScores)


def evaluator(modelName: str, rerankerName: str, datasetName: str,
              evalParams: dict, modelParams: dict, predictions: dict,
              userCheckinCounts = None, poiCheckinCounts = None,
              averageLocation = None, activeUsers = None):
    """
    Evaluate the model with the given parameters and return the evaluation metrics

    Parameters
    ----------
    modelName : str
        Name of the model to be evaluated
    rerankerName : str
        Name of the reranking method employed
    datasetName : str
        Name of the dataset to be evaluated
    evalParams : dict
        Dictionary of evaluation parameters
    modelParams : dict
        Dictionary of model parameters
    userCheckinCounts : pd.DataFrame
    ['user_id', 'checkins', 'unique_checkins', 'repeat_ratio', 'repeat_user']
        Dataframe of users, the number of checkins and unique POIs visited, the
        ratio of repeat visits, and whether or not they were designated a repeat
        or explore user.
    poiCheckinCounts : pd.DataFrame
    ['poi_id', 'checkins', 'short_head']
        Dataframe of POIs, the number of checkins, and whether or not they were
        designated a short-head or long-tail POI.
    """
    logger('Evaluating results ...')

    # Fetching the list of parameters
    usersList, usersCount, groundTruth, fusion, evaluationList = evalParams['usersList'], evalParams['usersCount'], evalParams[
        'groundTruth'], evalParams['fusion'], evalParams['evaluation']
    fairness = evalParams['fairness']
    poiCoos = evalParams['poiCoos']
    topK = evalParams['topK']
    evaluationList = [x['name'] for x in evaluationList]
    usersInGroundTruth = list((u for u in usersList if u in groundTruth))
    precision, recall, mean_ap, ndcg = [], [], [], []
    med_dist = []

    # Add caching policy (prevent a similar setting to be executed again)
    fairnessName = fairness
    if fairness in ('Provider', 'Both'):
        fairnessName += '_alpha' + str(evalParams['fairnessWeights']['provider'])
    if fairness in ('Consumer', 'Both'):
        fairnessName += '_beta' + str(evalParams['fairnessWeights']['consumer'])
    fileName = f'{modelName}_{rerankerName}_{fairnessName}_{datasetName}_{fusion}_{usersCount}user_top{topK}_limit{listLimit}'
    calculatedResults = open(f"{outputsDir}/Rec_{fileName}.txt", 'w+')

    # Initializing evaluation dataframe
    evalDataFrame = []
    print(f"Evaluation List: {evaluationList}")

    # Iterating over the users
    for counter, userId in tqdm(enumerate(usersInGroundTruth)):
        predicted = predictions[userId]
        actual = groundTruth[userId]
        if ('Precision' in evaluationList):
            precision.append(precisionk(actual, predicted[:topK]))
        if ('Recall' in evaluationList):
            recall.append(recallk(actual, predicted[:topK]))
        if ('NDCG' in evaluationList):
            ndcg.append(ndcgk(actual, predicted[:topK]))
        if ('mAP' in evaluationList):
            mean_ap.append(mapk(actual, predicted[:topK]))
        if not (averageLocation is None):
            med_dist.append(medianDistance(
                averageLocation[userId], predicted, poiCoos))
        # Writing the results to file
        calculatedResults.write('\t'.join([
            str(counter),
            str(userId),
            ','.join([str(lid) for lid in predicted])
        ]) + '\n')

    # Adding results to list
    print(f"Precisions: {precision[:20]}")
    print(f"NDCG: {ndcg[:20]}")

    metricsSet = {'precision': np.mean(precision), 'recall': np.mean(recall),
         'ndcg': np.mean(ndcg), 'map': np.mean(mean_ap),
         'mean_median_distance': np.mean(med_dist)}

    if not (activeUsers is None):
        metricsSet['gce_users'] = \
            gceGlobalUserFairness(groundTruth, predictions, activeUsers)

        if ('Precision' in evaluationList):
            precisionByGroup = accuracyMetricByUserGroup(precision, usersInGroundTruth, activeUsers)
            metricsSet['precision_active_users'] = precisionByGroup['active']
            metricsSet['precision_inactive_users'] = precisionByGroup['inactive']

    if not (poiCheckinCounts is None):
        metricsSet['gce_items'] = \
            gceGlobalItemFairness(groundTruth, predictions, topK, poiCheckinCounts)

        exposureByGroup = exposureMetricByItemGroup(groundTruth, predictions, topK, poiCheckinCounts)
        metricsSet['exposure_short_head'] = exposureByGroup['short_head']
        metricsSet['exposure_long_tail'] = exposureByGroup['long_tail']

    # Consolidate all metrics
    evalDataFrame.append(metricsSet)
    # Saving the results to file
    evalDataFrame = pd.DataFrame(evalDataFrame)
    print(evalDataFrame)
    # Saving evaluation results
    evalDataFrame.round(5).to_csv(
        f"{outputsDir}/Eval_{fileName}.csv", index=False)
    # Closing the file
    calculatedResults.close()
    # Logging the results
    logger(f'Evaluation results saved to {outputsDir}')
