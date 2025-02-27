import numpy as np
from utils import logger
from config import limitUsers, listLimit, itemExposureScalingFactor
from Evaluations.evaluator import evaluator
from Data.readDataSizes import readDataSizes
from Data.calculateActiveUsers import calculateActiveUsers
from Models.LORE.friendBased import friendBasedCalculations
from Models.LORE.additiveMarkovChain import additiveMarkovChainCalculations
from Models.LORE.kernelDensityEstimation import kernelDensityEstimationCalculations
from Models.USG.itemExposure import ItemExposureCalculations
from Models.USG.nearbyPopularPlaces import NearbyPopularPlacesCalculations
from Models.utils import readFriendData, readPoiCoos, readSparseTrainingData, readTestData, readTrainingCheckins, computeAverageLocation
from Models.Reranking import rerankPredictions
from Models.scoring import calculateScores

modelName = 'LORE'


class LOREMain:
    def main(datasetFiles, params):
        logger(f'Processing data using {modelName} model ...')

        # Reading data size from the selected dataset
        dataDictionary = readDataSizes(params['datasetName'], datasetFiles)
        users, pois = dataDictionary['users'], dataDictionary['pois']

        # Loading trainin items
        logger('Reading dataset instances ...')
        sparseTrainingMatrix, trainingMatrix, userCheckinCounts, poiCheckinCounts \
            = readSparseTrainingData(datasetFiles['train'], users['count'], pois['count'])
        trainingCheckins = readTrainingCheckins(
            datasetFiles['checkins'], sparseTrainingMatrix)
        sortedTrainingCheckins = {uid: sorted(trainingCheckins[uid], key=lambda k: k[1])
                                  for uid in trainingCheckins}
        socialRelations = readFriendData(
            datasetFiles['socialRelations'], 'list', None)
        groundTruth = readTestData(datasetFiles['test'])
        poiCoos = readPoiCoos(datasetFiles['poiCoos'])
        averageLocation = computeAverageLocation(
            datasetFiles['train'], users['count'], pois['count'], poiCoos)

        # Limit the number of users
        if (limitUsers != -1):
            logger(f'Limiting the number of users to {limitUsers} ...')
            users['count'] = limitUsers
            users['list'] = users['list'][:limitUsers]

        # Computing the final scores
        FCFScores = friendBasedCalculations(
            params['datasetName'], users, pois, socialRelations, poiCoos, sparseTrainingMatrix, groundTruth)
        KDEScores = kernelDensityEstimationCalculations(
            params['datasetName'], users, pois, poiCoos, sparseTrainingMatrix, groundTruth)
        AMCScores = additiveMarkovChainCalculations(
            params['datasetName'], users, pois, sortedTrainingCheckins, groundTruth)

        # Segmenting active users
        activeUsers = calculateActiveUsers(params['datasetName'], datasetFiles['train'])

        # Score calculation
        # (Moving this before evaluation so that we can test reranking methods)
        evalParams = {'usersList': users['list'], 'usersCount': users['count'],
                      'groundTruth': groundTruth, 'fusion': params['fusion'], 'poiList': pois['list'],
                      'trainingMatrix': trainingMatrix, 'evaluation': params['evaluation'],
                      'fusionWeights': params['fusionWeights'], 'poiCoos': poiCoos,
                      'fairness': params['fairness'], 'fairnessWeights': params['fairnessWeights'],
                      'topK': params['topK'], 'exposureModel': params['exposureModel']}
        modelParams = {'FCF': FCFScores, 'KDE': KDEScores, 'AMC': AMCScores}

        # Add fairness modules as needed
        if params['fairness'] in ('Provider', 'Both'):
            IScores = ItemExposureCalculations(
                params['datasetName'], users, pois, poiCheckinCounts, groundTruth,
                params['exposureModel'])
            modelParams['I'] = IScores
        if params['fairness'] in ('Consumer', 'Both'):
            NScores = NearbyPopularPlacesCalculations(
                params['datasetName'], users, pois, trainingMatrix, poiCoos, poiCheckinCounts, activeUsers, groundTruth)
            modelParams['N'] = NScores

        predictions, scores = calculateScores(
            modelName, evalParams, modelParams, listLimit)

        # Reranking
        predictions = rerankPredictions(
            params['reranker'],
            predictions,
            params['topK'],
            userCheckinCounts=userCheckinCounts,
            poiCheckinCounts=poiCheckinCounts,
            scalingFactor=itemExposureScalingFactor,
            predictionScores=scores
        )

        # Evaluation
        evaluator(
            modelName, params['reranker'], params['datasetName'], evalParams,
            modelParams, predictions, userCheckinCounts=userCheckinCounts,
            poiCheckinCounts=poiCheckinCounts, averageLocation=averageLocation,
            activeUsers=activeUsers
        )
