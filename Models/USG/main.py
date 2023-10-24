import numpy as np
from utils import logger
from config import limitUsers, topK, listLimit, itemExposureScalingFactor
from Evaluations.evaluator import evaluator
from Data.readDataSizes import readDataSizes
from Models.USG.powerLaw import powerLawCalculations
from Models.USG.itemExposure import ItemExposureCalculations
from Models.USG.userBased import userBasedCalculations
from Models.USG.friendBased import friendBasedCalculations
from Data.calculateActiveUsers import calculateActiveUsers
from Models.Reranking import rerankPredictions
from Models.utils import readTrainingData, readFriendData, readTestData, readPoiCoos, computeAverageLocation
from Models.scoring import calculateScores

modelName = 'USG'


class USGMain:
    def main(datasetFiles, params):
        logger(f'Processing data using {modelName} model ...')

        # Reading data size from the selected dataset
        dataDictionary = readDataSizes(params['datasetName'], datasetFiles)
        users, pois = dataDictionary['users'], dataDictionary['pois']

        # Loading training items
        logger('Reading dataset instances ...')
        trainingMatrix, userCheckinCounts, poiCheckinCounts = readTrainingData(
            datasetFiles['train'], users['count'], pois['count'], False)
        groundTruth = readTestData(datasetFiles['test'])
        socialRelations = readFriendData(
            datasetFiles['socialRelations'], 'dictionary', None)
        poiCoos = readPoiCoos(datasetFiles['poiCoos'])
        averageLocation = computeAverageLocation(
            datasetFiles['train'], users['count'], pois['count'], poiCoos)

        # Limit the number of users
        if (limitUsers != -1):
            logger(f'Limiting the number of users to {limitUsers} ...')
            users['count'] = limitUsers
            users['list'] = users['list'][:limitUsers]

        # Computing the final scores
        UScores = userBasedCalculations(
            params['datasetName'], users, pois, trainingMatrix, groundTruth)
        SScores = friendBasedCalculations(
            params['datasetName'], users, pois, socialRelations, trainingMatrix, groundTruth)
        GScores = powerLawCalculations(
            params['datasetName'], users, pois, trainingMatrix, poiCoos, groundTruth)

        # Segmenting active users
        activeUsers = calculateActiveUsers(params['datasetName'], datasetFiles['train'])

        # Score calculation
        # (Moving this before evaluation so that we can test reranking methods)
        evalParams = {'usersList': users['list'], 'usersCount': users['count'],
                      'groundTruth': groundTruth, 'fusion': params['fusion'], 'poiList': pois['list'],
                      'trainingMatrix': trainingMatrix, 'evaluation': params['evaluation'],
                      'fusionWeights': params['fusionWeights'], 'poiCoos': poiCoos,
                      'fairness': params['fairness']}
        modelParams = {'U': UScores, 'S': SScores, 'G': GScores}

        # Add fairness modules as needed
        if 'Provider' == params['fairness']:
            IScores = ItemExposureCalculations(
                params['datasetName'], users, pois, poiCheckinCounts, groundTruth)
            modelParams['I'] = IScores

        predictions, scores = calculateScores(
            modelName, evalParams, modelParams, listLimit)

        # Reranking
        predictions = rerankPredictions(
            params['reranker'],
            predictions,
            k=topK,
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
