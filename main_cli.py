import argparse
import logging

from utils import logger
from config import datasets, models, rerankers, fusions, evaluationMetrics, fairnessModules, fusionWeights, FairnessDict, topK, exposureModels
from commandParser import getUserChoices
from Data.loadDatasetFiles import loadDatasetFiles


parser = argparse.ArgumentParser()
parser.add_argument('model', help=f"Recommender model to use ({','.join(models)})")
parser.add_argument('dataset', help=f"Dataset to test on ({','.join(datasets)})")
parser.add_argument('fusion', help=f"Fusion method for the operands ({','.join(fusions)})")
parser.add_argument('--reranker', nargs='?', default=rerankers[0], help=f"Post-filter reranking method to use to use ({','.join(rerankers)})")
parser.add_argument('--fairness', nargs='?', default=fairnessModules[0], help=f"Fairness context to consider ({','.join(fairnessModules)})")
parser.add_argument('--provider_alpha', nargs='?', default=FairnessDict['provider'], help=f"Coefficient of provider fairness factor")
parser.add_argument('--exposure_model', nargs='?', default=exposureModels[0], help=f"Coefficient of provider fairness factor")
parser.add_argument('--consumer_beta', nargs='?', default=FairnessDict['consumer'], help=f"Coefficient of consumer fairness factor")
parser.add_argument('--evaluation', nargs='*', help=f"Metrics to evaluate ({','.join(evaluationMetrics)})")
parser.add_argument('--k', nargs='?', default=topK, help=f"Number of recommended POIs to evaluate per user")

if __name__ == '__main__':
    args = parser.parse_args()

    print('Validating your choices ...')
    selectedModelScopes = models[args.model]
    selectedDatasetScopes = datasets[args.dataset]
    ignoredContexts = []
    # Checking if dataset covers all scopes of models
    isCovered = all(
        item in selectedDatasetScopes for item in selectedModelScopes)
    if (not isCovered):
        difference = [
            item for item in selectedModelScopes if item not in selectedDatasetScopes]
        printMessage = f'Ignoring {difference} scope(s) of {args.model}, as not covered in {args.dataset}!'
        logger(printMessage, 'warn')
        ignoredContexts = difference
    # Checking if at least one evaluation metric is selected
    if (len(args.evaluation) == 0):
        printMessage = 'No evaluation metric has been selected!'
        logger(printMessage, 'error')
        exit()
    logger(f'User inputs: {args.__dict__}', 'info', True)

    # Initializing dataset items
    datasetFiles = loadDatasetFiles(args.dataset)
    logger(f'Dataset files: {datasetFiles}', 'info', True)
    # Exiting the program if dataset is not found
    if (datasetFiles == None):
        exit()

    fairnessWeights = FairnessDict.copy()
    fairnessWeights['provider'] = float(args.provider_alpha)
    fairnessWeights['consumer'] = float(args.consumer_beta)
    # Initializing parameters
    parameters = {
        "topK": int(args.k),
        "reranker": args.reranker,
        "fusion": args.fusion,
        "ignored": ignoredContexts,
        "fairness": args.fairness,
        "exposureModel": args.exposure_model,
        "datasetName": args.dataset,
        "evaluation": [{'name': e} for e in args.evaluation],
        "fusionWeights": fusionWeights,
        "fairnessWeights": fairnessWeights
    }
    logger(f'Processing parameters: {parameters}', 'info', True)
    # Dynamically loading the model
    module = __import__(
        'Models.' + args.model + '.main', fromlist=[''])
    selectedModule = getattr(module, args.model + 'Main')
    # Select the 'main' class in the module
    selectedModule.main(datasetFiles, parameters)
    # Closing the log file
    logger('CAPRI framework finished!')