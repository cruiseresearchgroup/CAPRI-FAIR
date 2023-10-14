from utils import logger
import inquirer
# from inquirer import prompt
from config import datasets, models, fusions, evaluationMetrics

modelChoices = []
fusionChoices = []
datasetChoices = []
evaluatorChoices = []


def initChoices():
    """
    Preparing choices for the questions

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    # Preparing model items
    for model in models:
        modelChoices.append(model)
    # Preparing dataset items
    for dataset in datasets:
        datasetChoices.append(dataset)
    # Preparing fusion items
    for fusion in fusions:
        fusionChoices.append(fusion)
    # Preparing evaluation metrics items
    for evaluator in evaluationMetrics:
        evaluatorChoices.append({'name': evaluator})


def interactiveCommandForm():
    """
    Generating the interactive form for the user to select the parameters

    Parameters
    ----------
    None

    Returns
    -------
    userInputs: dict
        Dictionary containing the user inputs
    """
    # Initiate choices
    initChoices()
    # Appy choices to the questions
    questions = [
        inquirer.List('Model',
            message="Choose the model you need:",
            choices=modelChoices
        ),
        inquirer.List('Dataset',
            message="Choose the dataset you need:",
            choices=datasetChoices
        ),
        inquirer.List('Fusion',
            message="Choose the fusion you need:",
            choices=fusionChoices
        ),
        inquirer.Checkbox('Evaluation',
            message="Choose at least one evaluation metric:",
            choices=evaluatorChoices
        ),
        inquirer.Confirm('Confirmation',
            message="Do you confirm your selected choices?",
            default=True
        ),
    ]
    # Showing the selected items to the user
    userInputs = inquirer.prompt(questions)
    return userInputs


def getUserChoices():
    """
    Getting the user inputs and validating them

    Parameters
    ----------
    None

    Returns
    -------
    userInputs: dict
        Dictionary containing the user inputs
    """
    userInputs = interactiveCommandForm()
    confirmation = userInputs['Confirmation']
    if (confirmation == True):
        print('Validating your choices ...')
        selectedModelScopes = models[userInputs['Model']]
        selectedDatasetScopes = datasets[userInputs['Dataset']]
        ignoredContexts = []
        # Checking if dataset covers all scopes of models
        isCovered = all(
            item in selectedDatasetScopes for item in selectedModelScopes)
        if (not isCovered):
            difference = [
                item for item in selectedModelScopes if item not in selectedDatasetScopes]
            printMessage = f'Ignoring {difference} scope(s) of {userInputs["Model"]}, as not covered in {userInputs["Dataset"]}!'
            logger(printMessage, 'warn')
            ignoredContexts = difference
        # Checking if at least one evaluation metric is selected
        if (len(userInputs['Evaluation']) == 0):
            printMessage = 'No evaluation metric has been selected!'
            logger(printMessage, 'error')
            return
        logger(f'User inputs: {userInputs}', 'info', True)
        userInputs['Ignored'] = ignoredContexts
        return userInputs
