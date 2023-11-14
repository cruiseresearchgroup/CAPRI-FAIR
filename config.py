import os

# Paths
dataDirectory = os.path.dirname(os.path.realpath(__file__)) + '/Data'
outputsDir = os.path.abspath('./Outputs/')

# Default Parameters
topK = 20  # Top-k items to evaluate (default: 10)
limitUsers = -1  # Limit the number of users (default: -1)
listLimit = 30  # Limit the length of recommendation list (default: 10)
activeUsersPercentage = [5, 20]  # Calculate [n] percents of users as active
itemExposureScalingFactor = 5 # Scaling factor for the item exposure reranking

# Key: Model name, Value: Covered Contexts
models = {
    "GeoSoCa": ["Geographical", "Social", "Categorical"],
    "LORE": ["Geographical", "Social", "Temporal"],
    "USG": ["Interaction", "Social", "Geographical"],
}
exposureModels = ["Linear", "PowerLaw", "Logistic"]

# List of reranking methods
rerankers = ["TopK", "Random", "ItemExposure"]

# Key: Dataset name, Value: Covered Contexts
datasets = {
    "Gowalla": ["Geographical", "Social", "Temporal", "Interaction"],
    "Yelp":  ["Geographical", "Social", "Temporal", "Categorical", "Interaction"],
    "Foursquare":  ["Geographical", "Social", "Temporal", "Interaction"],
}

# An array of selected operations
# TODO: "WeightedSum" is not implemented yet
fusions = ["Product", "Sum", "WeightedSum"]
fusionWeights = [0.3, 0.3, 0.2, 0.2]

# List of evaluation metrics
evaluationMetrics = ["Precision", "Recall", "mAP", "NDCG"]

# List of additional fairness context modules
fairnessModules = ["None", "Provider", "Consumer", "Both"]
FairnessDict = {"provider": 0.5, "consumer": 0.5}

# Models Dictionaries
GeoSoCaDict = {"alpha": 0.5}
LoreDict = {"alpha": 0.05, "deltaT": 3600 * 24}
USGDict = {"alpha": 0.1, "beta": 0.1, "eta": 0.05}
