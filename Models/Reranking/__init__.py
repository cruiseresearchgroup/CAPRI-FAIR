from . import basic
from . import fairness

def rerankPredictions(reranker, predictions, topK, **kwargs):
    """
    adsfasd
    """

    match reranker:
        case "TopK":
            return basic.topk_ranking(predictions, topK)
        case "Random":
            return basic.random_ranking(predictions, topK)
        case "ItemExposure":
            return fairness.item_exposure_ranking(predictions, **kwargs)
        case _:
            raise ValueError("Reranking method not found or implemented")