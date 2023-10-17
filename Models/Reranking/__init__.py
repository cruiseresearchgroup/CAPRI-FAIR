from . import basic
from . import fairness

def rerankPredictions(reranker, predictions, **kwargs):
    """
    adsfasd
    """

    match reranker:
        case "TopK":
            return basic.topk_ranking(predictions)
        case "Random":
            return basic.random_ranking(predictions)
        case "ItemExposure":
            return fairness.item_exposure_ranking(predictions, **kwargs)
        case _:
            raise ValueError("Reranking method not found or implemented")