from . import basic

def rerankPredictions(reranker, predictions, **kwargs):
    """
    adsfasd
    """

    match reranker:
        case "TopK":
            return basic.topk_ranking(predictions)
        case "Random":
            return basic.random_ranking(predictions)
        case _:
            raise ValueError("Reranking method not found or implemented")