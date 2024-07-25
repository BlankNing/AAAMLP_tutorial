import scipy.stats as stats
import numpy as np

def rank_mean(probas):
    """
    Create mean predictions using ranks
    :param probas: 2-d array of probability values
    :return: mean ranks
    """
    ranked = []
    for i in range(probas.shape[1]):
        rank_data = stats.rankdata(probas[:, i])
        ranked.append(rank_data)

    ranked = np.column_stack(ranked)
    return np.mean(ranked, axis=1)
