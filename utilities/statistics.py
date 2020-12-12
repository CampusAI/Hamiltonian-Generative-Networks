import numpy as np
import scipy.stats


def mean_confidence_interval(data, confidence=0.95):
    """Calculates mean and confidence interval from samples such that they lie within m +/- h 
    with the given confidence.

    Args:
        data (np.array): Sample to calculate the confidence interval.
        confidence (float): Confidence of the interval (betwen 0 and 1).
    """
    n = len(data)
    m, se = np.mean(data), scipy.stats.sem(data)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h
