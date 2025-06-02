"""Utils for Mistique."""
import numpy as np
import pandas as pd


def calculate_mspe(df, col1, col2):
    """Calculates MSPE between the two provided columns.

    Args:
        df (pd.DataFrame): DataFrame containing the columns to calculate MSPE.
        col1 (str)
        col2 (str)
    Returns:
        MSPE (float)
    """
    df_calc = df.copy()
    return ((df_calc[col1] - df_calc[col2]) ** 2).mean()


def euclid_distance(s1: pd.Series, s2: pd.Series) -> float:
    """Euclidean distance metric for time series."""
    return np.sqrt(((s1 - s2)**2).sum())


def DTWDistance(s1: pd.Series, s2: pd.Series) -> float:
    """Dynamic time warping distance metric.

    In time series analysis, dynamic time warping (DTW) is an algorithm for measuring similarity
    between two temporal sequences, which may vary in speed.

    Args:
        s1 (pd.Series): Time series data for the first unit.
        s2 (pd.Series): Time series data for the second unit.

    Returns:
        distance (float): The distance between s1 and s2 using dynamic time warping.

    References:
        https://nbviewer.org/github/alexminnaar/time-series-classification-and-clustering/blob/master/Time%20Series%20Classification%20and%20Clustering.ipynb
        https://en.wikipedia.org/wiki/Dynamic_time_warping
    """
    DTW = {}

    for i in range(len(s1)):
        DTW[(i, -1)] = float("inf")
    for i in range(len(s2)):
        DTW[(-1, i)] = float("inf")
    DTW[(-1, -1)] = 0

    for i in range(len(s1)):
        for j in range(len(s2)):
            dist = (s1[i] - s2[j])**2
            DTW[(i, j)] = dist + min(DTW[(i - 1, j)], DTW[(i, j - 1)], DTW[(i - 1, j - 1)])

    return np.sqrt(DTW[len(s1) - 1, len(s2) - 1])
