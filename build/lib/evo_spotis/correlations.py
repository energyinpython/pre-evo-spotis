import numpy as np
from scipy.stats import kendalltau

# Spearman coefficient
def spearman_coeff(R, Q):
    """
    Calculate Spearman rank correlation coefficient between two vectors

    Parameters
    ----------
        R : ndarray
            First vector containing values
        Q : ndarray
            Second vector containing values

    Returns
    -------
        float
            Value of correlation coefficient between two vectors
    """
    N = len(R)
    denominator = N*(N**2-1)
    numerator = 6*sum((R-Q)**2)
    rS = 1-(numerator/denominator)
    return rS


# weighted Spearman coefficient rw
def weighted_spearman_coeff(R, Q):
    """
    Calculate Weighted Spearman rank correlation coefficient between two vectors

    Parameters
    ----------
        R : ndarray
            First vector containing values
        Q : ndarray
            Second vector containing values

    Returns
    -------
        float
            Value of correlation coefficient between two vectors
    """
    N = len(R)
    denominator = N**4 + N**3 - N**2 - N
    numerator = 6 * sum((R - Q)**2 * ((N - R + 1) + (N - Q + 1)))
    rW = 1 - (numerator / denominator)
    return rW


# rank similarity coefficient WS
def WS_coeff(R, Q):
    """
    Calculate Rank smilarity coefficient between two vectors

    Parameters
    ----------
        R : ndarray
            First vector containing values
        Q : ndarray
            Second vector containing values

    Returns
    -------
        float
            Value of similarity coefficient between two vectors
    """
    N = len(R)
    numerator = 2**(-R.astype(np.float)) * np.abs(R - Q)
    denominator = np.max((np.abs(R - 1), np.abs(R - N)), axis = 0)
    return 1 - np.sum(numerator / denominator)


# Pearson coefficient
def pearson_coeff(R, Q):
    """
    Calculate Pearson correlation coefficient between two vectors

    Parameters
    ----------
        R : ndarray
            First vector containing values
        Q : ndarray
            Second vector containing values

    Returns
    -------
        float
            Value of correlation coefficient between two vectors
    """
    numerator = np.sum((R - np.mean(R)) * (Q - np.mean(Q)))
    denominator = np.sqrt(np.sum((R - np.mean(R))**2) * np.sum((Q - np.mean(Q))**2))
    corr = numerator / denominator
    return corr


# Kendall rank correlation coefficient
def kendall_coeff(R, Q):
    """
    Calculate Kendall rank correlation coefficient between two vectors

    Parameters
    ----------
        R : ndarray
            First vector containing values
        Q : ndarray
            Second vector containing values

    Returns
    -------
        float
            Value of correlation coefficient between two vectors
    """
    N = len(R)
    Ns, Nd = 0, 0
    for i in range(1, N):
        for j in range(i):
            if ((R[i] > R[j]) and (Q[i] > Q[j])) or ((R[i] < R[j]) and (Q[i] < Q[j])):
                Ns += 1
            elif ((R[i] > R[j]) and (Q[i] < Q[j])) or ((R[i] < R[j]) and (Q[i] > Q[j])):
                Nd += 1

    tau = (Ns - Nd) / ((N * (N - 1))/2)
    return tau


# Goodman Kruskal rank correlation coefficient
def goodman_kruskal_coeff(R, Q):
    """
    Calculate Goodman Kruskal rank correlation coefficient between two vectors

    Parameters
    ----------
        R : ndarray
            First vector containing values
        Q : ndarray
            Second vector containing values

    Returns
    -------
        float
            Value of correlation coefficient between two vectors
    """
    N = len(R)
    Ns, Nd = 0, 0
    for i in range(1, N):
        for j in range(i):
            if ((R[i] > R[j]) and (Q[i] > Q[j])) or ((R[i] < R[j]) and (Q[i] < Q[j])):
                Ns += 1
            elif ((R[i] > R[j]) and (Q[i] < Q[j])) or ((R[i] < R[j]) and (Q[i] > Q[j])):
                Nd += 1

    coeff = (Ns - Nd) / (Ns + Nd)
    return coeff


# Kendall rank correlation coefficient
def kendall_tau_coeff(R, Q):
    """
    Calculate Kendall rank correlation coefficient between two vectors

    Parameters
    ----------
        R : ndarray
            First vector containing values
        Q : ndarray
            Second vector containing values

    Returns
    -------
        float
            Value of correlation coefficient between two vectors
    """
    corr, _ = kendalltau(R, Q)
    return corr