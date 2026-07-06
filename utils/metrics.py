import numpy as np


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    mape = np.abs((pred - true) / true)
    mape = np.where(mape > 5, 0, mape)
    return np.mean(mape)


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


# -------------------------
# Additional Regression Metrics
# -------------------------

def R2(pred, true):
    """
    Coefficient of Determination (R²)

    Best = 1
    Can be negative if the model performs worse than predicting the mean.
    """
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - np.mean(true)) ** 2)

    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0

    return 1 - (ss_res / ss_tot)

def Explained_Variance(pred, true):
    """
    Explained Variance Score

    Best = 1
    """
    numerator = np.var(true - pred)
    denominator = np.var(true)

    if denominator == 0:
        return 1.0

    return 1 - numerator / denominator


def MedianAE(pred, true):
    """
    Median Absolute Error
    """
    return np.median(np.abs(pred - true))


def MaxError(pred, true):
    """
    Maximum Absolute Error
    """
    return np.max(np.abs(pred - true))


def SMAPE(pred, true):
    """
    Symmetric Mean Absolute Percentage Error
    """
    denominator = np.abs(true) + np.abs(pred)
    denominator = np.where(denominator == 0, 1e-8, denominator)

    return np.mean(2 * np.abs(pred - true) / denominator)


def WAPE(pred, true):
    """
    Weighted Absolute Percentage Error
    """
    denominator = np.sum(np.abs(true))

    if denominator == 0:
        return np.nan

    return np.sum(np.abs(pred - true)) / denominator


# -------------------------
# Main Metric Function
# -------------------------

def metric(pred, true, n_features=None):

    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    r2 = R2(pred, true)
    evs = Explained_Variance(pred, true)
    medae = MedianAE(pred, true)
    maxerr = MaxError(pred, true)
    smape = SMAPE(pred, true)
    wape = WAPE(pred, true)

    return mae, mse, rmse, mape, mspe, r2, evs, medae, maxerr, smape, wape