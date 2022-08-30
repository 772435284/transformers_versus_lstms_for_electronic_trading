'''
metrics.py
Based on: https://github.com/thuml/Autoformer/blob/main/utils/metrics.py
'''
import numpy as np
from sklearn.metrics import r2_score


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)

def Rsquared(pred, true):
    true = true.reshape(true.shape[-1], true.shape[0], true.shape[1])
    pred = pred.reshape(pred.shape[-1], pred.shape[0], pred.shape[1])
    # true  = true.squeeze()
    # pred  = pred.squeeze()
    scores = []
    for i in range(true.shape[0]):
        r = r2_score(true[i],pred[i],multioutput="raw_values")
        scores.append(r)
    #r = r2_score(true,pred,multioutput="raw_values")
    return scores

def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe
