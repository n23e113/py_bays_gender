import numpy as np


def precision_score(y_true, y_pred):
    return ((y_true == 1) * (y_pred == 1)).sum() / (y_pred == 1).sum()


def recall_score(y_true, y_pred):
    return ((y_true == 1) * (y_pred == 1)).sum() / (y_true == 1).sum()


def f1_score(y_true, y_pred):
    num = 2 * precision_score(y_true, y_pred) * recall_score(y_true, y_pred)
    deno = (precision_score(y_true, y_pred) + recall_score(y_true, y_pred))
    return num / deno
