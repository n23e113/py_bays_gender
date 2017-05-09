#import GenderFile
import numpy as np


def BayesClassifierFun(X, Y):
    allDataRow = X.shape[0]
    trainRow = int(0.7 * allDataRow)
    trainX = X[:trainRow, :]
    trainY = Y[:trainRow]
    p_y_0, p_y_1, x_v_0, x_v_1 = computParam(trainX, trainY)
    checkX = X[trainRow:, :]
    checkY = Y[trainRow:]
    return classifier(p_y_0, p_y_1, x_v_0, x_v_1, checkX, checkY)


def classifier(p_y_0, p_y_1, x_v_0, x_v_1, check_x, check_y):
    check_x_0 = check_x * x_v_0
    log_check_x_0 = np.where(check_x_0 != 0, np.log(check_x_0), 0)
    log_p_y_0 = np.where(p_y_0 != 0, np.log(p_y_0), 0)

    check_x_1 = check_x * x_v_1
    log_check_x_1 = np.where(check_x_1 != 0, np.log(check_x_1), 0)
    log_p_y_1 = np.where(p_y_1 != 0, np.log(p_y_1), 0)

    p_y_0_x = log_p_y_0 + np.sum(log_check_x_0, axis=1)
    p_y_1_x = log_p_y_1 + np.sum(log_check_x_1, axis=1)

    predict = np.where(p_y_0_x >= p_y_1_x, 0, 1)
    resV = np.where(predict == check_y, 1, 0)
    res = np.mean(resV)
    return res


def computParam(X, Y):
    y_n = Y.shape[0]
#    y_each_n = np.bincount(Y)
#    p_y_0 = y_each_n[0] / y_n
#    p_y_1 = y_each_n[1] / y_n

    n_y_0 = 0
    n_y_1 = 0
    column = X.shape[1]
    x_v_0 = np.zeros(column)
    x_v_1 = np.zeros(column)
#     for i in range(y_n):
#         x_i = X[i]
#         if Y[i] == 0:
#             n_y_0 += 1
#             for j in range(column):
#                 x_v_0[j] += x_i[j]
#         else:
#             n_y_1 += 1
#             for j in range(column):
#                 x_v_1[j] += x_i[j]
    xt = X.transpose()

    Y_n = np.where(Y == 0, 1, 0)
    x_v_0 = np.dot(xt, Y_n).transpose()
    x_v_1 = np.dot(xt, Y).transpose()

    x_v_0 = x_v_0 / n_y_0
    x_v_1 = x_v_1 / n_y_1
    p_y_0 = n_y_0 / y_n
    p_y_1 = n_y_1 / y_n
    return p_y_0, p_y_1, x_v_0, x_v_1
