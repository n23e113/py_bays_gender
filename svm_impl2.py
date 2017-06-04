import numpy as np
import random as rd


def Gauss_kernel(x, z, sigma=2):
    return np.exp(-np.sum((x - z)**2) / (2 * sigma**2))


def Linear_kernel(x, z):
    return np.sum(x * z)


kernal = Linear_kernel


def select(i, X, Y, A, b, E):
    pp = np.nonzero((A > 0))[0]
    if (pp.size > 0):
        j = findMax(i, pp, X, Y, A, b, E)
    else:
        j = findMax(i, range(A.shape[0]), X, Y, A, b, E)
    return j


def randJ(i, M):
    j = rd.sample(range(M), 1)
    while j == i:
        j = rd.sample(range(M), 1)
    return j[0]


def findMax(i, ls, X, Y, A, b, E):
    ansj = -1
    maxx = -1
    updateE(i, X, Y, A, b, E)
    for j in ls:
        if i == j:
            continue
        updateE(j, X, Y, A, b, E)
        deltaE = np.abs(E[i] - E[j])
        if deltaE > maxx:
            maxx = deltaE
            ansj = j
    if ansj == -1:
        return randJ(i)
    return ansj


def pred(X, Y, A, b, x_i):
    m = X.shape[0]
    ret = 0
    for i in range(m):
        ret += A[i] * Y[i] * kernal(X[i], x_i)
    return ret + b


def updateE(i, X, Y, A, b, E):
    E[i] = pred(X, Y, A, b, X[i]) - Y[i]


def svm_train(C, tol, max_passes, X, Y, threshold):
    m = X.shape[0]
    A = np.zeros(m)
    b = 0
    E = np.zeros(m)
    passes = 0
    while (passes < max_passes):
        num_changed_alphas = 0
        for i in range(m):
            a_i = A[i]
            x_i = X[i]
            y_i = Y[i]
            e_i = pred(X, Y, A, b, x_i) - y_i
            E[i] = e_i
            if (y_i * e_i < tol and a_i < C) or (y_i * e_i > tol and a_i > C):
                j = select(i, X, Y, A, b, E)
                a_j = A[j]
                x_j = X[j]
                y_j = Y[j]
                e_j = pred(X, Y, A, b, x_j) - y_j
                E[j] = e_j
                a_i_old = a_i
                a_j_old = a_j
                L = 0
                H = 0
                if (y_i != y_j):
                    L = max(0, a_j - a_i)
                    H = min(C, C + a_j - a_i)
                else:
                    L = max(0, a_i + a_j - C)
                    H = min(C, a_i + a_j)
                if (L == H):
                    continue

                K11 = kernal(x_i, x_i)
                K22 = kernal(x_j, x_j)
                K12 = kernal(x_i, x_j)
                eta = 2 * K12 - K11 - K22
                if (eta >= 0):
                    continue
                a_j = a_j - y_j * (e_i - e_j) / eta
                if a_j > H:
                    a_j = H
                elif a_j < L:
                    a_j = L
                A[j] = a_j
                if (np.abs(a_j - a_j_old) < threshold):
                    continue

                a_i = a_i + y_i * y_j * (a_j_old - a_j)
                A[i] = a_i
                b1 = b - e_i - y_i * (a_i - a_i_old) * K11 - \
                    y_j * (a_j - a_j_old) * K12
                b2 = b - e_i - y_i * (a_i - a_i_old) * K12 - \
                    y_j * (a_j - a_j_old) * K22

                if 0 < a_i and a_i < C:
                    b = b1
                elif 0 < a_j and a_j < C:
                    b = b2
                else:
                    b = (b1 + b2) / 2
                num_changed_alphas += 1
        if num_changed_alphas == 0:
            passes += 1
        else:
            passes = 0

    return A, b


def capW(A, X, Y):
    W = np.sum(A * Y * X, axis=0)


if __name__ == '__main__':
    X = np.array([(6, 8), (2, 3), (40, 56), (98, 23),
                  (40, 10), (23, 20), (15, 16)])
    Y = np.array([(1), (1), (1), (-1), (-1), (-1), (1)])
    X_Check = np.array([(6, 10), (13, 10), (10, 20),
                        (80, 48), (20, 30), (56, 59), (531, 200)])
    Y_Check = np.array([(1), (-1), (1), (-1), (1), (1), (-1)])
    YP_Check = np.zeros_like(Y_Check)
    A, b = svm_train(0.5, 0.1, 50, X, Y, 0.000001)
    for i in range(X_Check.shape[0]):
        x_i = X_Check[i]
        YP_Check[i] = pred(X, Y, A, b, x_i)
    pass
