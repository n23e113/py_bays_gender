import tensorflow as tf
import numpy as np
import F1Score
from numpy.ma.core import arange

g_nSplitCount = 20


def saveParam(p_y_0, p_y_1, x_v_0, x_v_1):
    np.save("py0", p_y_0)
    np.save("py1", p_y_1)
    np.save("xv0", x_v_0)
    np.save("xv1", x_v_1)


def restoreParam():
    try:
        p_y_0 = np.load("py0.npy")
        p_y_1 = np.load("py1.npy")
        x_v_0 = np.load("xv0.npy")
        x_v_1 = np.load("xv1.npy")
    except Exception:
        return np.nan, np.nan, np.nan, np.nan
    return p_y_0, p_y_1, x_v_0, x_v_1


def BayesClassifierFun(X, Y, trainCheckParam, hParam=1.0):
    allDataRow = X.shape[0]
    trainRow = int(trainCheckParam * allDataRow)
    trainX = X
    trainY = Y
    checkX = X[trainRow:, :]
    checkY = Y[trainRow:, :]

    p_y_0, p_y_1, x_v_0, x_v_1 = restoreParam()
    if (np.isnan(p_y_0)):
        p_y_0, p_y_1, x_v_0, x_v_1 = computParam(trainX, trainY)
        saveParam(p_y_0, p_y_1, x_v_0, x_v_1)

    pY = predict(p_y_0, p_y_1, x_v_0, x_v_1, checkX, hParam)
    acc = F1Score.f1_score(checkY, pY)
    return acc
    pass


def computParam(X, Y):
    ncount = min(g_nSplitCount, Y.shape[0])
    lx = np.vsplit(X, ncount)
    ly = np.vsplit(Y, ncount)
    mer_n_y_0 = 0
    mer_n_y_1 = 0
    mer_x_v_0 = np.zeros((1, X.shape[1]))
    mer_x_v_1 = np.zeros((1, X.shape[1]))
    idx = 0
    print("begin computParam")
    for i in arange(ncount):
        rn_y_0, rn_y_1, rx_v_0, rx_v_1 = computSubParam(lx[i], ly[i])

        mer_n_y_0 += rn_y_0
        mer_n_y_1 += rn_y_1
        mer_x_v_0 += rx_v_0
        mer_x_v_1 += rx_v_1
        print("processing computParam %(process)d" %
              {'process': (idx)})
        idx += 1
    print("end computParam")

    mer_n_y = mer_n_y_0 + mer_n_y_1
    p_y_0 = mer_n_y_0 / mer_n_y
    p_y_1 = mer_n_y_1 / mer_n_y
    x_v_0 = mer_x_v_0 / mer_n_y_0
    x_v_1 = mer_x_v_1 / mer_n_y_1
    return p_y_0, p_y_1, x_v_0, x_v_1


def computSubParam(X, Y):
    tX = tf.placeholder(tf.float32, X.shape)
    tY = tf.placeholder(tf.float32, Y.shape)
    y_n = tf.Variable(tY.shape[0], dtype=tf.int64)
    n_y_1 = tf.count_nonzero(tY)
    n_y_0 = y_n - n_y_1

    allOne = tf.ones_like(tY)
    allZero = tf.zeros_like(tY)
    tY_n = tf.where(tf.equal(tY, allZero), allOne, allZero)

    x_v_0_t = tf.transpose(tf.matmul(tX, tY_n, True, False))
    x_v_1_t = tf.transpose(tf.matmul(tX, tY, True, False))

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    rn_y_0, rn_y_1, rx_v_0, rx_v_1 = sess.run(
        [n_y_0, n_y_1, x_v_0_t, x_v_1_t], feed_dict={tX: X, tY: Y})
    sess.close()
    return rn_y_0, rn_y_1, rx_v_0, rx_v_1


def predict(p_y_0, p_y_1, x_v_0, x_v_1, checkX, hParam=1.0):
    ncount = min(g_nSplitCount, checkX.shape[0])
    lx = np.vsplit(checkX, ncount)

    ly = []
    idx = 0
    print("begin predict")
    for i in arange(ncount):
        ly.append(predictSub(p_y_0, p_y_1, x_v_0, x_v_1, lx[i], hParam))

        print("processing predict %(process)d" %
              {'process': (idx)})
        idx += 1
    print("end predict")

    return np.vstack(ly)


def predictSub(p_y_0, p_y_1, x_v_0, x_v_1, checkX, hParam=1.0):
    tp_y_0 = tf.placeholder(tf.float32, p_y_0.shape)
    tp_y_1 = tf.placeholder(tf.float32, p_y_1.shape)
    tx_v_0 = tf.placeholder(tf.float32, x_v_0.shape)
    tx_v_1 = tf.placeholder(tf.float32, x_v_1.shape)
    tcheckX = tf.placeholder(tf.float32, checkX.shape)
    allZero = tf.zeros_like(tcheckX)

    check_x_0 = tf.multiply(tcheckX, tx_v_0)
    log_check_x_0 = tf.where(
        tf.equal(check_x_0, allZero), allZero, tf.log(check_x_0))
    log_p_y_0 = tf.log(tp_y_0)

    check_x_1 = tf.multiply(tcheckX, tx_v_1)
    log_check_x_1 = tf.where(
        tf.equal(check_x_1, allZero), allZero, tf.log(check_x_1))
    log_p_y_1 = tf.log(tp_y_1)
    tp_log_hParam = tf.log(hParam)
    tp_y_0_x = log_p_y_0 + tf.reduce_sum(log_check_x_0, axis=1)
    tp_y_1_x = log_p_y_1 + tf.reduce_sum(log_check_x_1, axis=1) + tp_log_hParam

    tPredictY_ = tf.where(tp_y_0_x >= tp_y_1_x, tf.zeros(
        tp_y_0_x.shape, tf.int32), tf.ones(tp_y_0_x.shape, tf.int32))
    tPredictY = tf.reshape(tPredictY_, [-1, 1])
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    predictY = sess.run(
        [tPredictY], feed_dict={
            tp_y_0: p_y_0, tp_y_1: p_y_1,
            tx_v_0: x_v_0, tx_v_1: x_v_1, tcheckX: checkX})

    return np.reshape(predictY, [-1, 1])


def classifier(p_y_0, p_y_1, x_v_0, x_v_1, checkX, checkY):
    pass


if __name__ == '__main__':
    X = np.array([(1, 0, 1, 0, 0, 0), (1, 0, 0, 1, 0, 0), (0, 1, 0, 0, 1, 0),
                  (0, 1, 0, 0, 1, 0), (1, 0, 0, 0, 0, 1), (0, 1, 0, 0, 0, 1)])
    Y = np.array([[0], [1], [1], [0], [0], [1]])

    p_y_0, p_y_1, x_v_0, x_v_1 = computParam(X, Y)
    checkX = np.array([(1, 0, 0, 0, 1, 0), (0, 1, 1, 0, 0, 0)])
    checkY = np.array([[0], [1]])
    pY = predict(p_y_0, p_y_1, x_v_0, x_v_1, checkX)
    acc = F1Score.f1_score(checkY, pY)
    print(acc)
    pass
