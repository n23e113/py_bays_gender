import tensorflow as tf
import numpy as np
from python_distGender import F1Score


def BayesClassifierFun(X, Y, trainCheckParam):
    allDataRow = X.shape[0]
    trainRow = int(trainCheckParam * allDataRow)
    trainX = X[:trainRow, :]
    trainY = Y[:trainRow, :]
    p_y_0, p_y_1, x_v_0, x_v_1 = computParam(trainX, trainY)
    checkX = X[trainRow:, :]
    checkY = Y[trainRow:, :]

    pY = predict(p_y_0, p_y_1, x_v_0, x_v_1, checkX)
    acc = F1Score.f1_score(checkY, pY)
    return acc
    pass


def computParam(X, Y):

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

    tp_y_0 = n_y_0 / y_n
    tp_y_1 = n_y_1 / y_n
    tx_v_0 = x_v_0_t / tf.cast(n_y_0, tf.float32)
    tx_v_1 = x_v_1_t / tf.cast(n_y_1, tf.float32)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    p_y_0, p_y_1, x_v_0, x_v_1 = sess.run(
        [tp_y_0, tp_y_1, tx_v_0, tx_v_1], feed_dict={tX: X, tY: Y})
    sess.close()
    return p_y_0, p_y_1, x_v_0, x_v_1


def predict(p_y_0, p_y_1, x_v_0, x_v_1, checkX):
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

    tp_y_0_x = log_p_y_0 + tf.reduce_sum(log_check_x_0, axis=1)
    tp_y_1_x = log_p_y_1 + tf.reduce_sum(log_check_x_1, axis=1)

    tPredictY_ = tf.where(tp_y_0_x >= tp_y_1_x, tf.zeros(
        tp_y_0_x.shape, tf.int32), tf.ones(tp_y_0_x.shape, tf.int32))
    tPredictY = tf.reshape(tPredictY_, [-1, 1])
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    predictY, p_y_0_x, p_y_1_x, rcheck_x_0, rcheck_x_1, rlog_check_x_0, rlog_check_x_1 = sess.run(
        [tPredictY, tp_y_0_x, tp_y_1_x, check_x_0, check_x_1, log_check_x_0, log_check_x_1], feed_dict={
            tp_y_0: p_y_0, tp_y_1: p_y_1,
            tx_v_0: x_v_0, tx_v_1: x_v_1, tcheckX: checkX})

    return predictY

#     predict = np.where(p_y_0_x >= p_y_1_x, 0, 1)


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
