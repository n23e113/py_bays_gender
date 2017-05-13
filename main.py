import GenderFile
import io
import sys
import numpy as np
import BayesClassifier

if __name__ == '__main__':
    print("start")
    #sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='gb18030')
    print("start2")
    gf = GenderFile.GenderFile(
        "D:\coding\eclipse\projects\DistinguishGender/python_distGender/aiwar_train_data",
        "D:\coding\eclipse\projects\DistinguishGender/python_distGender/one_hot_data", 80000)
    gf.restore(False)
    acc = BayesClassifier.BayesClassifierFun(gf.getX(), gf.getY(), 0.7)
    print(acc)
    pass

if __name__ == '__main__C':
    X = np.array([(1, 0, 1, 0, 0, 0), (1, 0, 0, 1, 0, 0), (0, 1, 0, 0, 1, 0),
                  (0, 1, 0, 0, 1, 0), (1, 0, 0, 0, 0, 1), (0, 1, 0, 0, 0, 1)])
    Y = np.array([0, 1, 1, 0, 0, 1])
    p_y_0, p_y_1, x_v_0, x_v_1 = BayesClassifier.computParam(X, Y)
    checkX = np.array([(1, 0, 0, 0, 1, 0), (0, 1, 1, 0, 0, 0)])
    checkY = np.array([0, 1])
    acc = BayesClassifier.classifier(
        p_y_0, p_y_1, x_v_0, x_v_1, checkX, checkY)
    print(acc)
    pass
