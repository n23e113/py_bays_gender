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
        "D:\coding\eclipse\projects\DistinguishGender/python_distGender/one_hot_data", 1000)
    gf.restore()
    acc = BayesClassifier.BayesClassifierFun(gf.getX(), gf.getY())
    print(acc)
    pass
