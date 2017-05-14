import numpy as np
import os


class DataRawItem(object):
    def __init__(self, dataId, gender, platform, applist, area):
        self.dataId = dataId
        self.gender = gender
        self.platform = platform
        self.area = area
        self.appNm = np.fromstring(applist, dtype=int, sep=",")

    def getMax(self):
        return self.appNm.max()

    def getXData(self, maxType):
        ra = np.zeros(maxType, dtype=GenderFile.dtype)
        for elem in self.appNm.flat:
            ra[int(elem)] = True
        return ra

    def getYData(self):
        if self.gender == "male":
            return (0)
        else:
            return (1)


class GenderFile:
    dtype = np.bool
    proportion = 0.7
    splitCount = 20
    XFileName = "/appXData_%d.npy"
    YFileName = "/appYData_%d.npy"

    def __init__(self, path, tempPath, maxDataCount):
        self.__path = path
        self.__maxDataCount = maxDataCount
        self.__dataRow = 0
        self.__dataColumn = 0
        self.__data = np.zeros((0, 0), dtype=self.dtype)
        self.__y = np.zeros((0, 0), dtype=np.int)
        self.__trainIdx = 0
        self.__listRawData = []
        self.__tempDirPath = tempPath
        self.__cache = False

        pass

    def getX(self):
        return self.__data

    def getY(self):
        return self.__y

    def restore(self, loadCache):
        self.__cache = loadCache
        if loadCache and (os.path.exists(self.__tempDirPath)):
            lstData = []
            for i in range(GenderFile.splitCount):
                lstData.append(
                    np.load(self.__tempDirPath + self.XFileName % (i)))

            self.__data = np.vstack(lstData)

            sz = self.__data.shape
            self.__dataRow = sz[0]
            self.__dataColumn = sz[1]

            lstData = []
            for i in range(GenderFile.splitCount):
                lstData.append(
                    np.load(self.__tempDirPath + self.YFileName % (i)))
            self.__y = np.vstack(lstData)
        else:
            self.readFile()
            self.buildAppData()
            self.buildYData()

    def readFile(self):
        fo = open(self.__path, mode='r', encoding='UTF-8')
        contents = fo.readlines()
        fo.close()

        szContents = len(contents)
        self.__dataRow = szContents
        if self.__maxDataCount != -1:
            self.__dataRow = min(self.__maxDataCount, self.__dataRow)
        self.__trainIdx = self.proportion * self.__dataRow
        onepice = int(self.__dataRow / 100)
        idx = 0
        print("begin buildRawData")
        for eachLine in contents:
            if idx >= self.__dataRow:
                break

            if (idx % onepice == 0):
                print("processing buildRawData %(process)d" %
                      {'process': (idx / onepice)})
            idx += 1
            if eachLine:
                items = eachLine.split("\t")
                dataRawItem = DataRawItem(
                    items[0], items[1], items[2], items[3], items[4])
                self.__listRawData.append(dataRawItem)
        print("end buildRawData")

    def buildAppData(self):
        appTypNum = 0
        for elem in self.__listRawData:
            appTypNum = max(elem.getMax(), appTypNum)

        self.__dataColumn = int(appTypNum) + 1
        self.__data.resize((self.__dataRow, self.__dataColumn))

        onepice = int(self.__dataRow / 100)
        idx = 0
        print("begin buildAppData")
        for elem in self.__listRawData:
            self.__data[idx] = elem.getXData(self.__dataColumn)

            if (idx % onepice == 0):
                print("processing buildAppData %(process)d" %
                      {'process': (idx / onepice)})
            idx += 1
        print("end buildAppData")

        if self.__cache == False:
            return
        lstData = np.vsplit(self.__data, GenderFile.splitCount)

        if (os.path.exists(self.__tempDirPath)):
            __import__('shutil').rmtree(self.__tempDirPath)
        os.makedirs(self.__tempDirPath)
        idx = 0
        for pick_a in lstData:
            np.save(self.__tempDirPath + self.XFileName % (idx), pick_a)
            idx += 1
        #np.save(self.__tempDirPath, self.__data)
        # self.__data.tofile(self.__tempDirPath)

    def buildYData(self):
        self.__y.resize(self.__dataRow, 1)
        idx = 0
        for elem in self.__listRawData:
            self.__y[idx] = elem.getYData()
            idx += 1

        if self.__cache == False:
            return
        lstData = np.vsplit(self.__y, GenderFile.splitCount)
        idx = 0
        for pick_a in lstData:
            np.save(self.__tempDirPath + self.YFileName % (idx), pick_a)
            idx += 1

    def printData(self):
        print(self.__data)
