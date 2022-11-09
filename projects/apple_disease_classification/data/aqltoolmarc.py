import csv

# import tensorflow as tf
# from ..models import predict_model as pm
import batchselector as bs
import pandas as pd
import sys


model = ""


def readinp(pathfile):
    """
    read input aql-tool
    """
    g = open(pathfile, "r")
    inplist = csv.reader(g, delimiter=";")
    inp = {k: v for k, v in inplist}
    return inp


class Tfmodel:
    """
    Tensorflow model class
    """

    def __init__(self, pathfile):
        sys.path.append(r"C:\Users\marcr\MakeAIWork2\projects\maiw_project2\src\models")
        # self.modelname =
        self.model = tf.keras.models.load_model(pathfile)

    def predictbatch(batchpathfile):
        """voorspelling maken"""
        predictresult = pm.predictfromdata(self.model, batchpathfile)
        return predictresult


class Aqltool:
    def __init__(self):
        self.__sizecode_gil1 = self._sizecode_gil1("sizecode_gil1.csv")
        self.__aqlcode_gil1 = self._aqlcode_gil1("aqlcode_gil1.csv")

        print("Aqltool initialized....")
        pass

    def _aqlcode_gil1(self, aqlcodefp):
        # read file aqlcode tabel
        df = pd.read_csv(aqlcodefp, delimiter=";", index_col=0, header=[0, 1])
        return df

    def _sizecode_gil1(self, sizecodefp):
        # read file sizecode tabel
        df = pd.read_csv(sizecodefp, delimiter=";")
        return df

    def return_sizecode_gil1(self, lotsize):
        #
        test = self.__sizecode_gil1[:]["Max Lotsize"] > lotsize
        if any(test):
            sizecode = self.__sizecode_gil1[:]["aqlcode"][test.argmax()]
        else:
            sizecode = "N"
        # [test.argmax()]
        return sizecode  # self._sizecode_gil1.loc

    def return_aqlcode(self, sizecode, nrbadapples):
        # print(self.__aqlcode_gil1.keys())
        # if len(temp[temp == True][0]):
        temp = self.__aqlcode_gil1.loc[sizecode] >= nrbadapples
        return temp

        print(temp)
        if False:
            # if sizecode
            # temp2 = self.__aqlcode_gil1.iloc[rowindex] > nrbadapples
            xx
        aqlcode = temp.idxmax()
        # return self.__aqlcode_gil1
        return aqlcode
        pass


def test():
    print(dir())


if __name__ == "__main__":

    inputfile = (
        r"C:\Users\marcr\MakeAIWork2\projects\maiw_project2\src\aql_tool\aqltool.inp"
    )
    aqlcsv = readinp(inputfile)

    lotsize = int(aqlcsv["harvestbatchsize"])

    # batch selector
    # batchpathfile= aqlcsv['applesample']
    # bs.batchselector(batchpathfile) #files naar batchpathfile
    # nrbadapples,statistics= Tfmodel.predictbatch(batchpathfile)

    aqlt = Aqltool()
    sizecode = aqlt.return_sizecode_gil1(lotsize)
    aqlt.return_aqlcode(sizecode, nrbadappels)

    if False:
        # sizecode test
        lotsizes = [4, 80, 600, 6000, 1000000]
        for ls in lotsizes:
            print(aqlt.return_sizecode_gil1(ls))

    if False:
        sizecode = "C"
        nrbadapples = [0, 1, 2]
        for ba in nrbadapples:
            # aqlcode test
            x = aqlt.return_aqlcode(sizecode, ba)
            print(x)
