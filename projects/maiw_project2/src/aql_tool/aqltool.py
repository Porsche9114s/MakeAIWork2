import csv
import os
import sys
import time




os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # surpress warnings

# PATHS TOOL
modelpad = r"projects\maiw_project2\models"
sourcepad = r"projects\maiw_project2\src"
aqltoolpad = os.path.join(sourcepad, "aql_tool")
datapad = r"projects\maiw_project2\data"

sys.path.extend([os.path.join(sourcepad, "models"), os.path.join(sourcepad, "data")])

import predict_model as pm

import tensorflow as tf

# import batchselector as bs  # TO DO: INCORPORATION IN TOOL
import pandas as pd


class Tfmodel:
    """
    TENSORFLOW MODEL CLASS FOR AQLTOOL
    """

    def __init__(self, pathfile):
        self.modelpathfile = pathfile
        self.model = tf.keras.models.load_model(
            os.path.join(modelpad, self.modelpathfile)
        )

    def batchselector(self):
        batch = bs.batchselector()
        return batch

    def predict2truth(self, batchpathfile):
        """RETURN NUMBER OF CORRECT  PREDICTIONS ON SAMPLE"""
        # predictresult, clsnames = pm.predictfromdata(self.modelpathfile, batchpathfile)
        _, nrcorrectpred = pm.pred2truth(batchpathfile, self.modelpathfile)
        return nrcorrectpred

    def predictbatch(self, batchpathfile):
        """RETURN NUMBER OF NOT NORMAL APPLES IN PREDICTION OF SAMPLES"""
        _name_normalapple = "Normal_Apple"

        predictresult, clsnames = pm.predictfromdata(self.modelpathfile, batchpathfile)
        i_norm = clsnames.index(_name_normalapple)
        predictions = [pr.argmax() for pr in predictresult]
        nrnormapples = predictions.count(i_norm)
        nrbadapples = len(predictions) - nrnormapples
        print("Number Normal Apples Predicted:" + str(nrnormapples))
        print("Number Abnormal Apples Predicted:" + str(nrbadapples))
        return nrnormapples, nrbadapples


class Aqltool:
    def __init__(self, inputfile=True):
        print("Startup AQL tool...")
        print("Startup from: " + os.getcwd())
        time.sleep(0.5)
        self.startupcheck()

        self.__sizecode_gil1 = self._sizecode_gil1("sizecode_gil1.csv")
        self.__aqlcode_gil1 = self._aqlcode_gil1("aqlcode_gil1.csv")

        if inputfile == True:
            self.inputfile = os.path.join(aqltoolpad, "aqltool.inp")
        else:
            self.inputfile = inputfile
            # temp = self.readinp(inputfile)
        temp = self.readinp()
        print(temp.keys())
        self.lotsize = int(temp[r"harvestbatchsize"])
        self.applesamplepath = temp["applesample"]
        self.modelpath = temp["model"]
        print("Aqltool initialized....")
        time.sleep(0.5)
        # pass

    def startupcheck(
        self,
    ):
        # CHECKS AT STARTUP TOOL
        print("check paths... ")
        pathlist = [aqltoolpad, modelpad, sourcepad, datapad]
        for path in pathlist:
            assert os.path.exists(path), (
                "path not existing:" + path + "/n change startup path?"
            )
        print("valid pathnames :)")

    def readinp(
        self,
    ):
        """
        READ INPUT AQL-TOOL
        """
        g = open(self.inputfile, "r")
        inplist = csv.reader(g, delimiter=";")

        inp = {k: v for k, v in inplist}
        return inp

    def exportinp(self, pathfile):
        pass

    def _aqlcode_gil1(self, aqlcodefp):
        # READ FILE AQLCODE TABEL
        df = pd.read_csv(
            os.path.join(aqltoolpad, aqlcodefp),
            delimiter=";",
            index_col=0,
            header=[0, 1],
        )
        return df

    def _sizecode_gil1(self, sizecodefp):
        # READ FILE SIZECODE TABEL
        df = pd.read_csv(os.path.join(aqltoolpad, "sizecode_gil1.csv"), delimiter=";")
        return df

    def return_sizecode_gil1(
        self,
    ):
        # DETERMINE SIZECODE SAMPLE PROCEDURE ACCORDING TO LOTSIZE
        print("Determine sizecode according to lotsize...")
        #
        test = self.__sizecode_gil1[:]["Max Lotsize"] > self.lotsize
        if any(test):
            sizecode = self.__sizecode_gil1[:]["aqlcode"][test.argmax()]
        else:
            sizecode = "N"
        print("sizecode=" + sizecode + ", (lotsize=" + str(self.lotsize) + ")")
        return sizecode

    def return_aqlcode(self):
        # DETERMINE AQL CODE FOR LOT
        print("determine AQL code for batch....")
        sizecode = self.return_sizecode_gil1()

        nrnormapples, nrbadapples = Tfmodel(self.modelpath).predictbatch(
            self.applesamplepath
        )

        temp = self.__aqlcode_gil1.loc[sizecode] >= nrbadapples
        if not any(temp):
            aqlcode = "KL4"
        else:
            aqlcode = temp.idxmax()[0]
        print("AQL code =" + aqlcode)

        return aqlcode


if __name__ == "__main__":

    inputfile = r"aqltool.inp"
    modelname = "model_mobnet_kevin_v0"
    applesamplepath = r"raw\Test"

    if False:
        pred = Tfmodel(modelname).predictbatch(applesamplepath)
        nrcorrectpred = Tfmodel(modelname).predict2truth(applesamplepath)

    if True:
        inputfile = r"aqltool.inp"
        aqlt = Aqltool()
        aqlt.return_sizecode_gil1()
        aqlt.return_aqlcode()
