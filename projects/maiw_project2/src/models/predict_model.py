import tensorflow as tf
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
from PIL import Image




sourcepad = "projects\maiw_project2\src"
sys.path.extend([os.path.join(sourcepad, "models"), os.path.join(sourcepad, "data")])

datapath = r"projects\maiw_project2\data"
modelpath = r"projects\maiw_project2\models"

import make_dataset as md


def evaluatefromdata(modelname, datapath):
    # EVALUATE MODEL WITH DATA IN DATAPATH
    model = tf.keras.models.load_model(os.path.join(modelpath, modelname))
    eval_ds = md.make_predds(datapath)
    result = model.evaluate(eval_ds, verbose=False)
    return result


def predictfromdata(modelname, datapad):
    # MAKE PREDICTION OF A BATCH OF PICTURES
    # Input:
    #   modelname: name of model [str]
    #   datapad: path to data[str]
    # Output:
    #   predresult: result of prediction [......]
    #   clnames:    classnames [list]

    model = tf.keras.models.load_model(os.path.join(modelpath, modelname))
    predict_ds = md.make_predds(os.path.join(datapath, datapad))

    predresult = model.predict(predict_ds, verbose=False)
    clnames = predict_ds.class_names
    return predresult, clnames


def truthfromdata(datapad):
    # RETURN TRUTH LABEL FROM DATA IN DATAPAD
    predict_ds = md.make_predds(os.path.join(datapath, datapad))
    truth = [a[1] for a in predict_ds.as_numpy_iterator()]
    return truth


def classfromprediction(predictions):
    # DETERMINE MOST LIKELY CLASS FOR A SET OF PREDICTIONS
    # Input:
    #   predictions: predictions from model.predict
    # Output:
    #   clprediction [list ]

    clprediction = [pred.argmax() for pred in predictions]
    return clprediction


def pred2truth(datapad, modelname):
    """
    COMPARE PREDICTION WITH TRUTH LABEL
    Input:
        datapad: path to images [str]
        modelname: name of ML model [str]
    Output:
        endres = per apple corre
        nrbadapples = number of not normal apples [int]
    """
    truth = truthfromdata(datapad)
    predictions, clsnames = predictfromdata(modelname, datapad)

    zippie = zip(predictions, truth)
    compare = []
    for a in zippie:
        compare.append([a[0].argmax(), a[1].argmax()])
    comparearray = np.array(compare)

    endres = comparearray[:, 0] == comparearray[:, 1]
    nrcorrectpred = endres.sum()
    print("Right predicted: ", str(endres.sum()), "on " + str(len(endres)))
    return endres, nrcorrectpred


if __name__ == "__main__":
    modelname = "modelkevinalt_augment_v1"
    if True:
        result = evaluatefromdata(modelname, datapath)
        print("accuracy van testset:" + str(result[1]))

    pres, clsname = predictfromdata(modelname, datapath)

    predict_ds = md.make_predds(datapath)
    truth = [a[1] for a in predict_ds.as_numpy_iterator()]

    zippie = zip(pres, truth)
    compare = []
    for a in zippie:
        compare.append([a[0].argmax(), a[1].argmax()])
    comparearray = np.array(compare)
    endres = comparearray[:, 0] == comparearray[:, 1]
    print("Right predicted: ", str(endres.sum()), "on " + str(len(endres)))
