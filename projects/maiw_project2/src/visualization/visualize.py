from matplotlib import pyplot as plt
from tensorflow import math as tfmath
import os
import numpy as np
import pandas as pd


def confusionmatrix(predictions, labels, clsnames, filename):
    # Create visual plot of confusionmatrix
    # invoer
    #       predictions: predictions from set of pictures with  model (model.prediction())
    #       labels: truth labels in accordance with set of pictures with  original d  [numpy array]
    #       clsnames: [list]
    figpath = "C:/Users/marcr/MakeAIWork2/projects/maiw_project2/models"
    # print(labels)    # print(predictions)

    # berekenen confusion matrix
    print(len(predictions))
    print(len(labels))
    confmat = tfmath.confusion_matrix(predictions, labels)

    # Plotten resultaat
    plt.matshow(confmat)
    for (j, i), label in np.ndenumerate(confmat):
        plt.gca().text(i, j, label, ha="center", va="center")
        # ax2.text(i, j, label, ha="center", va="center")
    plt.ylabel("PREDICTION", fontsize=20)
    plt.xlabel("TRUTH", fontsize=20)
    plt.xticks(range(4), labels=clsnames, fontsize=14)
    plt.yticks(range(4), labels=clsnames, fontsize=14)
    plt.gcf().set_figwidth(16)
    plt.gcf().set_figheight(12)
    plt.gcf().savefig(os.path.join(figpath, filename))
    plt.close()
    return confmat


def modelaccuracy(logfile):
    # make plot of accuracy at different epochs.

    df = pd.read_csv(logfile)
    print(df)

    fig = plt.figure()
    plt.plot(df["epoch"], df["accuracy"], label="accuracy")
    plt.plot(df["epoch"], df["val_accuracy"], label="val_accuracy")

    plt.legend()
    # plt.show()
    # fig = plt.gcf()
    fig.savefig(logfile.replace("log", "png"))


if __name__ == "__main__":
    if False:
        # modelaccuracy
        modelaccuracy("C:/temp/project2_modelkevinalt_basis.log")
        modelaccuracy("C:/temp/project2_modelkevinalt_da_augment_v0.log")
        modelaccuracy("C:/temp/project2_modelkevinalt_augment_v1.log")

    if True:
        # model confusion matrix
        testdatapath = r"projects\maiw_project2\data\raw\Test"

        # onderstaande eigenlijk met een setup doen?

        os.path.append(r"projects\maiw_project2\src\data")
        import make_dataset as md

        predict_ds = md.make_predds(testdatapath)

        modelpath = r"C:\Users\marcr\MakeAIWork2\projects\maiw_project2\models"
        if True:
            modellen = [
                # "modelkevinalt_basis",
                # "modelkevinalt_da_augment_v0",
                # "modelkevinalt_augment_v1",
                "model_mobnet_v0",
            ]

            # labels uit dataset
            labels = md.labelsfromds(predict_ds)

            os.path.append(r"projects\maiw_project2\src\models")
            import predict_model as pm

            for modelname in modellen:
                # modelname = "modelkevinalt_basis"
                # prediction maken op basis van model
                predictions, clsnames = pm.predictfromdata(modelname, testdatapath)
                classprediction = pm.classfromprediction(predictions)

                confmat = confusionmatrix(
                    classprediction, labels, clsnames, "cf_" + modelname + ".png"
                )
