from tensorflow import keras
from keras import layers
from keras.callbacks import CSVLogger


# import inspect
import os
import sys


class Modelbuilder_v0:
    def __init__(self, verbose=False):
        print("initialize")
        self.verbose = verbose

    def model_kevin(self):
        """
        Model die geinspireerd is op artikel
        """
        model = keras.models.Sequential()
        model.add(layers.Input(shape=(224, 224, 3)))
        model.add(layers.Conv2D(128, (3, 3), activation="relu"))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation="relu"))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation="relu"))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation="relu"))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(32, (3, 3), activation="relu"))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dropout(0.1))
        model.add(layers.Dense(256, activation="relu"))
        model.add(layers.Dense(4))

        if self.verbose:
            model.summary()
        return model

    def model_kevin_alt(self):
        # xx
        model = keras.models.Sequential(
            [
                layers.Input(shape=(224, 224, 3)),
                layers.Rescaling(1.0 / 255, offset=0.0),
                # layers.Conv2D(
                #    128, (3, 3), input_shape=(224, 224, 3), activation="relu"
                # ),
                layers.Conv2D(128, (3, 3), activation="relu"),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation="relu"),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation="relu"),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation="relu"),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(32, (3, 3), activation="relu"),
                layers.MaxPooling2D((2, 2)),
                layers.Flatten(),
                layers.Dropout(0.1),
                layers.Dense(256, activation="relu"),
                layers.Dense(4, activation="softmax"),
            ],
            name="modelkevinalt",
        )
        if self.verbose:
            model.summary()
        return model

    def modkevalt_da(self):
        model = keras.models.Sequential(
            [
                layers.RandomFlip("horizontal_and_vertical"),
                layers.Input(shape=(224, 224, 3)),
                layers.Rescaling(1.0 / 255, offset=0.0),
                # layers.Conv2D(
                #    128, (3, 3), input_shape=(224, 224, 3), activation="relu"
                # ),
                layers.Conv2D(128, (3, 3), activation="relu"),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation="relu"),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation="relu"),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation="relu"),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(32, (3, 3), activation="relu"),
                layers.MaxPooling2D((2, 2)),
                layers.Flatten(),
                layers.Dropout(0.1),
                layers.Dense(256, activation="relu"),
                layers.Dense(4, activation="softmax"),
            ],
            name="modelkevinalt_da",
        )

        # model = keras.Model(model, name="grr")
        if self.verbose:
            model.summary()
        # model.name = "gtt"

        return model

    def tflearn_mobnet(self):
        mobnetmodel = keras.applications.MobileNetV2(
            input_shape=(224, 224, 3), include_top=False
        )
        mobnetmodel.trainable = False
        inputs = keras.Input(shape=(224, 224, 3))
        inputs = layers.Rescaling(1.0 / 255, offset=0.0)(inputs)
        x = mobnetmodel(inputs, training=False)
        x = keras.layers.GlobalMaxPooling2D()(x)
        outputs = keras.layers.Dense(4)(x)
        model = keras.Model(inputs, outputs)
        return model


class Modeltrainer_V0:
    def __init__(
        self,
        model,
    ):
        self.model = model
        sys.path.append("../data")
        import make_dataset as md

        self.make_dataset = md.make_predvalds

    """
    def make_dataset(self, pathtrainingdata):
        train_ds = keras.utils.image_dataset_from_directory(
            pathtrainingdata,
            validation_split=0.2,
            labels="inferred",
            subset="training",
            seed=123,
            label_mode="categorical",
            batch_size=32,  # defaultt: batch_size=32,
            image_size=(224, 224),
        )
        val_ds = keras.utils.image_dataset_from_directory(
            pathtrainingdata,
            validation_split=0.2,
            labels="inferred",
            subset="validation",
            seed=123,
            label_mode="categorical",
            batch_size=32,  # defaultt: batch_size=32,
            image_size=(224, 224),
        )
        return train_ds, val_ds  # , validation_ds
        # val_ds = layers.utils.image_dataset_from_directory(self.pathdata)
    """

    def _csvlog(self, csvpath, trainsetid):
        projectid = "project2"
        modelid = self.model.name
        csvlogfile = os.path.join(
            csvpath, projectid + "_" + modelid + "_" + trainsetid + ".log"
        )
        print(csvlogfile)
        csv_logger = CSVLogger(csvlogfile, separator=",", append=False)
        return csv_logger

    def _return_callbacklist(self, csvcb, earlycb, csvpath, trainsetid):
        cblist = []

        if earlycb == True:
            cearlys = keras.callbacks.EarlyStopping(
                monitor="val_loss",
                min_delta=0.02,
                patience=2,
                verbose=0,
                mode="auto",
                baseline=None,
                restore_best_weights=False,
            )
            cblist.append(cearlys)
            # cearlys = None
        if csvcb == True:
            csv_logger = self._csvlog(csvpath, trainsetid)
            cblist.append(csv_logger)
        return cblist

    def trainmodel(
        self, train_ds, val_ds=None, csvlog=True, earlystop=False, trainsetid=""
    ):
        # lossFunction = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        print(earlystop)
        csvpath = "C:/temp"
        lossFunction = keras.losses.CategoricalCrossentropy(from_logits=True)
        gdAlgorithm = keras.optimizers.Adam(learning_rate=0.001)
        nrOfEpochs = 15
        metrics = [
            "accuracy",
            # "categoricalcrossetropy",
            # keras.metrics.CategoricalAccuracy(),
        ]
        print(earlystop)
        cblist = self._return_callbacklist(csvlog, earlystop, csvpath, trainsetid)
        # if earlystop == False:
        #     cearlys = None
        # else:
        #     cearlys = keras.callbacks.EarlyStopping(
        #         monitor="val_loss",
        #         min_delta=0.02,
        #         patience=2,
        #         verbose=0,
        #         mode="auto",
        #         baseline=None,
        #         restore_best_weights=False,
        #     )
        # from keras.callbacks import CSVLogger
        # if csvlog == True:
        #     csv_logger = self._csvlog(csvpath, trainsetid)

        # csvlogfile = os.path.join(r"C:\temp", "project2_model0.log")
        # csv_logger = CSVLogger(csvlogfile, separator=",", append=False)
        # self.model.compile(optimizer=gdAlgorithm, loss=lossFunction, metrics="accuracy")
        self.model.compile(optimizer=gdAlgorithm, loss=lossFunction, metrics=[metrics])
        history = self.model.fit(
            train_ds,  # trainSet,
            # trainLabels,
            validation_data=val_ds,
            epochs=nrOfEpochs,
            batch_size=32,
            verbose=2,
            # callbacks=[cearlys, csv_logger],
            callbacks=cblist,
        )
        return model, history

    def exportmodelconfig(self, trainsetid, expfile):
        g = open(expfile, "w")
        g.write(self.model.name + ",")
        g.write(trainsetid + ",")
        g.close()

    def savemodel(self, trainsetid):
        modelid = model.name
        modelpath = r"C:\Users\marcr\MakeAIWork2\projects\maiw_project2\models"
        expfile = os.path.join(modelpath, modelid + "_" + trainsetid)
        self.model.save(expfile)


if __name__ == "__main__":
    if False:
        # model 1: Basismodel op basis van
        trainsetid = "basis"
        model = Modelbuilder_v0().model_kevin_alt()
        # datasets = Modeltrainer_V0(model).make_dataset(
        #    r"C:\Users\marcr\MakeAIWork2\projects\maiw_project2\data\raw\Train"
        # )
        model, history = Modeltrainer_V0(model).trainmodel(
            datasets[0], datasets[1], earlystop=True, trainsetid=trainsetid
        )
        expconfigfile = "C:/temp/p2_config.csv"
        Modeltrainer_V0(model).exportmodelconfig(trainsetid, expconfigfile)
        Modeltrainer_V0(model).savemodel(trainsetid)
        print("nono")
    if False:
        model = Modelbuilder_v0().modkevalt_da()
        trainsetid = "augment_v0"
        datasets = Modeltrainer_V0(model).make_dataset(
            r"C:\Users\marcr\MakeAIWork2\projects\maiw_project2\data\raw\Train"
        )
        print(datasets)

        model, history = Modeltrainer_V0(model).trainmodel(
            datasets[0], datasets[1], earlystop=True, trainsetid=trainsetid
        )
        expconfigfile = "C:/temp/p2_da_config.csv"
        Modeltrainer_V0(model).exportmodelconfig(trainsetid, expconfigfile)
        Modeltrainer_V0(model).savemodel(trainsetid)
        print("nono")
    if False:
        model = Modelbuilder_v0().model_kevin_alt()
        trainsetid = "augment_v1"
        datasets = Modeltrainer_V0(model).make_dataset(
            r"C:\Users\marcr\MakeAIWork2\projects\maiw_project2\data\processed\Train"
        )
        print(datasets)

        model, history = Modeltrainer_V0(model).trainmodel(
            datasets[0], datasets[1], earlystop=True, trainsetid=trainsetid
        )
        expconfigfile = "C:/temp/p2_da_v1_config.csv"
        Modeltrainer_V0(model).exportmodelconfig(trainsetid, expconfigfile)
        Modeltrainer_V0(model).savemodel(trainsetid)
        print("nono")

    if True:
        trainsetid = "mobnet_v0"
        model = Modelbuilder_v0().tflearn_mobnet()
        datasets = Modeltrainer_V0(model).make_dataset(
            r"C:\Users\marcr\MakeAIWork2\projects\maiw_project2\data\raw\Train"
        )
        print(datasets)

        model, history = Modeltrainer_V0(model).trainmodel(
            datasets[0], datasets[1], earlystop=False, trainsetid=trainsetid
        )
        expconfigfile = "C:/temp/p2_mobnet_config.csv"
        Modeltrainer_V0(model).exportmodelconfig(trainsetid, expconfigfile)
        Modeltrainer_V0(model).savemodel(trainsetid)
        print("nono")
