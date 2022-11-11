# -*- coding: utf-8 -*-
#import click
import logging
from pathlib import Path
import os

# from dotenv import find_dotenv, load_dotenv

from PIL import Image
import tensorflow as tf
import numpy as np
import random

pathdata = r"projects\maiw_project2\data"
pathraw = os.path.join(pathdata, "raw")
pathrawtrain = os.path.join(pathraw, "Train")
pathprocessed = os.path.join(pathdata, "processed")

try:
    pathprocessedtest = os.mkdir(os.path.join(pathprocessed, "Test"))
    pathprocesseduse = os.mkdir(os.path.join(pathprocessed, "Use"))
    pathprocessedtrain = os.mkdir(os.path.join(pathprocessed, "Train"))
except:
    pathprocessedtest = os.path.join(pathprocessed, "Test")
    pathprocesseduse = os.path.join(pathprocessed, "Use")
    pathprocessedtrain = os.path.join(pathprocessed, "Train")

appletypes = ["Blotch_Apple", "Normal_Apple", "Rot_Apple", "Scab_Apple"]

for var in appletypes:
    try:
        os.mkdir(os.path.join(pathprocessedtrain, var))
    except FileExistsError:
        pass


#@click.command()
#@click.argument("input_filepath", type=click.Path(exists=True))
#@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")


def _importimage(pafi):
    # import image file
    print(pafi)
    img = Image.open(pafi)
    arr = np.asarray(img).astype("uint8")
    return arr


def _exportimage(tf, fi, pathout, f_ext):
    # export numpy array to image
    def filemaker(pathout, f_ext):
        # fi = pathout.split("\\")[-1]
        fiout = fi.replace(".jpg", "_" + f_ext + ".jpg")
        pafiout = os.path.join(pathout, fiout)
        return pafiout

    arr = np.array(tf)
    img = Image.fromarray(arr)
    pafiout = filemaker(pathout, f_ext)
    print(pafiout)
    try:
        img.save(pafiout)
    except OSError:
        # happpens for example when image is in RGBA format
        rgb_im = img.convert("RGB")
        rgb_im.save(pafiout)


def imagezoom(pafi, pathout=""):
    #
    # zoom 50% picture and export
    arr = _importimage(pafi)
    try:
        tf_f = tf.image.central_crop(arr, 0.5)
    except:
        return
    fi = pafi.split("\\")[-1]
    _exportimage(tf_f, fi, pathout, "zo")


def imagerot90(pafi, pathout=""):
    # rotate picture random multiple of 90 degrees and export
    arr = _importimage(pafi)
    k = random.randint(1, 3)
    try:
        tf_f = tf.image.rot90(arr, k)
    except:
        return
    fi = pafi.split("\\")[-1]
    _exportimage(tf_f, fi, pathout, "rot90")


def imageflip(pafi, pathout=""):
    # flip left-right picture and export
    arr = _importimage(pafi)
    try:
        tf_f = tf.image.flip_left_right(arr)
    except:
        return
    fi = pafi.split("\\")[-1]
    _exportimage(tf_f, fi, pathout, "flr")


def batch_fliplr(pathin, pathout):
    # flip left right for batch of pictures
    files = os.listdir(pathin)
    for fi in files:
        pafi = os.path.join(pathin, fi)
        imageflip(pafi, pathout)


def batch_zoom(pathin, pathout):
    # zoom 50% for batch of pictures
    files = os.listdir(pathin)
    for fi in files:
        pafi = os.path.join(pathin, fi)
        imagezoom(pafi, pathout)


def batch_rot90(pathin, pathout):
    # rotate random multiple of 90 degrees for batch of pictures
    files = os.listdir(pathin)
    for fi in files:
        pafi = os.path.join(pathin, fi)
        imagerot90(pafi, pathout)


def make_predds(datapath):
    """
    make test dataset
    """
    print(datapath)
    predict_ds = tf.keras.utils.image_dataset_from_directory(
        datapath,
        labels="inferred",
        label_mode="categorical",
        batch_size=1,  # defaultt: batch_size=32,
        # batch_size=None,  # defaultt: batch_size=32,
        image_size=(224, 224),
        seed=123,
        shuffle=False,
    )
    return predict_ds
    # labels = pm.labelsfrompred(predict_ds)


def make_predvalds(pathtrainingdata):
    """
    make train en evaluation set

    """
    train_ds = tf.keras.utils.image_dataset_from_directory(
        pathtrainingdata,
        validation_split=0.2,
        labels="inferred",
        subset="training",
        seed=123,
        label_mode="categorical",
        batch_size=32,  # defaultt: batch_size=32,
        image_size=(224, 224),
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
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


def labelsfromds(dataset):
    # Rip of truth labels from dataset

    # haal uit een dataset de verzameling op.
    # for a, b in predict_ds:
    #    print(b)

    labels = [b.numpy().squeeze().argmax() for a, b in dataset]
    return labels


if __name__ == "__main__":
    """
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    # load_dotenv(find_dotenv())

    # main()
    """
    for var in appletypes:
        pathin = os.path.join(pathrawtrain, var)
        pathout = os.path.join(pathprocessedtrain, var)

        batch_fliplr(pathin, pathout)
        batch_zoom(pathin, pathout)
        batch_rot90(pathin, pathout)


"""
def confmatrix_anton():
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import ConfusionMatrixDisplay

    batchPredictions = model.predict(test_ds)
    predicted_categories = tf.argmax(batchPredictions, axis=1)
    true_categories = tf.concat([y for x, y in test_ds], axis=0)
    # confusion_matrix(predicted_categories, true_categories)
    confusion_matrix = confusion_matrix(true_categories, predicted_categories)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
    cm_display.plot()
    plt.show()
"""
