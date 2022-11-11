data_dir = "Test"
import os


def batchselector_jan():
    img_height = 360
    img_width = 360
    batch_size = 32

    test_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir, image_size=(img_height, img_width), batch_size=batch_size
    )

    appleDirectory_blotch = "Test/Blotch_Apple"
    appleDirectory_normal = "Test/Normal_Apple"
    appleDirectory_rot = "Test/Rot_Apple/"
    appleDirectory_Scab = "Test/Scab_Apple/"
    appleDirectory = "Test"
    edgeFiles = list()

    directory_list = [
        appleDirectory_blotch,
        appleDirectory_normal,
        appleDirectory_rot,
        appleDirectory_Scab,
    ]

    for directory in directory_list:
        for filename in os.listdir(directory):
            imgFile = os.path.join(directory, filename)
            edgeFiles.append(imgFile)
    print(edgeFiles)

    aql_set = random.choices(edgeFiles, k=60)
    print(aql_set)


def batchselector():
    pass


if __name__ == "__main__":
    print("yes")
