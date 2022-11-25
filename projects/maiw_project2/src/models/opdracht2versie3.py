# Import van libaries

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
from PIL import Image
import random

# Use is the real unlabeled data
# Use 1 contains random chosen 80 unlabeled apples
# Use 2 contains 77 healty apples and three other

data_dir = 'Train'

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir, validation_split=0.2, subset='training', seed=123, image_size=(224, 224), batch_size=120)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir, validation_split=0.2, subset='validation', seed=123, image_size=(224, 224), batch_size=120)

#apple_dir = 'Use'
#apple_dir = 'Use1'
apple_dir = 'Use2'
#apple_dir = 'Use3'

apple_ds = tf.keras.utils.image_dataset_from_directory(
    apple_dir, image_size=(224, 224), labels = None)

class_names = train_ds.class_names
num_classes = len(class_names)

# Preprocess
# The preprocess_input function is meant to adequate your image to the format the model requires
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

# Rescale
# A preprocessing layer which rescales input values to a new range.
tf.keras.layers.Rescaling(scale=1./255)

# Add augmentation
data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal_and_vertical"),
  layers.RandomRotation(0.2)])
"""
#apple_dir = 'Use'
#apple_dir = 'Use1'
apple_dir = 'Use2'


apple_ds = tf.keras.utils.image_dataset_from_directory(
    apple_dir, image_size=(224, 224), labels = None)

# class_names = train_ds.class_names
# num_classes = len(class_names)
"""
# Set image size
img_height = 224
img_width = 224
IMG_SIZE = img_width, img_height
IMG_SHAPE = IMG_SIZE + (3,)

# After testing and evaluation we decided to youse the pre-trained model. This model had the best performance and was still
# abble to perform fast on my labtop.

# Model van MobileNet
# A pre-trained model is a saved network that was previously trained on a large dataset, typically on a 
# large-scale image-classification task.

base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

base_model.summary()

base_model.trainable = False

# Feature extraction
# Aims to reduce the number of features in a dataset by creating new features from the existing ones

image_batch, label_batch = next(iter(train_ds))
feature_batch = base_model(image_batch)
print(feature_batch.shape)

# globalAveragePooling2d() function is used for applying global average pooling operation for spatial data

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)

prediction_layer = tf.keras.layers.Dense(4)
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape)

inputs = tf.keras.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = tf.keras.layers.Rescaling(scale=1./255)(inputs)
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = prediction_layer(x)

modelX = tf.keras.Model(inputs, outputs)

#print(modelX)

# Optimize
# Compile defines the loss function, the optimizer and the metrics
# Cross-entropy loss is used when adjusting model weights during training. The aim is to minimize the loss

base_learning_rate = 0.0001
modelX.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

modelX.summary()

len(modelX.trainable_variables)

# Standard loss and accuracy before training of the model with 25 training run's


initial_epochs = 30

loss0, accuracy0 = modelX.evaluate(val_ds)

print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))

# Start training the model.

history = modelX.fit(train_ds,
                    epochs=initial_epochs,
                    validation_data=val_ds)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,2.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

# Unfreeze the frosen part of the MobileNet.

base_model.trainable = True

# Let's take a look to see how many layers are in the base model
print("Number of layers in the base model: ", len(base_model.layers))

# Fine-tune from this layer onwards
fine_tune_at = 100

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
  layer.trainable = False

# Optimize
# Compile defines the loss function, the optimizer and the metrics
# Cross-entropy loss is used when adjusting model weights during training. The aim is to minimize the loss

modelX.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer = tf.keras.optimizers.RMSprop(learning_rate=base_learning_rate/10),
              metrics=['accuracy'])

modelX.summary()

#vanaf 100 dus 100 ook en 154 ook

len(modelX.trainable_variables)

# Continue training the model.

fine_tune_epochs = 8
total_epochs =  initial_epochs + fine_tune_epochs

history_fine = modelX.fit(train_ds,
                         epochs=total_epochs,
                         initial_epoch=history.epoch[-1],
                         validation_data=val_ds)

# show the values after the second training of the model in a diagram.

acc += history_fine.history['accuracy']
val_acc += history_fine.history['val_accuracy']

loss += history_fine.history['loss']
val_loss += history_fine.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylim([0.2, 0.95])
plt.plot([initial_epochs-1,initial_epochs-1],
          plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper left')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([0, 2.0])
plt.plot([initial_epochs-1,initial_epochs-1],
         plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

#Evaluate the model.

modelX.evaluate(train_ds)

# Make a Confusion matrix to show the labels and prediction

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

data_dir = 'test'

test_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir, seed=123, image_size=(224, 224),shuffle = False, batch_size=120)

batchPredictions = modelX.predict(test_ds)

predicted_categories = tf.argmax(batchPredictions, axis=1)

true_categories = tf.concat([y for x, y in test_ds], axis=0)

cm = confusion_matrix(true_categories, predicted_categories)

cmd = ConfusionMatrixDisplay(cm, display_labels=['blotch','normal','rot','scab'])

cmd.plot()

# Save the model.
modelX.save('janmodelpython')


batchPredictions = modelX.predict(apple_ds)

predicted_categories = tf.argmax(batchPredictions, axis=1)
print(predicted_categories)




# Load het getrainde model.

# modelX = tf.keras.models.load_model('janmodelpython')



# Make a list from the outcome and print them.

predicted_categories[0]
predicted_categories[0].numpy().tolist()
samplebatchnp = predicted_categories.numpy()
samplebatchlist = samplebatchnp.tolist()
#print(samplebatchlist)
print(samplebatchlist.count(0))
print(samplebatchlist.count(1))
print(samplebatchlist.count(2))
print(samplebatchlist.count(3))



# AQL tester.

blotchApple = samplebatchlist.count(0)
HealtyApple = samplebatchlist.count(1)
rotApple = samplebatchlist.count(2)
scabApple = samplebatchlist.count(3)

def AQLTestscore():

    for apple in samplebatchlist:
        
        if HealtyApple >=79:
            print (f'The batch has been quallified as: Class 1\n')
        elif HealtyApple >=75:
            print (f'The batch has been quallified as: Class 2\n')
        elif HealtyApple >=73:
            print (f'The batch has been quallified as: Class 3\n')
        else:
            print (f'The batch has been rejected\n')

        print (f'The total batch of {len(samplebatchlist)} apples consists of:\n'
        f'Healthy apples: {HealtyApple}\n'
        f'Blotch apples: {blotchApple}\n'
        f'Rot apples: {rotApple}\n'
        f'Scab apples: {scabApple}\n')
        break
    
AQLTestscore()


# Show the values in a wedge Diagram.
appleLabels = ['Normal', 'Blotch', 'Rot', 'Scab']
 
data = [HealtyApple, blotchApple, rotApple, scabApple]

# Creating explode data
explode = (0.0, 0.0, 0.0, 0.0) # not necessary

# Wedge properties
wp = { 'linewidth' : 0.5, 'edgecolor' : "white" }

# Creating autocpt arguments
def func(pct, allvalues):
	absolute = int(pct / 100.*np.sum(allvalues))
	return "{:.1f}%\n".format(pct, absolute)

# Creating plot
fig, ax = plt.subplots(figsize =(10, 7))
wedges, texts, autotexts = ax.pie(data,
								autopct = lambda pct: func(pct, data),
								explode = explode,
								labels = appleLabels,
								shadow = False,
								startangle = 15,
								wedgeprops = wp,
								textprops = dict(color ="#115C53"))

# Adding legend
ax.legend(wedges, appleLabels,
		title ="Apples",
		loc ="upper left",
		bbox_to_anchor =(1, 0, 0.5, 1))

plt.setp(autotexts, size = 8, weight ="bold")
ax.set_title("Samenstelling appelbatch") # add Classification as title

#show plot
plt.show() 

# SentenceTransformers is a Python framework for state-of-the-art sentence, text and image embeddings.
# Choosen model
# all-MiniLM-L12-v2 	68.70 Performance sentence.
#                   	50.82	Performance semantic.
#                     59.76	Average performance.
#                     7500	Speed.
#                     120 MB Modelsize.


from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L12-v2')


accuracyBatch = modelX.evaluate(apple_ds)

sentences = [f'There are {HealtyApple} healthy apples in this batch',
                                  f'The accuracy of this prediction is {accuracyBatch} for loss and accuracyS',
                                  f'This batch contains {rotApple} rotten apple(s)',
                                  f'This batch contains {scabApple} scabbed apple(s)',
                                  f'This batch contains {blotchApple} blotched apple(s)',
                                    'Class 1 are at least 79 healty apples for example supermarkets ore greengrocer',
                                    'Class 2 are at least 75 healty apples used for exapmle in applesause',
                                    'Class 3 are at least 73 healty apples used for syrup',
                                    'A rejected batch can be used for petfood',
                                    'We have 4 types of apples, Blotch, Normal, Rot and Scab',
                                    'A batch consists of 80 apples',
                                    'Am I so smart ore are you so dum',
                                    'AQL means Acceptance Quality limit',
                                    'The lot size is between 5.000 and 10.000 apples']


query_embedding = model.encode(input('what is your question?'), convert_to_tensor=True)
passage_embedding = model.encode(sentences, convert_to_tensor=True)
answerArray = util.dot_score(query_embedding, passage_embedding)

print("Similarity:", answerArray)

answerLocation = answerArray.argmax()

print(sentences[answerLocation.item()])








