import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.densenet import DenseNet121
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras import backend as K
import keras.utils
from keras.applications import densenet
from keras.models import load_model
import keras
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from keras.applications.densenet import DenseNet201
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import densenet
from keras.models import Sequential, Model, load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback
from keras import regularizers
from keras import backend as K
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
import csv
import matplotlib.image as mpimg
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from PIL import Image
from tensorflow.keras import models,layers
print(tf.__version__)


for dirname, _, filenames in os.walk(r'E:\ML-PROJECTS\Dataset\tomato dataset'):   #train dataset address
    for filename in filenames:
        print(os.path.join(dirname, filename))

tomato_bacterial_spot_dir=r'E:\ML-PROJECTS\Dataset\tomato dataset\Bacterial_spot227'  #address of all the folders inside the dataset 
tomato_early_blight_dir=r'E:\ML-PROJECTS\Dataset\tomato dataset\Early_blight227'
tomato_late_blight_dir=r'E:\ML-PROJECTS\Dataset\tomato dataset\Late_blight227'
tomato_leaf_mold_dir=r'E:\ML-PROJECTS\Dataset\tomato dataset\Leaf_Mold227'
tomato_septoria_leaf_spot_dir=r'E:\ML-PROJECTS\Dataset\tomato dataset\Septoria_leaf_spot227'
tomato_spider_mite_dir=r'E:\ML-PROJECTS\Dataset\tomato dataset\Two-spotted_spider_mite227'
tomato_target_spot_dir=r'E:\ML-PROJECTS\Dataset\tomato dataset\Target_Spot227'
tomato_yellow_leaf_curl_dir=r'E:\ML-PROJECTS\Dataset\tomato dataset\Tomato_Yellow_Leaf_Curl_Virus227'
tomato_mosaic_virus_dir=r'E:\ML-PROJECTS\Dataset\tomato dataset\Tomato_mosaic_virus227'
tomato_healthy_dir=r'E:\ML-PROJECTS\Dataset\tomato dataset\healthy227'

nrows=10
ncols=10
pic_index=0
fig=plt.gcf()
fig.set_size_inches(10,10)

tomato_bacterial_spot_names=os.listdir(tomato_bacterial_spot_dir)
tomato_early_blight_names=os.listdir(tomato_early_blight_dir)
tomato_late_blight_names=os.listdir(tomato_late_blight_dir)
tomato_leaf_mold_names=os.listdir(tomato_leaf_mold_dir)
tomato_septoria_leaf_spot_names=os.listdir(tomato_septoria_leaf_spot_dir)
tomato_spider_mite_names=os.listdir(tomato_spider_mite_dir)
tomato_target_spot_names=os.listdir(tomato_target_spot_dir)
tomato_yellow_leaf_curl_names=os.listdir(tomato_yellow_leaf_curl_dir)
tomato_mosaic_virus_names=os.listdir(tomato_mosaic_virus_dir)
tomato_healthy_names=os.listdir(tomato_healthy_dir)

pic_index += 10
next_tomato_bacterial_spot_pix = [os.path.join(tomato_bacterial_spot_dir, fname)
                for fname in tomato_bacterial_spot_names[pic_index-10:pic_index]]
next_tomato_early_blight_pix = [os.path.join(tomato_early_blight_dir, fname)
                for fname in tomato_early_blight_names[pic_index-10:pic_index]]
next_tomato_late_blight_pix = [os.path.join(tomato_late_blight_dir, fname)
                for fname in tomato_late_blight_names[pic_index-10:pic_index]]
next_tomato_leaf_mold_pix = [os.path.join(tomato_leaf_mold_dir, fname)
                for fname in tomato_leaf_mold_names[pic_index-10:pic_index]]
next_tomato_septoria_leaf_spot_pix = [os.path.join(tomato_septoria_leaf_spot_dir, fname)
                for fname in tomato_septoria_leaf_spot_names[pic_index-10:pic_index]]
next_tomato_spider_mite_pix = [os.path.join(tomato_spider_mite_dir, fname)
                for fname in tomato_spider_mite_names[pic_index-10:pic_index]]
next_tomato_target_spot_pix = [os.path.join(tomato_target_spot_dir, fname)
                for fname in tomato_target_spot_names[pic_index-10:pic_index]]
next_tomato_yellow_leaf_curl_pix = [os.path.join(tomato_yellow_leaf_curl_dir, fname)
                for fname in tomato_yellow_leaf_curl_names[pic_index-10:pic_index]]
next_tomato_mosaic_virus_pix = [os.path.join(tomato_mosaic_virus_dir, fname)
                for fname in tomato_mosaic_virus_names[pic_index-10:pic_index]]
next_tomato_healthy_pix = [os.path.join(tomato_healthy_dir, fname)
                for fname in tomato_healthy_names[pic_index-10:pic_index]]

for i, img_path in enumerate(
        next_tomato_bacterial_spot_pix + next_tomato_early_blight_pix + next_tomato_late_blight_pix + next_tomato_leaf_mold_pix + next_tomato_septoria_leaf_spot_pix + next_tomato_spider_mite_pix + next_tomato_target_spot_pix + next_tomato_yellow_leaf_curl_pix + next_tomato_mosaic_virus_pix + next_tomato_healthy_pix):
    # Set up subplot; subplot indices start at 1
    sp = plt.subplot(nrows, ncols, i + 1)

    sp.axis('Off')  # Don't show axes (or gridlines)

    img = mpimg.imread(img_path)
    plt.imshow(img)

plt.show()

labels=['bacterial_spot','early_blight','late_blight','leaf_mold','septoria_leaf_spot','spider_mite','target_spot','yellow_leaf_curl','mosaic_virus','healthy']

train_data_dir = r'E:\ML-PROJECTS\Dataset\tomato dataset'
validation_data_dir = r'E:\ML-PROJECTS\Dataset\tomato test set'

img_width, img_height = 224, 224
nb_train_samples = 8144
nb_validation_samples = 8041
epochs = 5
batch_size = 12
n_classes = 10

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    fill_mode = 'constant',
    cval = 1,
    rotation_range = 5,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(256,256),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(256,256),
    batch_size=batch_size,
    class_mode='categorical')

# input_shape = (32, 256, 256, 3)
n_classes = 10

# data_scaling = tf.keras.Sequential([
#   layers.experimental.preprocessing.Resizing(256, 256),
#   layers.experimental.preprocessing.Rescaling(1./255)
# ])
# data_augmentation = tf.keras.Sequential([
#   layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
#   layers.experimental.preprocessing.RandomRotation(0.2),
#   layers.experimental.preprocessing.RandomContrast(0.5),
#   tf.keras.layers.experimental.preprocessing.RandomZoom(0.2),
# #   tf.keras.layers.experimental.preprocessing.RandomHeight(0.2),
# #   tf.keras.layers.experimental.preprocessing.RandomWidth(0.2)

# ])

model = models.Sequential([
    layers.Conv2D(32, kernel_size = (3,3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64,  kernel_size = (3,3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64,  kernel_size = (3,3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(n_classes, activation='softmax'),
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)
model_history = model.fit(
    train_generator,
    batch_size=32,
    #steps_per_epoch=len(dataset_test)// batch_size,
    validation_data=validation_generator,
    verbose=1,
    epochs=1,
)
# model.build(input_shape=input_shape)
# def build_model():
#     base_model = densenet.DenseNet121(input_shape=(224, 224, 3),
#                                     #   weights=r'E:\ML-PROJECTS\DenseNet-BC-121-32.h5',
#                                       pooling='max')
#     for layer in base_model.layers:
#         layer.trainable = True

#     x = base_model.output
#     x = Dense(1000, kernel_regularizer=regularizers.l1_l2(0.01), activity_regularizer=regularizers.l2(0.01))(x)
#     x = Activation('relu')(x)
#     x = Dense(500, kernel_regularizer=regularizers.l1_l2(0.01), activity_regularizer=regularizers.l2(0.01))(x)
#     x = Activation('relu')(x)
#     predictions = Dense(n_classes, activation='softmax')(x)
#     model = Model(inputs=base_model.input, outputs=predictions)

#     return model

# model = build_model()
# model.compile(loss=keras.losses.CategoricalCrossentropy(), optimizer='adam', metrics=['acc', 'mse'])

# # early_stop = EarlyStopping(monitor='val_loss', patience=8, verbose=1, min_delta=1e-4)
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, verbose=1, min_delta=1e-4)
# callbacks_list = [early_stop, reduce_lr]

# model_history = model.fit(
#     train_generator,
#     steps_per_epoch=int(8000/batch_size),
#     epochs=epochs,
#     validation_data=validation_generator,
#     validation_steps=int(2000/batch_size)
#     )

model.save('tomatoTest.h5')
    
  
    

plt.figure(0)
plt.plot(model_history.history['accuracy'], 'r')
plt.plot(model_history.history['val_accuracy'], 'g')
plt.xticks(np.arange(0, 20, 1.0))
plt.rcParams['figure.figsize'] = (8, 6)
plt.xlabel("Num of Epochs")
plt.ylabel("Accuracy")
plt.title("Training Accuracy vs Validation Accuracy")
plt.legend(['train', 'validation'])

plt.figure(1)
plt.plot(model_history.history['loss'], 'r')
plt.plot(model_history.history['val_loss'], 'g')
plt.xticks(np.arange(0, 20, 1.0))
plt.rcParams['figure.figsize'] = (8, 6)
plt.xlabel("Num of Epochs")
plt.ylabel("Loss")
plt.title("Training Loss vs Validation Loss")
plt.legend(['train', 'validation'])
plt.show()

pred = model.predict_generator(validation_generator, steps=None, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=1)
predicted = np.argmax(pred, axis=1)

print('Confusion Matrix')
cm = confusion_matrix(validation_generator.classes, np.argmax(pred, axis=1))
plt.figure(figsize = (30,20))
sns.set(font_scale=1.4) #for label size
sns.heatmap(cm, annot=True, annot_kws={"size": 12}) # font size
plt.show()
print()
print('Classification Report')
print(classification_report(validation_generator.classes, predicted, target_names=labels))

# def predict_one(model):
#     image_batch, classes_batch = next(validation_generator)
#     predicted_batch = model.predict(image_batch)
#     for k in range(0,image_batch.shape[0]):
#       image = image_batch[k]
#       pred = predicted_batch[k]
#       the_pred = np.argmax(pred)
#       predicted = labels[the_pred]
#       val_pred = max(pred)
#       the_class = np.argmax(classes_batch[k])
#       value = labels[np.argmax(classes_batch[k])]
#       plt.figure(k)
#       isTrue = (the_pred == the_class)
#       plt.title(str(isTrue) + ' - class: ' + value + ' - ' + 'predicted: ' + predicted + '[' + str(val_pred) + ']')
#       plt.imshow(image)

# predict_one(model)

from sklearn.metrics import confusion_matrix
# TP = confusion[1,1] # true positive 
# TN = confusion[0,0] # true negatives
# FP = confusion[0,1] # false positives
# FN = confusion[1,0] # false negatives
num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = confusion_matrix(validation_generator.classes, np.argmax(pred, axis=1))
    total1=sum(sum(cm1))
    Accuracy = (cm1[0,0]+cm1[1,1])/total1
    Specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    Sensitivity = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    
print(Accuracy)
print("This is the specificity",Specificity)
print("This is the sensitivity",Sensitivity)