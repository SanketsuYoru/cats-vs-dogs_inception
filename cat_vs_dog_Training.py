# Helper libraries
import os
import random
import numpy as np
from shutil import copyfile
import matplotlib.pyplot as plt
# from lr_utils import load_dataset
import matplotlib.image  as mpimg
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3

#Init training folder and test folder
print(tf.__version__)
current_path = os.path.dirname(__file__)

#Split data trian: test= 9:1
def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):

    files = []
    for filename in os.listdir(SOURCE):
        file = SOURCE + filename
        if os.path.getsize(file) > 0:
            files.append(filename)
        else:
            print(filename + "'s length =0, skip...")

    training_length = int(len(files) * SPLIT_SIZE)
    shuffled_set = random.sample(files, len(files))
    training_set = shuffled_set[:training_length]
    testing_set = shuffled_set[training_length:]

    for filename in training_set:
        this_file = SOURCE + filename
        destination = TRAINING + filename
        copyfile(this_file, destination)

    for filename in testing_set:
        this_file = SOURCE + filename
        destination = TESTING + filename
        copyfile(this_file, destination)

if not os.path.exists(current_path+"/Saved-Weight/cp.ckpt.index"):
    try:
        os.mkdir(current_path+'/Saved-Weight')
        os.mkdir(current_path+'/cats-v-dogs')
        os.mkdir(current_path+'/cats-v-dogs/training')
        os.mkdir(current_path+'/cats-v-dogs/testing')
        os.mkdir(current_path+'/cats-v-dogs/training/cats')
        os.mkdir(current_path+'/cats-v-dogs/training/dogs')
        os.mkdir(current_path+'/cats-v-dogs/testing/cats')
        os.mkdir(current_path+'/cats-v-dogs/testing/dogs')
        print("mkdir done")
        CAT_SOURCE_DIR = current_path+"/PetImages/Cat/"
        TRAINING_CATS_DIR = current_path+"/cats-v-dogs/training/cats/"
        TESTING_CATS_DIR = current_path+"/cats-v-dogs/testing/cats/"
        DOG_SOURCE_DIR = current_path+"/PetImages/Dog/"
        TRAINING_DOGS_DIR = current_path+"/cats-v-dogs/training/dogs/"
        TESTING_DOGS_DIR = current_path+"/cats-v-dogs/testing/dogs/"

        split_size = .9
        split_data(CAT_SOURCE_DIR, TRAINING_CATS_DIR, TESTING_CATS_DIR, split_size)
        split_data(DOG_SOURCE_DIR, TRAINING_DOGS_DIR, TESTING_DOGS_DIR, split_size)
        print("split data done")

    except OSError:
        print(OSError.message)
else:
    print("data is already classified")


#Data generator
TRAINING_DIR = current_path+"/cats-v-dogs/training/"
train_datagen = ImageDataGenerator(
        rescale=1/255.0,
        horizontal_flip=True,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        fill_mode='nearest'
)
train_generator = train_datagen.flow_from_directory(
          TRAINING_DIR,
          class_mode='binary',
          batch_size=50,
          target_size=(150,150)
        )

VALIDATION_DIR = current_path+"/cats-v-dogs/testing/"
validation_datagen = ImageDataGenerator(rescale=1/255.)
validation_generator = validation_datagen.flow_from_directory(
          VALIDATION_DIR,
          batch_size=50,
          class_mode='binary',
          target_size=(150,150)
        )

#Init model
#Create check point call back function
checkpoint_path = current_path+"/Saved-Weight/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)



model=InceptionV3(input_shape=(150,150,3),
weights=None,
classes=1,
classifier_activation='sigmoid',
include_top=True)
if os.path.exists(current_path+"/Saved-Weight/cp.ckpt.index"):
    print('weight loded')
    model.load_weights(checkpoint_path)
model.summary()
model.compile(optimizer='Adam',
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])
history=model.fit(train_generator,epochs=100,steps_per_epoch=500,validation_data=validation_generator, validation_steps=50,callbacks=[cp_callback])

# #graph
acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(len(acc)) # Get number of epochs

#------------------------------------------------
# Plot training and validation accuracy per epoch
#------------------------------------------------
fig=plt.figure('Training and validation')  
sub_1 = fig.add_subplot(1,2,1)
sub_1.plot(epochs, acc,'y',label='training')
sub_1.plot(epochs, val_acc, 'b',label='validation')
sub_1.set_title('Accuracy')
sub_1.legend()
#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------
sub_2 = fig.add_subplot(1,2,2)
sub_2.plot(epochs, loss,'y',label='training')
sub_2.plot(epochs, val_loss, 'b',label='validation')
sub_2.set_title('Loss')
sub_2.legend() 
plt.show()