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
from tensorflow.keras.preprocessing import image

#Init training folder and test folder
print(tf.__version__)
current_path = os.path.dirname(__file__)

# path='C:\\Users\\I1661\\Desktop\\209.jpg'

#Init model and load weights
def initModel():
    checkpoint_path = current_path+"/Saved-Weight/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
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
    print('model successfully initialized...\n')
    return model,True


def pridect(path):
    img = image.load_img(path, target_size=(150,150))
    x = image.img_to_array(img)/255.0
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    result=model.predict(images,batch_size=1)
    print(result)
    if result[0]>0.5:
        print(path +" "+str((float(result[0]))*100)+ " %"+" is a dog\n")
    else:
        print(path +" "+str((1-float(result[0]))*100)+ " %"+" is a cat\n")


model,iscompleted=initModel()

print('please input image path , input exit to finish')
while(True and iscompleted):
    path=input()
    if (path!='exit'):
        pridect(path)
    else:
        break

