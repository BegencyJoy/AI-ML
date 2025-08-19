

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras .models import Sequential
from tensorflow.keras .layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout,BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import roc_curve
from google.colab import drive

from warnings import filterwarnings
filterwarnings('ignore')

from google.colab import drive
drive.mount('/content/drive')

train_datagen=ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    rotation_range=20,
    horizontal_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2

)
train_datagen=ImageDataGenerator(rescale=1./255)

train_set=train_datagen.flow_from_directory('/content/drive/MyDrive/CNN2/Train',target_size=(64,64),batch_size=32,class_mode='binary')

test_set=train_datagen.flow_from_directory('/content/drive/MyDrive/CNN2/Test',target_size=(64,64),batch_size=32,class_mode='binary',shuffle=False)

model=Sequential([
    Conv2D(32,(3,3),activation='relu',input_shape=(64,64,3)),
    MaxPooling2D(2,2),
    Conv2D(32,(3,3),activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(32,(3,3),activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(512,activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(256,activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(1,activation='sigmoid')
])

model.summary()

tf.keras.utils.plot_model(
    model,
    show_shapes=True,
    show_dtype=True,
    show_layer_activations=True
)

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

early_stopping=tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

history=model.fit(
    train_set,
    epochs=20,
    validation_data=test_set,
    callbacks=[early_stopping]
)

history_df=pd.DataFrame(history.history)
history_df[['loss','val_loss']].plot(title='loss vs val_loss')
history_df[['accuracy','val_accuracy']].plot(title='accuracy vs val_accuracy')
plt.show()

y_true=test_set.classes
y_pred=model.predict(test_set)
fpr,tpr,threshold=roc_curve(y_true,y_pred)
optimal_idx=np.argmax(tpr-fpr)
optimal_threshold=threshold[optimal_idx]
print("Optimal threshold:",optimal_threshold)

from tensorflow.keras.utils import load_img,img_to_array

def classify_image(image_path,threshold=0.5):
  test_image=load_img(image_path,target_size=(64,64))
  plt.imshow(test_image)
  test_image=img_to_array(test_image)/255.0
  test_image=np.expand_dims(test_image,axis=0)
  result=model.predict(test_image)
  print("Result array:",result)
  if result >= threshold:
    print("withoutmask")
  else:
    print("withmask")

classify_image('/content/without_mask_3821.jpg',threshold=optimal_threshold)

classify_image('/content/with_mask_3725.jpg',threshold=optimal_threshold)