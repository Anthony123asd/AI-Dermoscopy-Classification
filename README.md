# AI-Dermoscopy-Classification
Using Machine Learning to classify images into which type of Breast Cancer 

This project is submitted as my final project in Digital Talent Scholarship 2019 - Artificial Intelligence Program.

For working on this project, I use Jupyter Notebook as the Application to run the python code.

Download the images here : PH2_dermoscopy.zip (http://bit.ly/2W0NTVk)

The condition given for the projects are:
  - There are 600 images of breast cancer with 3 types (Common Nevus, Atypical Nevus, and Melanoma)
  - The images' formats are BMP and PPM
  - The tree folders are like the picture below for the BMP files and PPM files. Our .ipynb and .py files are in PH2_dermoscopy folder

![image](https://user-images.githubusercontent.com/18510738/63907955-96066100-ca47-11e9-8eff-c437ad9f7335.png)

![image](https://user-images.githubusercontent.com/18510738/63908035-d4038500-ca47-11e9-9589-29591a512a8b.png)

## Preparing the Data

First, we have to know which picture belongs to which type of Breast Cancer. Luckily, we have the dataset in the PH2Dataset.csv. So we just have to import using pandas dataframe
```
import pandas as pd
df = pd.read_csv('PH2Dataset.csv').drop(columns='Histological Diagnosis')
```
The code below will give us Dataframes of Image Name and Types of the breast cancer. I use it to 
```
df_CN = df[['Image Name','Common Nevus']].dropna()
df_AN = df[['Image Name','Atypical Nevus']].dropna()
df_M = df[['Image Name','Melanoma']].dropna()
#df.head()
df_CN_list = []
df_AN_list = []
df_M_list  = []
for i in df_CN['Image Name']:
    df_CN_list.append(i)
for i in df_AN['Image Name']:
    df_AN_list.append(i)
for i in df_M['Image Name']:
    df_M_list.append(i)
```

Next we copy the images data in the PH2_Dataset folder into one folder and classify them into several folders using their type so we could use it easily.

```
new_folder = 'original_dataset'
!mkdir $new_folder

!mkdir $new_folder\Melanoma
!mkdir $new_folder\NonMelanoma


path = 'PH2Dataset\PH2_Dataset_images'
for filename in os.listdir(path):
    if filename in df_CN_list:
        for fileImages in os.listdir(path+'/'+filename+'/'+filename+"_Dermoscopic_Image"):
            image_path = (path+'\\'+filename+'\\'+filename+'_Dermoscopic_Image\\'+fileImages)
            !copy $image_path $new_folder\\Common_Nevus
    if filename in df_AN_list:
        for fileImages in os.listdir(path+'/'+filename+'/'+filename+"_Dermoscopic_Image"):
            image_path = (path+'\\'+filename+'\\'+filename+'_Dermoscopic_Image\\'+fileImages)
            !copy $image_path $new_folder\\Atypical_Nevus
    elif filename in df_M_list: 
        for fileImages in os.listdir(path+'/'+filename+'/'+filename+"_Dermoscopic_Image"):
            image_path = (path+'\\'+filename+'\\'+filename+'_Dermoscopic_Image\\'+fileImages) 
            !copy $image_path $new_folder\\Melanoma
``` 
## Creating Train, Test, and Val folders

I use Pillow to open the image and resize it into numpy array into 128x128 size. Then the label for each image is extracted using split() function. If you notice it, there are 2 imagePaths variables, the first it for .BMP format image and the second is for .PPM format image.
```
import numpy as np
import os
from PIL import Image
from imutils import paths

imagePaths = paths.list_images("original__dataset")
#imagePaths = paths.list_files("original_inpainted_dataset",validExts='.ppm')

data = []
labels = []
Cancers = {'Common_Nevus':1,'Atypical_Nevus':2,'Melanoma':3,'NonMelanoma':4}


for imagePath in imagePaths:
    pillow = Image.open(imagePath)
    pillow = pillow.resize((128,128))
    image = np.array(pillow)
    data.append(image)
    label = imagePath.split(os.path.sep)[-2]
    labels.append(Cancers[label])
    pillow.close()

data
```
I use train_test_split function using the sklearn.model_selection
```
from sklearn.model_selection import train_test_split
X_train, X_test,Y_train, Y_test = train_test_split(data ,labels, test_size=0.2, random_state=1)
X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, test_size=0.5, random_state=1)
print(len(X_train),len(X_test),len(X_val))
```
and then convert it to numpy array
```
X_train = np.array(X_train)
X_test = np.array(X_test)
X_val = np.array(X_val)
X_train.shape
```

The shape of the X_train should be ([the amount of image], 128, 128, 3) matrix.

In this project, I test deep learning CNN model and some sklearn model for the models

## Using CNN
the CNN model that I use is from this reference : https://github.com/deep-learning-indaba/indaba-2018/blob/master/Practical_2_Convolutional_Neural_Networks.ipynb . At the time of this project, I haven't learn much about building customize deep-learning model.
```
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"   
import tensorflow as tf
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
"""Model function for CNN."""

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=48, kernel_size=(3, 3), activation=tf.nn.relu, input_shape=( 128, 128, 3), padding='same'),
    tf.keras.layers.MaxPooling2D(pool_size=(3, 3)),
    tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation=tf.nn.relu, padding='same'),
    tf.keras.layers.MaxPooling2D(pool_size=(3, 3)),
    tf.keras.layers.Conv2D(filters=192, kernel_size=(3, 3), activation=tf.nn.relu, padding='same'),
    tf.keras.layers.Conv2D(filters=192, kernel_size=(3, 3), activation=tf.nn.relu, padding='same'),
    tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation=tf.nn.relu, padding='same'),
    tf.keras.layers.MaxPooling2D(pool_size=(3, 3)),
])

model.add(tf.keras.layers.Flatten())  # Flatten "squeezes" a 3-D volume down into a single vector.
model.add(tf.keras.layers.Dense(1024, activation=tf.nn.relu))
model.add(tf.keras.layers.Dropout(rate=0.5))
model.add(tf.keras.layers.Dense(1024, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
model.summary()
```

```
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=30, batch_size=32,validation_data=(X_val,Y_val))
```
The output should look like this
```
Train on 576 samples, validate on 72 samples
Epoch 1/30
576/576 [==============================] - 9s 16ms/sample - loss: 9.2006 - acc: 0.4965 - val_loss: 0.7010 - val_acc: 0.5139
```
and then you could wait a few minutes for the model to complete the training. 
After training the model, we use evaluate to know the accuracy of the prediction
```
metric_values = model.evaluate(x=X_test, y=Y_test)

print('Final TEST performance')
for metric_value, metric_name in zip(metric_values, model.metrics_names):
  print('{}: {}'.format(metric_name, metric_value))
```

## Using Random Forest Classifier, Support Vector Machine (SVM) or Multi Layer Perceptron (MLP).
In this method, I already had past experiences using this classifier, so i just re-use the code in **classify_image.py**. Just type in these codes to use it
```
%run -i classify_image.py -d original_dataset -m svm
```
The output should look like this
```
featureType : stat 1
[INFO] extracting image features...
[INFO] loading data...
[INFO] using 'svm' model
[INFO] evaluating...
              precision    recall  f1-score   support

    Melanoma       0.88      0.88      0.88         8
 NonMelanoma       0.97      0.97      0.97        32

    accuracy                           0.95        40
   macro avg       0.92      0.92      0.92        40
weighted avg       0.95      0.95      0.95        40
```
The usage of these models are not really optimum because there are limited times given for the test. Therefore I appologize for the minimum usage of these models here.
