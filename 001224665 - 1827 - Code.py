import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.image import imread
#importaing the libarys needed for calculations, graph visulisation and data processing

DataDir='pedestrian-no-pedestrian/data'
os.listdir(DataDir)
#tells the code where to find the data set and lists the contents

TrainingDPath=DataDir+'/train/'
ValidationDPath=DataDir+'/validation/'
ValidationDPath
os.listdir(ValidationDPath)
os.listdir(TrainingDPath)
os.listdir(TrainingDPath+'pedestrian')[5]
#specifies and the folders in the data set specifing which data is verified and what is used for training

pedestrian=TrainingDPath+'pedestrian/'+'pic_073.jpg'
imread(pedestrian).shape
#gets a specifc photo to train on and gets the CNN

plt.imshow(imread(pedestrian))
#displays the image ussed from the first CNN (outdated)

os.listdir(TrainingDPath+'no pedestrian')
no_pedestrian=TrainingDPath+'no pedestrian/'+'train (612).jpg'
plt.imshow(imread(no_pedestrian))
#same as above, gets the layers and displays the photo but for a validated non pedestrian photot

len(os.listdir(TrainingDPath+'pedestrian'))
len(os.listdir(TrainingDPath+'no pedestrian'))
len(os.listdir(ValidationDPath+'pedestrian'))
len(os.listdir(ValidationDPath+'no pedestrian'))
#gets the quantity of each image in the data sets folders

dim1=[]
dim2=[]
for image_filename in os.listdir(ValidationDPath+'pedestrian'):
    img=imread(ValidationDPath+'pedestrian/'+image_filename)
    d1,d2,colors=img.shape
    dim1.append(d1)
    dim2.append(d2)
dim1[0:10]
#loops through all the pediastrian images and gets thier layers and data
sns.jointplot(x=dim1,y=dim2)
#creats a joint plot to visualize how images are disributed

np.mean(dim1)
np.mean(dim2)
#gets the avarage for the images above

dim1=[]
dim2=[]
for image_filename in os.listdir(TrainingDPath+'pedestrian'):
    img=imread(TrainingDPath+'pedestrian/'+image_filename)
    d1,d2,colors=img.shape
    dim1.append(d1)
    dim2.append(d2)

sns.jointplot(x=dim1,y=dim2)
np.mean(dim1)
np.mean(dim2)
#all the same processes as above

image_shape=(200,200,3)
#sets the shape of the images and the last number is quanitiy of colour channels

from tensorflow.keras.preprocessing.image import ImageDataGenerator
imread(pedestrian).max()
image_gen=ImageDataGenerator(rescale=1/255,shear_range=0.1,zoom_range=0.1,fill_mode='nearest')
image_gen.flow_from_directory(TrainingDPath)
image_gen.flow_from_directory(ValidationDPath)
#using image generation to make images workable

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,MaxPool2D,Dropout,Flatten
from tensorflow.keras.callbacks import EarlyStopping
#importing the CNN functions

model=Sequential()
model.add(Conv2D(filters=32,kernel_size=(3,3),input_shape=image_shape,activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(filters=64,kernel_size=(3,3),input_shape=image_shape,activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(filters=64,kernel_size=(3,3),input_shape=image_shape,activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128,activation='relu')) #same as adding activation in 2nd para
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
#creating the CNN model all layers convolutional layers, max pooling, a flattening, dense, dropout, and an output

early_stop=EarlyStopping(monitor='val_loss',patience=2)
#sets up earling stopping whle monitering loss

batch_size=16
train_image_gen = image_gen.flow_from_directory(TrainingDPath,target_size=image_shape[:2],color_mode='rgb',
                                                batch_size=batch_size,class_mode='binary')
val_image_gen = image_gen.flow_from_directory(ValidationDPath,target_size=image_shape[:2],color_mode='rgb',
                                               batch_size=batch_size,class_mode='binary',shuffle=False)
#create data generators for the training and validation sets

train_image_gen.class_indices
#outputs the class indices

results=model.fit(train_image_gen,epochs=15,validation_data=val_image_gen,callbacks=[early_stop])
#joins the model and training data to a variable

losses=pd.DataFrame(model.history.history)
losses.plot()
#outputs data table of the training and loss data

model.evaluate_generator(val_image_gen)
#final evaluation

pred=model.predict_generator(val_image_gen)
pred[:5]
predictions=pred>0.5
predictions[:5]
val_image_gen.classes

from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(val_image_gen.classes,predictions))
confusion_matrix(val_image_gen.classes,predictions)
