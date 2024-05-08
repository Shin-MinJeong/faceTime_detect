
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import classification_report, log_loss, accuracy_score
from sklearn.model_selection import train_test_split

# https://www.kaggle.com/datasets/roobansappani/hand-gesture-recognition/data
directory = 'D:/HandGesture/images'

# 데이터셋 확인
Name=[]
for file in os.listdir(directory):
    if file[-4:]!='pt.m' and file[-4:]!='.txt':
        Name+=[file]
print(Name)
print(len(Name))

N=[]
for i in range(len(Name)):
    N+=[i]
    
normal_mapping=dict(zip(Name,N)) 
reverse_mapping=dict(zip(N,Name)) 

def mapper(value):
    return reverse_mapping[value]

File=[]
for file in os.listdir(directory):
    File+=[file]
    print(file)
    
# 데이터 전처리
dataset=[]
testset=[]
count=0

for file in File:
    path=os.path.join(directory,file)
    t=0
    for im in os.listdir(path):
        if im[-4:]!='pt.m' and im[-4:]!='.txt':
            image=load_img(os.path.join(path,im), grayscale=False, color_mode='rgb', target_size=(60,60))
            image=img_to_array(image)
            image=image/255.0
            if t<400:
                dataset.append([image,count])
            else:   
                testset.append([image,count])
            t+=1
    count=count+1
    
data,labels0=zip(*dataset)
test,tlabels0=zip(*testset)

labels1=to_categorical(labels0)
data=np.array(data)
labels=np.array(labels1)

tlabels1=to_categorical(tlabels0)
test=np.array(test)
tlabels=np.array(tlabels1)

trainx,testx,trainy,testy=train_test_split(data,labels,test_size=0.2,random_state=44)

print(trainx.shape)

print(testx.shape)

print(trainy.shape)

print(testy.shape)
# 전처리 끝

# 모델구성
datagen = ImageDataGenerator(horizontal_flip=True,vertical_flip=True,rotation_range=20,zoom_range=0.2, width_shift_range=0.2,height_shift_range=0.2,shear_range=0.1,fill_mode="nearest")

pretrained_model3 = tf.keras.applications.DenseNet201(input_shape=(60,60,3),include_top=False,weights='imagenet',pooling='avg')
pretrained_model3.trainable = False

inputs3 = pretrained_model3.input
x3 = tf.keras.layers.Dense(128, activation='relu')(pretrained_model3.output)

outputs3 = tf.keras.layers.Dense(10, activation='softmax')(x3)

model = tf.keras.Model(inputs=inputs3, outputs=outputs3)
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
#모델구성 끝

'''
모델 컴파일 및 학습
datagen.flow 함수를 사용하여 데이터 증강을 적용한 데이터를 생성
이 후 모델에 전달하여 학습
학습은 20epoch 동안 진행
'''
his=model.fit(datagen.flow(trainx,trainy,batch_size=32),validation_data=(testx,testy),epochs=20)
#모델 컴파일 및 학습 끝

#모델의 예측값과 실제값을 비교하여 classification report를 출력
y_pred=model.predict(testx)
pred=np.argmax(y_pred,axis=1)
ground = np.argmax(testy,axis=1)
print(classification_report(ground,pred))

get_acc = his.history['accuracy']
value_acc = his.history['val_accuracy']
get_loss = his.history['loss']
validation_loss = his.history['val_loss']

epochs = range(len(get_acc))

plt.plot(epochs, get_acc, 'r', label='Accuracy of Training data')
plt.plot(epochs, value_acc, 'b', label='Accuracy of Validation data')
plt.title('Training vs validation accuracy')
plt.legend(loc=0)
plt.figure()
plt.show()

epochs = range(len(get_loss))

plt.plot(epochs, get_loss, 'r', label='Loss of Training data')
plt.plot(epochs, validation_loss, 'b', label='Loss of Validation data')
plt.title('Training vs validation loss')
plt.legend(loc=0)
plt.figure()
plt.show()

load_img("D:/HandGesture/images/rock_on/1317.jpg",target_size=(60,60))

image=load_img("D:/HandGesture/images/rock_on/1317.jpg",target_size=(60,60))

image=img_to_array(image) 
image=image/255.0
prediction_image=np.array(image)
prediction_image= np.expand_dims(image, axis=0)

prediction=model.predict(prediction_image)
value=np.argmax(prediction)
move_name=mapper(value)

print("Prediction is {}.".format(move_name))

print(test.shape)

prediction2=model.predict(test)

print(prediction2.shape)

PRED=[]
for item in prediction2:
    value2=np.argmax(item)      
    PRED+=[value2]
    
ANS=tlabels0
accuracy=accuracy_score(ANS,PRED)

print(accuracy)

model.save('handgest.hdf5')
model2 = keras.models.load_model('handgest.hdf5')
prediction3=model2.predict(test)

PRED3=[]
for item in prediction3:
    value3=np.argmax(item)      
    PRED3+=[value3]
accuracy3=accuracy_score(ANS,PRED3)

print(accuracy3)