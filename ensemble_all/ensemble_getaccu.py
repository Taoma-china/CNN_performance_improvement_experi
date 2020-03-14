from scipy import stats
import keras
from keras.models import Sequential
import numpy as np
from keras.layers import LocallyConnected2D,Dropout, Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import SGD
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from keras.models import Model, Input
import tensorflow as tf
from keras.optimizers import Adam
## deal with data
import random

from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
from matplotlib import pyplot
from numpy import mean
from numpy import std
from numpy import array
from numpy import argmax
from keras.models import load_model





train_data = np.loadtxt('zip_train.txt',dtype=np.float32)
test_data = np.loadtxt('zip_test.txt',dtype=np.float32)
train_label = train_data[:,0]
test_label = test_data[:,0]
test_label_c = test_label
train_data=np.delete(train_data,0,axis=1)
test_data = np.delete(test_data,0,axis=1)
test_data = np.reshape(test_data,(-1,1,16,16))
test_label = keras.utils.to_categorical(test_label,10)
train_label = keras.utils.to_categorical(train_label,10)

test_data_34 =np.reshape(test_data,(2007,256))
test_data_56 =np.reshape(test_data,(-1,16,16,1))
def evaluate_model(model,train_data,train_label,test_data, test_label,def_lr,def_momen,def_bcs):
#optimizer
    sgd = SGD(lr=def_lr,momentum=def_momen)
    adam = Adam(lr=def_lr)
    #compile
    model.compile(optimizer=sgd,
            loss='categorical_crossentropy',
            metrics=['accuracy'])

    print('Training ------------')


    model.fit(train_data,train_label,nb_epoch=10,batch_size=def_bcs,verbose=1)

    #print('\nTesting-------------')
    score= model.evaluate(test_data, test_label, verbose=0)
    _ = score[0]
    test_acc = score[1]
    #print ('Test loss:', score[0])
    #print ('Test mae', score[1])
    #print ('Test accuracy',score[2])

    return model,test_acc


def ensemble_predictions(members,testX):
    yhats = [model.predict(testX) for model in members]
    yhats = array(yhats)

    summed =np.sum(yhats,axis=0)
    result = argmax(summed,axis=1)

    return result

def evaluate_n_members(members,testX,testy):
    subset=members
    yhat = ensemble_predictions(subset,testX)
    return accuracy_score(testy,yhat)






members =[]
model1 = load_model('1-1.h5')
model2 = load_model('1-2.h5')
model3 = load_model('2-1.h5')
model4 = load_model('2-2.h5')
model5 = load_model('3-1.h5')
model6 = load_model('3-2.h5')


pre_model1=model1.predict(test_data)
#model 3 4 need test_data(2007,256)
pre_model1=(argmax(pre_model1,axis=1))
members.append(pre_model1)

pre_model2=model2.predict(test_data)
pre_model2=(argmax(pre_model2,axis=1))
members.append(pre_model2)

pre_model3=model3.predict(test_data_34)
pre_model3=(argmax(pre_model3,axis=1))
members.append(pre_model3)

pre_model4=model4.predict(test_data_34)
pre_model4=(argmax(pre_model4,axis=1))
members.append(pre_model4)

pre_model5=model5.predict(test_data_56)
pre_model5=(argmax(pre_model5,axis=1))
members.append(pre_model5)

pre_model6=model6.predict(test_data_56)
pre_model6=(argmax(pre_model6,axis=1))
members.append(pre_model6)
#print(members)

members=np.array(members)
print(members)

members=stats.mode(members,axis=0)[0]

pre_test = members[0]
count=0
for i in range(2007):

    if pre_test[i]==test_label_c[i]:
        count=count+1
print ("the number of correct: ",count)
print ("accuracy: ",(count/2007))
print ("error: ",(1-(count/2007)))




