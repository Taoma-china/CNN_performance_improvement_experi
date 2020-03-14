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










def sampling(dataMat,number):
    sample=[]
    for i in range(number):
        sample.append(dataMat[random.randint(0,len(dataMat)-1)])
    return sample

train_data = np.loadtxt('zip_train.txt',dtype=np.float32)
test_data = np.loadtxt('zip_test.txt',dtype=np.float32)
train_label = train_data[:,0]
test_label = test_data[:,0]
test_label_c = test_label
train_data=np.delete(train_data,0,axis=1)
test_data = np.delete(test_data,0,axis=1)

test_data_f =test_data
train_data_f =train_data
train_label_f=train_label
train_data_l=train_data
test_data_l=test_data


#input_shape= train_data[7291,16,16,1].shape
#model_input= Input(shape=input_shape)

#train_data = train_data/255
#test_data = test_data/255
train_data = train_data.reshape(-1,1,16,16)
print(train_data.shape)

input_shape = train_data[0,:,:,:].shape
print(input_shape)
model_input = Input(shape=(input_shape))
print('3')
print(model_input)

test_data=test_data.reshape(-1,1,16,16)
test_label = keras.utils.to_categorical(test_label,10)
train_label = keras.utils.to_categorical(train_label,10)
def cnn_model(model_input,initializers):
    
    
    #add layer....
    x = Sequential()       
    
    #conv 1
    x = Convolution2D(
        nb_filter=32,
        nb_row=5,
        nb_col=5,
        border_mode='same',
        input_shape=(1,16,16),
        kernel_initializer=initializers,
        bias_initializer='zeros'
        
        )(model_input)
    x = Activation('relu')(x)
    x = Dropout(0.2)(x)
    #pooling
    x = MaxPooling2D(
        pool_size=(2,2),
        strides=(2,2),
        border_mode='same',#padding method
        )(x)
    
        

    #conv 2 
    x = Convolution2D(64,5,5,border_mode='same',
        kernel_initializer=initializers, 
        bias_initializer ='zero'
        )(x)
    x = Activation('relu')(x)

    #pooling
    x = MaxPooling2D(pool_size=(2,2),border_mode='same')(x)

    #fully
    x = Flatten()(x)
    x = Dense(1024,kernel_initializer=initializers,bias_initializer='zeros')(x)
    x = Activation('relu')(x)
    x = Dropout(0.2)(x)
    #fully
    x = Dense(10, kernel_initializer=initializers,bias_initializer='zeros')(x)
    x = Activation('softmax')(x)

    #optimizer
    #sgd = SGD(lr=def_lr,momentum=def_momen)

    model = Model(model_input, x, name ='CNN')
    return model


def fully_model(model_input,initializers):
    
     
    x = Sequential()
    x =Dense(512,activation='relu', input_shape=(1,16,16),kernel_initializer=initializers,bias_initializer='zeros')(model_input)
    x = Dropout(0.2)(x)
    x = Dense(512,activation='relu',kernel_initializer=initializers,bias_initializer='zeros')(x)
    x = Dropout(0.2)(x)
    x = Dense(10,activation='softmax',kernel_initializer=initializers,bias_initializer='zeros')(x)


    

    model = Model(model_input, x, name ='fully')
    return model



def locally_model(model_input,initializers):
    
     
    x = Sequential()
    x = LocallyConnected2D(64,(3,3),input_shape=(16,16,1),init=initializers)(model_input)
    x = Activation('relu')(x)
    x = LocallyConnected2D(32,(3,3),init=initializers)(x)
    x = Flatten()(x)
    x = Dense(10)(x)
    x = Activation('softmax')(x)


    

    model = Model(model_input, x, name ='locally')
    return model







































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

def evaluate_n_members(members,n_members,testX,testy):
    subset=members[:n_members]
    yhat = ensemble_predictions(subset,testX)
    return accuracy_score(testy,yhat)



scores=[]
members=[]
for i in range(3):
    
    
    if i ==0:
        keras.initializers.Initializer()
        def_lr=1e-3
        def_bcs=256
        def_momen=0.99
        initializers=keras.initializers.he_normal(seed=None)
        model =cnn_model(model_input, initializers) 
        model,test_acc = evaluate_model(model,train_data,train_label,test_data,test_label,def_lr,def_momen,def_bcs)
        print('>%.3f' % test_acc)
        scores.append(test_acc)
        members.append(model)
        
        model.save("1-1.h5")
        #2
    
        keras.initializers.Initializer()
        def_lr=1e-3
        def_bcs=128
        def_momen=0.99
        initializers=keras.initializers.VarianceScaling(scale=1.0,mode='fan_in', distribution='normal',seed=None)
        model =cnn_model(model_input, initializers) 
        model,test_acc = evaluate_model(model,train_data,train_label,test_data,test_label,def_lr,def_momen,def_bcs)
        print('>%.3f' % test_acc)
        scores.append(test_acc)
        members.append(model)
        model.save('1-2.h5')
    if i ==1:
        train_data_f=train_data_f.reshape(7291,256)
        test_data_f = test_data_f.reshape(2007,256)
        model_input=Input(shape=(256,))
        keras.initializers.Initializer()
        def_lr = 0.001
        def_bcs=32
        def_momen=0.99
        initializers = keras.initializers.VarianceScaling(scale=1.0,mode='fan_in',distribution='normal',seed=None)
        
        model =fully_model(model_input, initializers) 

        model,test_acc = evaluate_model(model,train_data_f,train_label,test_data_f,test_label,def_lr,def_momen,def_bcs)
        print('>%.3f' % test_acc)
        scores.append(test_acc)
        members.append(model)
        model.save("2-1.h5")

    #2
        keras.initializers.Initializer()
        def_lr = 0.0001
        def_bcs=16
        def_momen=0.99
        
 
        initializers = keras.initializers.he_uniform(seed=None)
        
        model =fully_model(model_input, initializers) 

        model,test_acc = evaluate_model(model,train_data_f,train_label,test_data_f,test_label,def_lr,def_momen,def_bcs)
        print('>%.3f' % test_acc)
        scores.append(test_acc)
        members.append(model)
        model.save('2-2.h5')



    if i==2:


        train_data_l=np.reshape(train_data_l,(7291,16,16,1))
        test_data_l=np.reshape(test_data_l,(2007,16,16,1))




        
        model_input = Input(shape=(16,16,1))
        keras.initializers.Initializer()
        def_lr = 0.01
        def_bcs=256
        def_momen=0.99
        initializers=keras.initializers.lecun_uniform(seed=None)



        model =locally_model(model_input, initializers) 

        model,test_acc = evaluate_model(model,train_data_l,train_label,test_data_l,test_label,def_lr,def_momen,def_bcs)
        print('>%.3f' % test_acc)
        scores.append(test_acc)
        members.append(model)
        model.save('3-1.h5')


    
        keras.initializers.Initializer()
        def_lr = 0.01
        def_bcs=128
        def_momen=0.99
        initializers=keras.initializers.TruncatedNormal(mean=0.0,stddev=0.05,seed=None)



        model =locally_model(model_input, initializers) 

        model,test_acc = evaluate_model(model,train_data_l,train_label,test_data_l,test_label,def_lr,def_momen,def_bcs)
        print('>%.3f' % test_acc)
        scores.append(test_acc)
        members.append(model)

        model.save('3-2.h5')







print ('Estimated Accuracy %.3f (%.3f)'%(mean(scores),std(scores)))

single_scores=[]
ensemble_scores=[]

for i in range(1,7):
    ensemble_score =evaluate_n_members(members,i,test_data,test_label_c)
    newy_enc = to_categorical(test_label_c)
    _,single_score = members[i-1].evaluate(test_data,newy_enc,verbose=0)
    print('> %d: single=%.3f, ensemble=%.3f'%(i,single_score,ensemble_score))
    ensemble_scores.append(ensemble_score)
    single_scores.append(single_score)
print ('Accuracy %.3f (%.3f)' % (mean(single_scores),std(single_scores)))

x_axis=[i for i in range(1,7)]
pyplot.plot(x_axis,single_scores,marker='o',linestyle='None')
pyplot.plot(x_axis,ensemble_scores,marker='o')
pyplot.savefig('ensemble_network_total')
pyplot.show()

'''
keras.initializers.Initializer()
def_lr = 1e-4
def_bcs=256
def_momen=1


initializers = keras.initializers.he_normal(seed=None)

'''
        #a='he_normal' #//regular
        #b='.'         #//regular
#model =cnn_model(model_input, initializers) 
#history=compile_and_train(model,train_data1,train_label1,test_data,test_label,def_lr,def_momen,def_bcs)


'''

plt.figure()
fig, ax = plt.subplots(2,1,figsize=(10,10))

ax[0].plot(history.history['loss'],color='r',label='train_loss')
ax[0].plot(history.history['val_loss'], color='g',label='test_loss')
ax[0].legend(loc=3,bbox_to_anchor=(0.9,0),fontsize=5)
ax[0].grid(True)

ax[1].plot(history.history['accuracy'], color='r',label='train_acc')
ax[1].plot(history.history['val_accuracy'], color='g',label='test_acc')
    
ax[1].legend(loc=3,bbox_to_anchor=(0.9,0),ncol=1,fontsize=5)
ax[1].grid(True)
    

plt.savefig("cnn ensemble(network)")
plt.show()



'''
