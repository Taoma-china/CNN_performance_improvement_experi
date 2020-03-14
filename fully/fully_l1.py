import keras
from keras.models import Sequential
import numpy as np
from keras.layers import Dropout,Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, RMSprop
import matplotlib.pyplot as plt
from keras import regularizers
# deal with data
train_data = np.loadtxt('zip_train.txt',dtype=np.float32)
test_data = np.loadtxt('zip_test.txt',dtype=np.float32)
train_label = train_data[:,0]
test_label = test_data[:,0]
train_data=np.delete(train_data,0,axis=1)
test_data = np.delete(test_data,0,axis=1)


train_data=train_data.reshape(7291,256)
test_data=test_data.reshape(2007,256)




test_label = keras.utils.to_categorical(test_label,10)
train_label = keras.utils.to_categorical(train_label,10)

def fully_model(initializers,def_lr,def_bcs,def_momen,drop,k,ak):
    
    
    model = Sequential()
    model.add(Dense(512,activation='relu', input_shape=(256,),kernel_initializer=initializers,bias_initializer='zeros',kernel_regularizer=k,activity_regularizer=ak))
    model.add(Dropout(drop))
    model.add(Dense(512,activation='relu',kernel_initializer=initializers,bias_initializer='zeros',kernel_regularizer=k,activity_regularizer=ak))
    model.add(Dropout(drop))
    model.add(Dense(10,activation='softmax',kernel_initializer=initializers,bias_initializer='zeros',kernel_regularizer=k,activity_regularizer=ak))
    model.summary()

    sgd = SGD(lr=def_lr,momentum=def_momen)
    model.compile(loss='categorical_crossentropy',
            optimizer=sgd,
            metrics=['mae','accuracy'])

    history = model.fit(train_data, train_label,
            batch_size = 128,
            epochs=10,
            verbose=1,
            validation_data=(test_data, test_label))
    score = model.evaluate(test_data, test_label, verbose=0)
    print ('Test loss:', score[0])
    print ('Test mae', score[1])
    print ('Test accuracy', score[2])
    #
    return history


keras.initializers.Initializer()
def_lr = 0.001
def_bcs=32
def_momen=0.99
plt.figure()

#for j in range(11):# test dropout

#    if j ==0:
#        
#        drop=0.0
#        a='drop 0.0'
#        c='b'
#    if j ==1:
#        drop=0.1
#        a='drop 0.1'
#        c='c'
#    if j ==2:
#        drop=0.3
#        a='drop 0.3'
#        c='g'
#    if j ==3:
#        drop=0.4
#        a='drop 0.4'
#        c='k'
#    if j==5:
#        drop=0.5
#        a='drop 0.5'
#        c='m'
#    if j==6:
#        drop=0.6
#        a='drop 0.6'
#        c='r'
#    if j==7:
#        drop=0.7
#        a='drop 0.7'
#        c='w'
#    if j==8:
#        drop=0.8
#        a='drop 0.8'
#        c='y'
#    if j==9:
#        drop=0.9
#        a='drop 0.9'
#        c='#708090'
#    if j==10:
#        drop=1
#        a='drop 1.0'
#        c='#c0c0c0'




for j in range(9):# test l1
    drop=0.3
    
    if j ==0:
        ak=regularizers.l1(1)
        k=regularizers.l1(1)
        a='kernel l1 1.0'
        c='b'
    if j ==1:
        ak=regularizers.l1(0.1)
        k=regularizers.l1(0.1)
        a='kernel l1 0.1'
        c='c'
    if j ==2:
        ak=regularizers.l1(0.01)
        k=regularizers.l1(0.01)
        a='kernel l1 0.01'
        c='g'
    if j ==3:
        ak=regularizers.l1(0.03)
        k=regularizers.l1(0.03)
        a='kernel l1 0.03'
        c='k'
    if j==4:
        ak=regularizers.l1(0.009)
        k=regularizers.l1(0.009)
        a='kernel l1 0.009'
        c='m'
    if j==5:
        ak=regularizers.l1(10)
        k=regularizers.l1(10)
        a='kernel l1 10'
        c='y'
    if j==6:
        ak=regularizers.l1(0.00001)
        k=regularizers.l1(0.00001)
        a='kernel l1 0.00001'
        c='w'
    if j==7:
        ak=regularizers.l1(0.000001)
        k=regularizers.l1(0.000001)
        a='kernel l1 0.000001'
        c='r'
    if j==8:
        ak=regularizers.l1(0.0000001)
        k=regularizers.l1(0.0000001)
        a='kernel l1 0.0000001'
        c='#c0c0c0'
    initializers=keras.initializers.VarianceScaling(scale=1.0,mode='fan_in',distribution='normal',seed=None)
     

    b='s'#when test lr
    history=fully_model(initializers,def_lr,def_bcs,def_momen,drop,k,ak)
 
 
 
 

    
    

    plt.plot(history.history['val_accuracy'], color=c, marker=b, label ='TeA of '+a)
    
    plt.legend(loc=3,bbox_to_anchor=(0.9,0.1),ncol=1,fontsize=5)
    plt.grid(True)
    

plt.savefig("fully_l1_all")
plt.show()




