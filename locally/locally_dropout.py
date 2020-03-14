import keras
from keras.models import Sequential
import numpy as np
from keras.layers import LocallyConnected2D, Dropout,Dense, Activation, Convolution2D, MaxPooling2D, Flatten
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


train_data=train_data.reshape(7291,16,16,1)
test_data=test_data.reshape(2007,16,16,1)




test_label = keras.utils.to_categorical(test_label,10)
train_label = keras.utils.to_categorical(train_label,10)
def locally_model(initializers,def_lr,def_bcs,def_momen,drop):
    
    model = Sequential()

    model.add(LocallyConnected2D(64,(3,3), input_shape=(16,16,1),init=initializers))
    model.add(Activation('relu'))
    model.add(Dropout(drop))

    model.add(LocallyConnected2D(32,(3,3),init=initializers))
    model.add(Dropout(drop))
    model.add(Flatten())
    model.add(Dense(10))
    model.add(Activation('softmax'))
    model.summary()

    sgd = SGD(lr=def_lr,momentum=def_momen)
    model.compile(loss='categorical_crossentropy',
        optimizer=sgd,
        metrics=['mae','accuracy'])
    
    history = model.fit(train_data, train_label,
        batch_size = def_bcs,
        epochs=10,
        verbose=1,
        validation_data=(test_data, test_label))
    score = model.evaluate(test_data, test_label, verbose=0)
    print ('Test loss:', score[0])
    print ('Test mae', score[1])
    print ('Test accuracy',score[2])


    return history


    


keras.initializers.Initializer()
def_lr = 0.01
def_bcs=128
def_momen=0.99
plt.figure()

for j in range(11):# test dropout

    if j ==0:
        
        drop=0.0
        a='drop 0.0'
        c='b'
    if j ==1:
        drop=0.1
        a='drop 0.1'
        c='c'
    if j ==2:
        drop=0.3
        a='drop 0.3'
        c='g'
    if j ==3:
        drop=0.4
        a='drop 0.4'
        c='k'
    if j==5:
        drop=0.5
        a='drop 0.5'
        c='m'
    if j==6:
        drop=0.6
        a='drop 0.6'
        c='r'
    if j==7:
        drop=0.7
        a='drop 0.7'
        c='w'
    if j==8:
        drop=0.8
        a='drop 0.8'
        c='y'
    if j==9:
        drop=0.9
        a='drop 0.9'
        c='#708090'
    if j==10:
        drop=1
        a='drop 1.0'
        c='#c0c0c0'





    initializers=keras.initializers.lecun_uniform(seed=None)
     

    b='s'#when test lr
    history=locally_model(initializers,def_lr,def_bcs,def_momen,drop)
 
 
 
 

    
    

    plt.plot(history.history['val_accuracy'], color=c, marker=b, label ='TeA of '+a)
    
    plt.legend(loc=3,bbox_to_anchor=(0.9,0.1),ncol=1,fontsize=5)
    plt.grid(True)
    

plt.savefig("locally_dropout")
plt.show()




