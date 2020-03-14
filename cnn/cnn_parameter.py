import keras
from keras.models import Sequential
import numpy as np
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import SGD
import matplotlib.pyplot as plt
# deal with data
train_data = np.loadtxt('zip_train.txt',dtype=np.float32)
test_data = np.loadtxt('zip_test.txt',dtype=np.float32)
train_label = train_data[:,0]
test_label = test_data[:,0]
train_data=np.delete(train_data,0,axis=1)
test_data = np.delete(test_data,0,axis=1)


train_data=train_data.reshape(-1,1,16,16)
test_data=test_data.reshape(-1,1,16,16)




test_label = keras.utils.to_categorical(test_label,10)
train_label = keras.utils.to_categorical(train_label,10)

def cnn_model(initializers,def_lr,def_bcs,def_momen):
    
    #add layer....
    model =Sequential()
    
    #conv 1
    model.add(Convolution2D(
        nb_filter=32,
        nb_row=5,
        nb_col=5,
        border_mode='same',
        input_shape=(1,16,16),
        kernel_initializer=initializers,
        bias_initializer='zeros'
        
        ))
    model.add(Activation('relu'))

    #pooling
    model.add(MaxPooling2D(
        pool_size=(2,2),
        strides=(2,2),
        border_mode='same',#padding method
        )
    
        )

    #conv 2 
    model.add(Convolution2D(64,5,5,border_mode='same',
        kernel_initializer=initializers, 
        bias_initializer ='zero'
        ))
    model.add(Activation('relu'))

    #pooling
    model.add(MaxPooling2D(pool_size=(2,2),border_mode='same'))

    #fully
    model.add(Flatten())
    model.add(Dense(1024,kernel_initializer=initializers,bias_initializer='zeros'))
    model.add(Activation('relu'))

    #fully
    model.add(Dense(10, kernel_initializer=initializers,bias_initializer='zeros'))
    model.add(Activation('softmax'))

    #optimizer
    sgd = SGD(lr=def_lr,momentum=def_momen)

    #compile
    model.compile(optimizer=sgd,
            loss='categorical_crossentropy',
            metrics=['mae','accuracy'])

    print ('Training ------------')

    history = model.fit(train_data,train_label,nb_epoch=10,batch_size=def_bcs,verbose=1,validation_data=(test_data, test_label))

    print('\nTesting-------------')
    score = model.evaluate(test_data, test_label, verbose=0)
    print ('Test loss:', score[0])
    print ('Test mae', score[1])
    print ('Test accuracy',score[2])
    return history
keras.initializers.Initializer()
def_lr = 1e-4
def_bcs=16
def_momen=1
plt.figure()
fig, ax = plt.subplots(2,1,figsize=(10,10))
for i in range(13):#//three training per time, by changing the numebr 13.
#i=0 //get the only last three to compare

#for j in range(3): #//get the only last three to compare

#for j in range(4):   #//compare different learning rate
#    i=0              #//only run i=0(he_normal) to compute lr
#    if j==0:            #compare he_normal in different learning rate
#        def_lr=1e-4
#        a='lr=0.0001'
#        b='1'
#    if j==1:
#        def_lr=1e-3
#        a='lr=0.001'
#        b='2'
#    if j==2:
#        def_lr=1e-2
#        a='lr=0.01'
#        b='3'
#    if j==3:
#        def_lr=1e-1
#        a='lr=0.1'
#        b='4'   




#for j in range(6): #//compare the different batch size
#    def_lr=0.001
#    i=0
#    
#    if j==0:
#        def_bcs=8
#        a='batchSize=8'
#        b='1'
#
#    if j==1:
#        def_bcs=16
#        a='batchSize=16'
#        b='2'
        
#    if j==2:
#        def_bcs=32
#        a='batchSize=32'
#        b='3'


#   if j==3:
#        def_bcs=64
#        a='batchSize=64'
#        b='4'
#
#    if j==4:
#        def_bcs=128
#        a='batchSize=128'
#        b='+'

#   if j==5:
#        def_bcs=256
#        a='batchSize=256'
#        b='<'


#for j in range(3):#//momentum changed
#    i=0
#    def_bcs=256
#    if j ==0:

#        def_momen=0.5
#        a='m=0.5'
#        b='<'

#    if j ==1:

#        def_momen=0.9
#        a='m=0.9'
#        b='>'
#    if j==2:
#        def_momen=0.99
#        a='m=0.99'
#        b='*'

        
    if i==0:
        initializers = keras.initializers.he_normal(seed=None)
        a='he_normal' #//regular
        b='.'         #//regular
        history=cnn_model(initializers,def_lr,def_bcs,def_momen)
        #i=5 //when compare last three
    elif i ==1:
        initializers=keras.initializers.lecun_normal(seed=None)
        a='lecun_normal'
        b='v'
        history=cnn_model(initializers,def_lr,def_bcs,def_momen)

    elif i ==2:
        initializers=keras.initializers.he_uniform(seed=None)
        a='he_uniform'
        b='<'
        history=cnn_model(initializers,def_lr,def_bcs,def_momen)

    elif i ==3:
        initializers=keras.initializers.glorot_uniform(seed=None)
        a='glorot_uniform'
        b='>'
        history=cnn_model(initializers,def_lr,def_bcs,def_momen)

    elif i ==4:
        initializers=keras.initializers.glorot_normal(seed=None)
        a='glorot_normal'
        b='1'
        history=cnn_model(initializers,def_lr,def_bcs,def_momen)

    elif i ==5:
        initializers=keras.initializers.lecun_uniform(seed=None)
        a='lecun_uniform'
        b='2'
        history=cnn_model(initializers,def_lr,def_bcs,def_momen)
        #i=7 //when compare last three
    

    elif i ==6:
        initializers=keras.initializers.Orthogonal(gain=1.0,seed=None)
        a='Orthogonal'
        b='4'
        history=cnn_model(initializers,def_lr,def_bcs,def_momen)

    elif i ==7:
        initializers=keras.initializers.VarianceScaling(scale=1.0,mode='fan_in',distribution='normal',seed=None)
        a='VarianceScaling'
        b='s'
        history=cnn_model(initializers,def_lr,def_bcs,def_momen)
        
    elif i ==8:
        initializers=keras.initializers.TruncatedNormal(mean=0.0,stddev=0.05,seed=None)
        a='TruncatedNormal'
        b='p'
        history=cnn_model(initializers,def_lr,def_bcs,def_momen)

    elif i ==9:
        initializers=keras.initializers.RandomUniform(minval=-0.05,maxval=0.05,seed=None)
        a='RandomUniform'
        b='*'
        history=cnn_model(initializers,def_lr,def_bcs,def_momen)

    elif i ==10:
        initializers=keras.initializers.RandomNormal(mean=0.0,stddev=0.05,seed=None)
        a='RandomNormal'
        b='h'
        history=cnn_model(initializers,def_lr,def_bcs,def_momen)

    elif i ==11:
        initializers=keras.initializers.Constant(value=10)
        a='Constant'
        b='H'
        history=cnn_model(initializers,def_lr,def_bcs,def_momen)

    elif i ==12:
        initializers=keras.initializers.Ones()
        a='ones'
        b='+'
        history=cnn_model(initializers,def_lr,def_bcs,def_momen)

 
 
 
 
 
 
 
 
 
 

    
    
    ax[0].plot(history.history['loss'],color='r',marker=b,label='TrL of '+a)
    ax[0].plot(history.history['val_loss'], color='g',marker=b,label='TeL of '+a)
    ax[0].legend(loc=3,bbox_to_anchor=(0.9,0),fontsize=5)
    ax[0].grid(True)

    ax[1].plot(history.history['accuracy'], color='r', marker=b,label='TrA of '+ a)
    ax[1].plot(history.history['val_accuracy'], color='g', marker=b, label ='TeA of '+a)
    
    ax[1].legend(loc=3,bbox_to_anchor=(0.9,0),ncol=1,fontsize=5)
    ax[1].grid(True)
    

plt.savefig("cnn(different momentum)")
plt.show()




