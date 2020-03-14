import keras
from keras.models import Sequential
import numpy as np
from keras.layers import LocallyConnected2D,Dropout,Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, RMSprop
import matplotlib.pyplot as plt
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

def locally_model(initializers,def_lr,def_bcs,def_momen):
    
    model = Sequential()

    model.add(LocallyConnected2D(64,(3,3), input_shape=(16,16,1),init=initializers))
    model.add(Activation('relu'))


    model.add(LocallyConnected2D(32,(3,3),init=initializers))
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
def_momen=1
plt.figure()
fig, ax = plt.subplots(2,1,figsize=(10,10))
for i in range(13):#//three training per time, by changing the numebr 13.
#i=5 #get the only last three to compare

#for j in range(3): #//get the only last three to compare

#for j in range(4):   #//compare different learning rate
#    i=5              #//only run i=0(he_normal) to compute lr
#    if j==0:            #compare he_normal in different learning rate
#        def_lr=1e-4
#        a='lr=0.0001'
#        c='m'
#    if j==1:
#        def_lr=1e-3
#        a='lr=0.001'
#        c='c'
#    if j==2:
#        def_lr=1e-2
#        a='lr=0.01'
#        c='r'
#    if j==3:
#        def_lr=1e-1
#        a='lr=0.1'
#        c='y'   




#for j in range(6): #//compare the different batch size
#    def_lr=0.01
#    i=5
   
#    if j==0:
#        def_bcs=8
#        a='batchSize=8'
#        c='r'

#    if j==1:
#        def_bcs=16
#        a='batchSize=16'
#        c='y'
        
#    if j==2:
#        def_bcs=32
#        a='batchSize=32'
#        c='g'


#    if j==3:
#        def_bcs=64
#        a='batchSize=64'
#        c='m'

#    if j==4:
#        def_bcs=128
#        a='batchSize=128'
#        c='b'

#    if j==5:
#        def_bcs=256
#        a='batchSize=256'
#        c='k'


#for j in range(3):#//momentum changed
#    i=5
#    def_lr =0.01
#    def_bcs =128
#    if j ==0:
#        def_momen=0.5
#        a='m=0.5'
#        c='r'

#    if j ==1:

#        def_momen=0.9
#        a='m=0.9'
#        c='g'
#    if j==2:
#        def_momen=0.99
#        a='m=0.99'
#        c='k'



    if i==0:
        initializers = keras.initializers.he_normal(seed=None)
        a='he_normal' #//regular
        c='r'
        history=locally_model(initializers,def_lr,def_bcs,def_momen)
       
    elif i ==1:
        initializers=keras.initializers.lecun_normal(seed=None)
        a='lecun_normal'
        c='g'
        history=locally_model(initializers,def_lr,def_bcs,def_momen)

    elif i ==2:
        initializers=keras.initializers.he_uniform(seed=None)
        a='he_uniform'
        c='b'
        history=locally_model(initializers,def_lr,def_bcs,def_momen)
        #i=4 #compare last four
    elif i ==3:
        initializers=keras.initializers.glorot_uniform(seed=None)
        a='glorot_uniform'
        c='c'
        history=locally_model(initializers,def_lr,def_bcs,def_momen)
       
    elif i ==4:
        initializers=keras.initializers.glorot_normal(seed=None)
        a='glorot_normal'
        c='m'
        history=locally_model(initializers,def_lr,def_bcs,def_momen)
    elif i ==5:
        initializers=keras.initializers.lecun_uniform(seed=None)
        a='lecun_uniform'
        c='y'#test lr
        history=locally_model(initializers,def_lr,def_bcs,def_momen)
        
        #i=8 #compare last three

    elif i ==6:
        initializers=keras.initializers.Orthogonal(gain=1.0,seed=None)
        a='Orthogonal'
        c='k'
        history=locally_model(initializers,def_lr,def_bcs,def_momen)

    elif i ==7:
        initializers=keras.initializers.VarianceScaling(scale=1.0,mode='fan_in',distribution='normal',seed=None)
        a='VarianceScaling'#when test lr 
        c='w'
        history=locally_model(initializers,def_lr,def_bcs,def_momen)
       
    elif i ==8:
        initializers=keras.initializers.TruncatedNormal(mean=0.0,stddev=0.05,seed=None)
        a='TruncatedNormal'
        c='#D2691E'
        history=locally_model(initializers,def_lr,def_bcs,def_momen)
       # i=10#compare last three
    elif i ==9:
        initializers=keras.initializers.RandomUniform(minval=-0.05,maxval=0.05,seed=None)
        a='RandomUniform'
        c='#FFE4C4'
        history=locally_model(initializers,def_lr,def_bcs,def_momen)

    elif i ==10:
        initializers=keras.initializers.RandomNormal(mean=0.0,stddev=0.05,seed=None)
        a='RandomNormal'
        c='#A9A9A9'
        history=locally_model(initializers,def_lr,def_bcs,def_momen)

    elif i ==11:
        initializers=keras.initializers.Constant(value=10)
        a='Constant'  
        c='#556B2F'
        history=locally_model(initializers,def_lr,def_bcs,def_momen)

    elif i ==12:
        initializers=keras.initializers.Ones()
        a='ones'
        c='#F0F8FF'
        history=locally_model(initializers,def_lr,def_bcs,def_momen)

 
 
 
 
 
 
 
 
 
 

    
    
    ax[0].plot(history.history['loss'],color=c,marker='+',label='TrL of '+a)
    ax[0].plot(history.history['val_loss'], color=c,marker='*',label='TeL of '+a)
    ax[0].legend(loc=3,bbox_to_anchor=(0.9,0),fontsize=5)
    ax[0].grid(True)

    ax[1].plot(history.history['accuracy'], color=c, marker='+',label='TrA of '+ a)
    ax[1].plot(history.history['val_accuracy'], color=c, marker='*', label ='TeA of '+a)
    
    ax[1].legend(loc=3,bbox_to_anchor=(0.9,0),ncol=1,fontsize=5)
    ax[1].grid(True)
    

plt.savefig("locally different different momen ")
plt.show()




