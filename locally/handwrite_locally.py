import keras
from keras.models import Sequential
import numpy as np
from keras.layers import Dense, Activation, Dropout, MaxPooling2D, Flatten, LocallyConnected2D
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
# deal with data
train_data = np.loadtxt('zip_train.txt',dtype=np.float32)
test_data = np.loadtxt('zip_test.txt',dtype=np.float32)
train_label = train_data[:,0]
test_label = test_data[:,0]
train_data=np.delete(train_data,0,axis=1)
test_data = np.delete(test_data,0,axis=1)

print (train_data.shape)
train_data=np.reshape(train_data,(7291,16,16,1))
test_data=np.reshape(test_data,(2007,16,16,1))

#train_data /=255
#test_data /=255


print (train_label)
print (train_label.shape)
test_label = keras.utils.to_categorical(test_label,10)
train_label = keras.utils.to_categorical(train_label,10)
print (train_label.shape)
print (train_label)
model = Sequential()

model.add(LocallyConnected2D(64,(3,3), input_shape=(16,16,1),init='VarianceScaling'))
model.add(Activation('relu'))


model.add(LocallyConnected2D(32,(3,3)))
model.add(Flatten())
model.add(Dense(10))
model.add(Activation('softmax'))
model.summary()


model.compile(loss='categorical_crossentropy',
        optimizer=RMSprop(),
        metrics=['mae','accuracy'])

history = model.fit(train_data, train_label,
        batch_size = 128,
        epochs=10,
        verbose=1,
        validation_data=(test_data, test_label))
score = model.evaluate(test_data, test_label, verbose=0)
print ('Test loss:', score[0])
print ('Test mae', score[1])
print ('Test accuracy',score[2])
plt.figure()
fig, ax = plt.subplots(2,1,figsize=(10,10))
ax[0].plot(history.history['loss'],color='r',label='Training Loss')
ax[0].plot(history.history['val_loss'], color='g',label='Test loss')
ax[0].legend(loc='best',shadow=True)
ax[0].grid(True)

ax[1].plot(history.history['accuracy'], color='r', label='Training Accuracy')
ax[1].plot(history.history['val_accuracy'], color='g', label ='Test Accuracy')
ax[1].legend(loc='best',shadow=True)
ax[1].grid(True)

plt.savefig("locally")
plt.show()

