import keras
from keras.models import Sequential
import numpy as np
from keras.layers import Dense, Activation, Dropout, MaxPooling2D, Flatten
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
from keras import initializers
# deal with data
train_data = np.loadtxt('zip_train.txt',dtype=np.float32)
test_data = np.loadtxt('zip_test.txt',dtype=np.float32)
train_label = train_data[:,0]
test_label = test_data[:,0]
train_data=np.delete(train_data,0,axis=1)
test_data = np.delete(test_data,0,axis=1)


train_data=train_data.reshape(7291,256)
test_data=test_data.reshape(2007,256)

train_data /=255
test_data /=255




test_label = keras.utils.to_categorical(test_label,10)
train_label = keras.utils.to_categorical(train_label,10)


model = Sequential()
keras.initializers.Zeros()
model.add(Dense(512,activation='relu', input_shape=(256,), ))
model.add(Dropout(0.2))
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10,activation='softmax'))
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
print ('Test accuracy', score[2])
#loss_history = history.history["loss"]
#accuracy_history = history.history["accuracy"]

#numpy_loss_history = np.array(loss_history)
#numpy_accuracy_history= np.array(loss_history)

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

plt.savefig("Fully")
plt.show()
