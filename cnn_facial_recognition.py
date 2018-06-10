""" Facial Recognition using CNN's
- by Vicente Opaso V.
"""

from keras.layers import Activation, Dense, Flatten, Dropout
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.utils import to_categorical
from keras.losses import categorical_crossentropy
from keras import optimizers
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

def getImg(N, dataset_path, setTypeFolderName):
    classes_label = {} # example: {0: 'person0', 1: 'person1', ..}
    X=np.empty((N, 224, 224, 3), dtype=np.uint8)
    y=np.empty((N, 1), dtype=np.uint8)
    i=0
    dir = os.path.join(dataset_path, setTypeFolderName)
    folders = os.listdir(dir)
    class_num = -1
    for folder in folders:
        dir2 = os.path.join(dir, folder)
        if os.path.isdir(dir2):
            class_num += 1
            classes_label[class_num] = folder
            suffix = '.jpg'
            for file in os.listdir(dir2):
                if file.endswith(suffix):
                    file_name = os.path.join(dir2, file)
                    img = cv2.imread(file_name)
                    if img is not None:
                        img=cv2.resize(img,(224, 224))
                        X[i, ...] = img
                        y[i, ...] = class_num
                        i = i+1
    return X, y, classes_label

def getClassNames(N, dataset_path, setTypeFolderName):
    classes_label = {}
    dir = os.path.join(dataset_path, setTypeFolderName)
    folders = os.listdir(dir)
    class_num = -1
    for folder in folders:
        dir2 = os.path.join(dir, folder)
        if os.path.isdir(dir2):
            class_num += 1
            classes_label[class_num] = folder
    return classes_label

print('Loading train and validation data..')

trainFolderName = 'train'
testFolderName = 'test'
valFolderName = 'val'
train_size = 52105
val_size = 6514
test_size = 6463
num_classes=100
dataset_path = './vggface2_dataset100'

# print('Loading train...')
x_train, y_train, train_classes_label = getImg(train_size, dataset_path, trainFolderName)

# print('Loading val...')
x_val, y_val, val_classes_label =  getImg(val_size, dataset_path, valFolderName)

print(y_train[-10:])
print(x_train.shape[0], 'train samples')
print(x_val.shape[0], 'validation samples')
  
classes_label = getClassNames(num_classes, dataset_path, 'test')


# -----------------------------
# Building AlexNet model 
# -----------------------------

img_rows, img_cols = 224, 224
input_shape = (img_rows, img_cols, 3)

y_train = to_categorical(y_train, num_classes=num_classes)
y_val = to_categorical(y_val, num_classes=num_classes)

#Container definition and 1st layer of AlexNet.
model = Sequential()
model.add(ZeroPadding2D((2,2), input_shape=input_shape))
model.add(Convolution2D(96, (11,11), strides=(4,4), padding='valid'))
model.add(Activation(activation='relu'))
model.add(BatchNormalization())
# print(model.output_shape) # network dimension before max-pooling
model.add(MaxPooling2D((3,3), strides=(2,2)))
# print(model.output_shape) # network dimension after max-pooling

#2nd layer
model.add(ZeroPadding2D((2,2)))
model.add(Convolution2D(256, (5, 5), padding='valid'))
model.add(Activation(activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((3,3), strides=(2,2)))
# print(model.output_shape) # 2th layer dimension
#model.summary()

#3rd layer
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(384, (3, 3), padding='valid'))
model.add(Activation(activation='relu'))
# print(model.output_shape) # 3th layer dimension
#model.summary()

#4th layer
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(384, (3, 3), padding='valid'))
model.add(Activation(activation='relu'))
# print(model.output_shape) # 4th layer dimension
#model.summary()

#5th layer
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), padding='valid'))
model.add(Activation(activation='relu'))
model.add(MaxPooling2D((3,3), strides=(2,2)))
# print(model.output_shape) # 5th layer dimension
#model.summary()

#6th layer
model.add(Flatten())
#print(model.output_shape) # flatten dimension
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))

#7th layer 
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))

#8th layer
model.add(Dense(num_classes, activation='softmax'))
# print(model.output_shape) # 8th layer dimension
model.summary()


# -----------------------------
##### Training the model
# -----------------------------

batch_size, epochs = 128, 15
weights_path = 'weights.hdf5'

model.compile(loss=categorical_crossentropy,
              optimizer=optimizers.Adam(lr=0.0001),
              metrics=['accuracy'])
checkpointer = ModelCheckpoint(filepath=weights_path, verbose=1, save_best_only=True)
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_val, y_val),callbacks=[checkpointer])

# We keep with the best model
model.load_weights(weights_path)
del x_train, y_train, x_val, y_val # freeing memory



# -----------------------------
# Visualization of the validation/training accuracy and loss
# -----------------------------

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()



# -----------------------------
# Evaluating the model
# -----------------------------

print('Loading test...')
x_test, y_test_or, test_classes_label =  getImg(test_size, dataset_path, testFolderName)
print(x_test.shape[0], 'test samples')

y_test = to_categorical(y_test_or, num_classes=num_classes)
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])



# -----------------------------
## Confusion matrix and best/worst performance
# -----------------------------

np.set_printoptions(threshold=1)
predictions = model.predict_classes(x_test, verbose=0)
m = np.zeros([num_classes,num_classes])
for i in range(test_size):
  tagged = int(y_test_or[i][0])
  predicted = int(predictions[i])
  m[predicted][tagged] = m[predicted][tagged]+1

worst = num_classes
worst_index = 0
best = 0
best_index = 0
for i in range(0,100):
  diagonal = m[i][i]
  total = np.sum(m[i])
  if diagonal/total > best:
    best = diagonal/total
    best_index = i
  if diagonal/total < worst:
    worst = diagonal/total
    worst_index = i
print('The worst class is: {0} with an accuracy of: {1}'.format(classes_label[worst_index],worst))
print('The best class is: {0} with an accuracy of: {1}'.format(classes_label[best_index],best))

# confusion matrix plot
from matplotlib import pyplot as plt
plt.imshow(m, interpolation='nearest')
plt.ylabel('Predicted')
plt.xlabel('Labeled')
plt.show()
