import numpy as np
import keras, os, random
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from scipy.misc import imread, imshow, imresize

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, InputLayer
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD

def create_basic_model(img_size):
    input_num_units = (img_size, img_size, 3)
    hidden_num_units = 500
    output_num_units = 3

    model = Sequential([
        InputLayer(input_shape=input_num_units),
        Flatten(),
        Dense(units=hidden_num_units, activation='relu'),
        Dense(units=output_num_units, activation='softmax'),
    ])

    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def create_cnn_model(img_size):
    input_num_units = (img_size, img_size, 3)
    pool_size = (2, 2)
    hidden_num_units = 500
    output_num_units = 3
    model = Sequential([
        InputLayer(input_shape=input_num_units),

        Conv2D(25, (5, 5), activation='relu'),
        MaxPooling2D(pool_size=pool_size),

        Conv2D(25, (5, 5), activation='relu'),
        MaxPooling2D(pool_size=pool_size),

        Conv2D(25, (4, 4), activation='relu'),

        Flatten(),

        Dense(units=hidden_num_units, activation='relu'),

        Dense(units=output_num_units, input_dim=hidden_num_units, activation='softmax'),
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def create_cnn_deep_model(img_size):
    nb_classes = 3
    model = Sequential()

    # Conv 1
    model.add(Conv2D(32, (3, 3), padding='same',
                            input_shape=(img_size, img_size, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization(epsilon=1e-06, axis=1, momentum=0.9))

    # Conv 2
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization(epsilon=1e-06, axis=1, momentum=0.9))

    # Conv 3
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization(epsilon=1e-06, axis=1, momentum=0.9))

    # Fully connected layer 1
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # Fully connected layer 2
    # model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    # Train the model using SGD + momentum
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    return model


data_dir = '/media/DATA/fromUbuntu/analyticsvidhya/age_detection/train_Sruxd3S'
train = pd.read_csv(os.path.join(data_dir, 'train.csv'))

print 'no of images: '
print len(train)

# #read training images
temp = []
count_img = 0
image_size = 64
# for img_name in train.ID:
#     print 'image name: ' + img_name
#     img_path = os.path.join(data_dir, 'Train', img_name)
#     img = imread(img_path)
#     print 'image size: ' + str(img.shape[0]) + 'x' + str(img.shape[1])
#     img = imresize(img, (image_size, image_size))
#     img = img.astype('float32') # this will help us in later stage
#     temp.append(img)
#     count_img += 1
#     print 'processed ' + str(count_img) + ' of ' + str(len(train))
#
# train_x = np.stack(temp)
# train_x = train_x / 255.
#
# np.save('train_images_64x64.npy', train_x)

#load training images
# temp = np.load('train_images_32x32.npy')
# images = temp.tolist()
# images = images[:]
# count = 0
# for img in temp:
#     tmp_img = np.flip(img,1)
#     images.append(tmp_img)
#     count += 1
#     # print 'processed ' + str(count) + ' of ' + str(len(temp)) + ' with shape: '
#     # print tmp_img.shape
#     # print len(images)
#
# train_x = np.array(images)
train_x = np.load('train_images_'+str(image_size)+'x'+str(image_size)+'.npy')

print 'distribution of classes: '
print train.Class.value_counts(normalize=True)

lb = LabelEncoder()
train_y = lb.fit_transform(train.Class)
train_y = keras.utils.np_utils.to_categorical(train_y)

# tmp_y = train_y.tolist()
# tmp_y = tmp_y[:]
# for c in train_y:
#     tmp_y.append(c)
#
# y_train = np.array(tmp_y)

print 'length of y: '
print len(train_y)

epochs = 10
batch_size = 32


# model = create_basic_model(image_size)
model = create_cnn_deep_model(image_size)
model.fit(train_x, train_y, batch_size=batch_size,epochs=epochs,verbose=1, validation_split=0.2)


