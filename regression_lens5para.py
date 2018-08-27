'''
Based on:

https://gist.github.com/neilslater/40201a6c63b4462e6c6e458bab60d0b4


gotta change labels -> continuous data
try >1 inputs : remember categorical was 4x1 not 1D

'''

# -*- coding: utf-8 -*-
import numpy as np
import os
import cv2
import pandas as pd

from sklearn.cross_validation import train_test_split
from keras.optimizers import RMSprop, Adam, Adadelta
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, RMSprop, adam
from keras import backend as K
K.set_image_dim_ordering('tf')




Dir1 = '/home/nes/Desktop/ConvNetData/lens/AllTrainTestSets/JPG/'
Dir2 = ['single/', 'stack/'][1]
Dir3 = ['0/', '1/'][1]
data_path = Dir1 + Dir2 + Dir3 + 'TrainingData/'
names = ['lensed', 'unlensed']
data_dir_list = ['lensed_outputs', 'unlensed_outputs']

image_size = img_rows = 45
img_cols = 45
num_channel = 1
num_epoch = 5
batch_size = 16

num_classes = 2
num_files = 800*num_classes

num_samples = num_files

num_para = 5
cv_size = 200



def load_train():
    img_data_list = []
    # labels = []

    # for name in names:
    for labelID in [0, 1]:
        name = names[labelID]
        for img_ind in range(num_files / num_classes):

            input_img = np.load(data_path + '/' + name + '_outputs/' + name + str(img_ind) + '.npy')
            if np.isnan(input_img).any():
                print(labelID, img_ind, ' -- ERROR: NaN')
            else:

                img_data_list.append(input_img)
                # labels.append([labelID, 0.5*labelID, 0.33*labelID, 0.7*labelID, 5.0*labelID] )

    img_data = np.array(img_data_list)
    img_data = img_data.astype('float32')
    # labels = np.array(labels)
    # labels = labels.astype('float32')

    img_data /= 255
    print (img_data.shape)

    if num_channel == 1:
        if K.image_dim_ordering() == 'th':
            img_data = np.expand_dims(img_data, axis=1)
            print (img_data.shape)
        else:
            img_data = np.expand_dims(img_data, axis=4)
            print (img_data.shape)

    else:
        if K.image_dim_ordering() == 'th':
            img_data = np.rollaxis(img_data, 3, 1)
            print (img_data.shape)

    X_train = img_data
    # y_train = np_utils.to_categorical(labels, num_classes)
    labels = np.load(Dir1 + Dir2 + Dir3 + 'Train5para.npy')
    # print labels1.shape
    print(labels.shape)

    para5 = labels[:,2:]
    np.random.seed(12345)
    shuffleOrder = np.arange(X_train.shape[0])
    np.random.shuffle(shuffleOrder)
    X_train = X_train[shuffleOrder]
    y_train = para5[shuffleOrder]

    # print y_train[0:10]
    # print y_train[0:10]

    return X_train, y_train

def read_and_normalize_train_data():
    train_data, train_target = load_train()
    train_data = np.array(train_data, dtype=np.float32)
    train_target = np.array(train_target, dtype=np.float32)
    m = train_data.mean()
    s = train_data.std()

    print ('Train mean, sd:', m, s )
    train_data -= m
    train_data /= s
    print('Train shape:', train_data.shape)
    print(train_data.shape[0], 'train samples')
    return train_data, train_target



def create_model_orig():
    nb_filters = 8
    nb_conv = 5

    model = Sequential()
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                            border_mode='valid',
                            input_shape=(image_size, image_size, 1) ) )
    model.add(Activation('relu'))

    model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    model.add(Activation('relu'))

    model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    model.add(Activation('relu'))

    model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Convolution2D(nb_filters*2, nb_conv, nb_conv))
    model.add(Activation('relu'))

    model.add(Convolution2D(nb_filters*2, nb_conv, nb_conv))
    model.add(Activation('relu'))

    model.add(Convolution2D(nb_filters*2, nb_conv, nb_conv))
    model.add(Activation('relu'))

    model.add(Convolution2D(nb_filters*2, nb_conv, nb_conv))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_para))
    model.add(Activation('linear'))

    model.compile(loss='mean_squared_error', optimizer=Adadelta())
    return model

# def train_model(batch_size = 32, num_epoch = 20):




def create_model():
    nb_filters = 32
    nb_conv = 3

    model = Sequential()
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                            border_mode='valid',
                            input_shape=(image_size, image_size, 1) ) )
    model.add(Activation('relu'))

    # model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    # model.add(Activation('relu'))
    #
    # model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    # model.add(Activation('relu'))

    model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    # model.add(Convolution2D(nb_filters*2, nb_conv, nb_conv))
    # model.add(Activation('relu'))
    #
    # model.add(Convolution2D(nb_filters*2, nb_conv, nb_conv))
    # model.add(Activation('relu'))
    #
    model.add(Convolution2D(nb_filters*2, nb_conv, nb_conv))
    model.add(Activation('relu'))

    model.add(Convolution2D(nb_filters*2, nb_conv, nb_conv))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_para))
    model.add(Activation('linear'))

    model.compile(loss='mean_squared_error', optimizer=Adadelta())
    return model

# def train_model(batch_size = 32, num_epoch = 20):


train_data, train_target = read_and_normalize_train_data()
train_data = train_data[0:num_samples,:,:,:]
train_target = train_target[0:num_samples]

X_train, X_valid, y_train, y_valid = train_test_split(train_data, train_target,
                                                          test_size=cv_size, random_state=3)

model = create_model()
ModelFit = model.fit(X_train, y_train, batch_size=batch_size, nb_epoch= num_epoch, verbose=2,
              validation_data=(X_valid, y_valid) )

predictions_valid = model.predict(X_valid, batch_size= batch_size, verbose=2)
print(predictions_valid)
    # batch_size can be different for validation. Better to keep just one while testing
    # compare = pd.DataFrame(data={'original':y_valid.reshape((cv_size,)),
    # 'prediction':predictions_valid.reshape((cv_size,))})
    # compare.to_csv('ModelOutputs/compare.csv')

model.save('ModelOutRegression/cnn_regressionLens_test.hdf5')
    # print compare

# return model

# ModelFit = train_model(num_epoch = 10)

# from keras.callbacks import History
# history = History()

# ModelFit

plotLossAcc = False
if plotLossAcc:
    import matplotlib.pylab as plt

    train_loss= ModelFit.history['loss']
    val_loss= ModelFit.history['val_loss']
    # train_acc= ModelFit.history['acc']
    # val_acc= ModelFit.history['val_acc']
    epochs= range(1, num_epoch+1)


    fig, ax = plt.subplots(2,1, sharex= True, figsize = (7,5))
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace= 0.02)
    ax[0].plot(epochs,train_loss)
    ax[0].plot(epochs,val_loss)
    ax[0].set_ylabel('loss')
    # ax[0].set_ylim([0,1])
    # ax[0].set_title('Loss')
    ax[0].legend(['train_loss','val_loss'])

    # accuracy doesn't make sense for regression

    # ax[1].plot(epochs,train_acc)
    # ax[1].plot(epochs,val_acc)
    # ax[1].set_xlabel('num of Epochs')
    # ax[1].set_ylabel('accuracy')
    # ax[1].set_ylim([0,1])
    # ## ax[1].set_title('Accuracy')
    # ax[1].legend(['train_acc','val_acc'], loc=1)

    plt.show()


Check_model = False
if Check_model:
    ModelFit.summary()
    ModelFit.get_config()
    ModelFit.layers[0].get_config()
    ModelFit.layers[0].input_shape
    ModelFit.layers[0].output_shape
    ModelFit.layers[0].get_weights()
    np.shape(ModelFit.layers[0].get_weights()[0])
    ModelFit.layers[0].trainable

    from keras.utils.vis_utils import plot_model
    plot_model(ModelFit, to_file='model_100runs_test.png', show_shapes=True)

