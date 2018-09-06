#Copy of the code to edit model architecture
#Testing model architectures to predict 2 parameters (redshift and magnification)

import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, RMSprop, Adadelta, Adam
from keras import backend as K
K.set_image_dim_ordering('tf')
import time
time_i = time.time()




Dir1 = '/users/jbbutler129/Desktop/Argonne_Files/Galaxy_Lens_Regression/JPG/'
Dir2 = ['single/', 'stack/'][1]
Dir3 = ['0/', '1/'][1]
data_path = Dir1 + Dir2 + Dir3 + 'TrainingData/'
names = ['lensed', 'unlensed'][0]
data_dir_list = ['lensed_outputs', 'unlensed_outputs']

num_epoch = 10
batch_size = 16
learning_rate = .001  # Warning: lr and decay vary across optimizers
decay_rate = 0.01
opti_id = 1  # [SGD, Adadelta, RMSprop]
loss_id = 0 # [mse, mae] # mse is always better

image_size = img_rows = 45
img_cols = 45
num_channel = 1
num_classes = 1
num_files = 8000*num_classes
num_samples = num_files
num_para = 1


def load_train():
    img_data_list = []
    # labels = []

    #for name in names:
    #for labelID in [0, 1]:
        #name = names[labelID]
    for img_ind in range( int(num_files / num_classes) ):

        input_img = np.load(data_path + '/' + names + '_outputs/' + names + str(img_ind) + '.npy')
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

    #*******************************#
    para5 = labels[0:8000,5]
    print(para5)
    print(para5.shape)
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

def create_model_1():
    nb_filters = 32
    nb_conv = 3

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

    if opti_id == 0:
        sgd = SGD(lr=learning_rate, decay=decay_rate)
        # lr = 0.01, momentum = 0., decay = 0., nesterov = False
        model.compile(loss='mean_squared_error', optimizer=sgd)
    elif opti_id == 1:
        # Adadelta = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
        adadelta = Adadelta(lr = learning_rate, decay = decay_rate)
        model.compile(loss='mean_squared_error', optimizer= adadelta)
    else:
        # rmsprop = RMSprop(lr=learning_rate, decay=decay_rate)
        rmsprop = RMSprop()
        # lr = 0.001, rho = 0.9, epsilon = 1e-8, decay = 0.
        model.compile(loss='mean_squared_error', optimizer=rmsprop)

    # model.compile(loss=loss_fn , optimizer='sgd', metrics=["accuracy"])
    # model.compile(loss='mean_squared_error', optimizer=Adadelta())
    return model

#this one works pretty well, keep this for both lr of .001 and .01
##**keep some dropout
def create_model_2():

    model = Sequential()

    model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(image_size, image_size, 1) ))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    #model.add(BatchNormalization(axis=1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    #model.add(BatchNormalization(axis=1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.25))

    model.add(Convolution2D(128, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    #model.add(Convolution2D(128, 3, 3, border_mode='same'))
    #model.add(Activation('relu'))
    #model.add(BatchNormalization(axis=1))
    model.add(MaxPooling2D(pool_size=(2,2)))
    #model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    # model.add(BatchNormalization(axis=1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))

    model.add(Convolution2D(16, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(16, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    # model.add(BatchNormalization(axis=1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))

    ""
    #model.add(Convolution2D(128, 3, 3, border_mode='same'))
    #model.add(Activation('relu'))
    #model.add(Convolution2D(128, 3, 3, border_mode='same'))
    #model.add(Activation('relu'))
    #model.add(BatchNormalization(axis=1))
    #model.add(MaxPooling2D(pool_size=(2,2)))
    #model.add(Dropout(0.25)
    ""
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(num_para))
    model.add(Activation('linear'))


    if opti_id == 0:
        sgd = SGD(lr=learning_rate, decay=decay_rate)
        # lr = 0.01, momentum = 0., decay = 0., nesterov = False
        model.compile(loss='mean_squared_error', optimizer=sgd)
    elif opti_id == 1:
        # Adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        adam = Adam(lr = learning_rate, decay = decay_rate)
        model.compile(loss='mean_squared_error', optimizer= adam)

    elif opti_id == 2:
        # Adadelta = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
        adadelta = Adadelta(lr = learning_rate, decay = decay_rate)
        model.compile(loss='mean_squared_error', optimizer= adadelta)

    else:
        # rmsprop = RMSprop(lr=learning_rate, decay=decay_rate)
        rmsprop = RMSprop()
        # lr = 0.001, rho = 0.9, epsilon = 1e-8, decay = 0.
        model.compile(loss='mean_squared_error', optimizer=rmsprop)


    return model

def create_model_3():

    model = Sequential()

    model.add(Convolution2D(16, 3, 3, border_mode='same', input_shape=(image_size, image_size, 1)))
    model.add(Activation('relu'))
    model.add(Convolution2D(16, 3, 3))
    model.add(Activation('relu'))
    # model.add(BatchNormalization(axis=1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.25))

    model.add(Convolution2D(16, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(16, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    # model.add(BatchNormalization(axis=1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.25))

    model.add(Convolution2D(32, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    # model.add(BatchNormalization(axis=1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.25))

    model.add(Convolution2D(32, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    # model.add(BatchNormalization(axis=1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    # model.add(BatchNormalization(axis=1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.25))

    ""
    # model.add(Convolution2D(128, 3, 3, border_mode='same'))
    # model.add(Activation('relu'))
    # model.add(Convolution2D(128, 3, 3, border_mode='same'))
    # model.add(Activation('relu'))
    # model.add(BatchNormalization(axis=1))
    # model.add(MaxPooling2D(pool_size=(2,2)))
    # model.add(Dropout(0.25)
    ""
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_para))
    model.add(Activation('linear'))

    if opti_id == 0:
        sgd = SGD(lr=learning_rate, decay=decay_rate)
        # lr = 0.01, momentum = 0., decay = 0., nesterov = False
        model.compile(loss='mean_squared_error', optimizer=sgd)
    elif opti_id == 1:
        # Adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        adam = Adam(lr=learning_rate, decay=decay_rate)
        model.compile(loss='mean_squared_error', optimizer=adam)

    elif opti_id == 2:
        # Adadelta = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
        adadelta = Adadelta(lr=learning_rate, decay=decay_rate)
        model.compile(loss='mean_squared_error', optimizer=adadelta)

    else:
        # rmsprop = RMSprop(lr=learning_rate, decay=decay_rate)
        rmsprop = RMSprop()
        # lr = 0.001, rho = 0.9, epsilon = 1e-8, decay = 0.
        model.compile(loss='mean_squared_error', optimizer=rmsprop)

    return model

#An unnumbered create_model method, used to test different models. Models that perform well are saved above
#and labelled by a number
def create_model():

    model = Sequential()

    model.add(Convolution2D(8, 3, 3, border_mode='same', input_shape=(image_size, image_size, 1) ))
    model.add(Activation('relu'))
    model.add(Convolution2D(8, 3, 3))
    model.add(Activation('relu'))
    #model.add(BatchNormalization(axis=1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.25))

    model.add(Convolution2D(16, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    #model.add(BatchNormalization(axis=1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    #model.add(Convolution2D(128, 3, 3, border_mode='same'))
    #model.add(Activation('relu'))
    #model.add(BatchNormalization(axis=1))
    model.add(MaxPooling2D(pool_size=(2,2)))
    #model.add(Dropout(0.25))

    model.add(Convolution2D(128, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    # model.add(BatchNormalization(axis=1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))

    model.add(Convolution2D(16, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(16, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    # model.add(BatchNormalization(axis=1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))

    ""
    #model.add(Convolution2D(128, 3, 3, border_mode='same'))
    #model.add(Activation('relu'))
    #model.add(Convolution2D(128, 3, 3, border_mode='same'))
    #model.add(Activation('relu'))
    #model.add(BatchNormalization(axis=1))
    #model.add(MaxPooling2D(pool_size=(2,2)))
    #model.add(Dropout(0.25)
    ""
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(num_para))
    model.add(Activation('linear'))


    if opti_id == 0:
        sgd = SGD(lr=learning_rate, decay=decay_rate)
        # lr = 0.01, momentum = 0., decay = 0., nesterov = False
        model.compile(loss='mean_squared_error', optimizer=sgd)
    elif opti_id == 1:
        # Adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        adam = Adam(lr = learning_rate, decay = decay_rate)
        model.compile(loss='mean_squared_error', optimizer= adam)

    elif opti_id == 2:
        # Adadelta = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
        adadelta = Adadelta(lr = learning_rate, decay = decay_rate)
        model.compile(loss='mean_squared_error', optimizer= adadelta)

    else:
        # rmsprop = RMSprop(lr=learning_rate, decay=decay_rate)
        rmsprop = RMSprop()
        # lr = 0.001, rho = 0.9, epsilon = 1e-8, decay = 0.
        model.compile(loss='mean_squared_error', optimizer=rmsprop)


    return model

train_data, train_target = read_and_normalize_train_data()

# from sklearn.cross_validation import train_test_split
# train_data = train_data[0:num_samples,:,:,:]
# train_target = train_target[0:num_samples]
# X_train, X_valid, y_train, y_valid = train_test_split(train_data, train_target,
#                                                           test_size=cv_size,random_state=3)
# ModelFit = model.fit(X_train, y_train, batch_size=batch_size, nb_epoch= num_epoch, verbose=2, validation_data=(X_valid, y_valid) )

X_train = train_data[0:num_samples,:,:,:]
y_train = train_target[0:num_samples]

model = create_model_3()


ModelFit = model.fit(X_train, y_train, batch_size=batch_size, nb_epoch= num_epoch, verbose=1,
                     validation_split= 0.2 )

# model.save('ModelOutputs/cnn_regressionLens_test2.hdf5')

plotLossAcc = True
if plotLossAcc:
    import matplotlib.pylab as plt

    train_loss= ModelFit.history['loss']
    val_loss= ModelFit.history['val_loss']
    # train_acc= ModelFit.history['acc']
    # val_acc= ModelFit.history['val_acc']
    epochs= range(1, num_epoch+1)


    fig, ax = plt.subplots(1,1, sharex= True, figsize = (7,5))
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace= 0.02)
    ax.plot(epochs,train_loss)
    ax.plot(epochs,val_loss)
    ax.set_ylabel('loss')
    # ax.set_ylim([0.07,0.1])
    # ax[0].set_title('Loss')
    ax.legend(['train_loss','val_loss'])

    # accuracy doesn't make sense for regression

    plt.show()


SaveModel = True
if SaveModel:
    epochs = np.arange(1, num_epoch+1)
    train_loss = ModelFit.history['loss']
    val_loss = ModelFit.history['val_loss']

    training_hist = np.vstack([epochs, train_loss, val_loss])


    fileOut = 'RegressionStackNew_opti' + str(opti_id) + '_loss' + str(loss_id) + '_lr' + str(learning_rate) + '_decay' + str(decay_rate) + '_batch' + str(batch_size) + '_epoch' + str(num_epoch)

    model.save('/users/jbbutler129/Desktop/Argonne_Files/Galaxy_Lens_Regression/Model_Runs/' + fileOut + '.hdf5')
    np.save('/users/jbbutler129/Desktop/Argonne_Files/Galaxy_Lens_Regression/Model_Runs/' + fileOut + '.npy', training_hist)

time_j = time.time()
print(time_j - time_i, 'seconds')