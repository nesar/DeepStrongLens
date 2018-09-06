'''
Based on:
https://gist.github.com/neilslater/40201a6c63b4462e6c6e458bab60d0b4
Remaining:  1) Get std dev on predictions
            2) Deeper, wider networks ? - right now, loss function seems converging - but testing is bad
            3) Use sklearn, pipeline etc
            4) Use callbacks
            5) Plot conditional prob distribution kinda thing
            6) What metric to use to check accuracy of all testimages?
'''

# -*- coding: utf-8 -*-
import numpy as np
from keras import backend as K
import matplotlib.pylab as plt
K.set_image_dim_ordering('tf')
from keras.models import load_model
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
import glob
import time
time_i = time.time()




Dir1 = '/users/jbbutler129/Desktop/Argonne_Files/Galaxy_Lens_Regression/JPG/'
Dir2 = ['single/', 'stack/'][1]
Dir3 = ['0/', '1/'][1]
data_path = Dir1 + Dir2 + Dir3 + 'TestData/'
names = ['lensed', 'unlensed']
data_dir_list = ['lensed_outputs', 'unlensed_outputs']

image_size = img_rows = 45
img_cols = 45
num_channel = 1
# num_epoch = 10
# batch_size = 8

num_classes = 2
num_files = num_classes*10
# num_para = 5


# num_samples = 1999
# cv_size = 2000



#loaded_model = load_model('ModelOutputs/cnn_regressionLens_test2.hdf5')

loaded_model = load_model('/users/jbbutler129/Desktop/Argonne_Files/Galaxy_Lens_Regression/Model_Runs/Model4.hdf5')


# filelist = sorted(glob.glob(DirIn +'*.npy'))   # All
# hyperpara = '*opti1*lr0.001*decay0.1*batch16*epoch500*'
#hyperpara = 'Deeper*300*'

#filelist = sorted(glob.glob(DirIn + hyperpara + '*.hdf5'))
#histlist = sorted(glob.glob(DirIn + hyperpara + '*.npy'))

#print(len(filelist))

#for i in range(len(filelist)):
   # fileIn = filelist[i]
   # histIn = histlist[i]
    #loaded_model = load_model(fileIn)
    #print(fileIn)
    #history = np.load(histIn)
    #print(histIn)



def load_test():
    img_data_list = []
    # labels = []

    # for name in names:
    for labelID in [0,  1 ]:
        name = names[labelID]
        for img_ind in range( int(num_files / num_classes) ):

            input_img = np.load(data_path + name + str(img_ind) + '.npy')
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

    X_test = img_data
    # y_train = np_utils.to_categorical(labels, num_classes)
    labels = np.load(Dir1 + Dir2 + Dir3 + 'Test5para.npy')
    print(labels.shape)

    para5 = labels[:,2:]
    np.random.seed(12345)
    shuffleOrder = np.arange(X_test.shape[0])
    np.random.shuffle(shuffleOrder)
    X_test = X_test[shuffleOrder]
    y_test = para5[shuffleOrder]
    # y_train = labels1[shuffleOrder]

    # print y_train[0:10]
    # print y_train[0:10]

    return X_test, y_test

def read_and_normalize_test_data():
    test_data, test_target = load_test()
    test_data = np.array(test_data, dtype=np.float32)
    test_target = np.array(test_target, dtype=np.float32)
    m = test_data.mean()
    s = test_data.std()

    print ('Test mean, sd:', m, s )
    test_data -= m
    test_data /= s
    print('Test shape:', test_data.shape)
    print(test_data.shape[0], 'test samples')
    return test_data, test_target




rescaleMin, rescaleMax = np.load(Dir1 + Dir2 + Dir3 + 'RescalingMinMax5para.npy')
print(rescaleMin.shape)

test_data, test_target = read_and_normalize_test_data()
test_data = test_data[0:num_files,:,:,:]
test_target = test_target[0:num_files]


########## Predictions ######################

print('vel-dispersion  ellipticity  orientation  z  magnification')

predictions = np.zeros_like(test_target)

for i in range(num_files):
    test_img = np.expand_dims(test_data[i], axis=0)
    predictions[i] = loaded_model.predict(test_img, batch_size= 1, verbose=0)[0]
    print('')


######### Check #####################

for i in range(num_files):

    print('true: ', rescaleMin + (rescaleMax - rescaleMin)*test_target[i])
    print('pred: ', rescaleMin + (rescaleMax - rescaleMin)*np.array(predictions[i]))
    print(30*'-')




import matplotlib.pylab as plt

# plt.figure(10)
fig, ax = plt.subplots(2, 3, figsize=(10, 6))
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)


ax[0, 0].plot(  test_target[:, 0], predictions[:, 0],
                'kx', label = 'rescaled vel-dispersion')
ax[0, 1].plot( test_target[:, 1], predictions[:, 1], 'kx',
               label = 'rescaled ellipticity')
ax[0, 2].plot( test_target[:, 2], predictions[:, 2], 'kx',
               label = 'rescaled orientation')
ax[1, 0].plot( test_target[:, 3], predictions[:, 3], 'kx',
               label = 'rescaled redshift')
ax[1, 1].plot( test_target[:, 4], predictions[:, 4], 'kx',
               label = 'rescaled magnification')

ax[0, 0].set_xlabel('true')
ax[0, 0].set_ylabel('pred')
ax[0, 1].set_xlabel('true')
ax[0, 1].set_ylabel('pred')
ax[0, 2].set_xlabel('true')
ax[0, 2].set_ylabel('pred')
ax[1, 0].set_xlabel('true')
ax[1, 0].set_ylabel('pred')
ax[1, 1].set_xlabel('true')
ax[1, 1].set_ylabel('pred')

ax[0, 0].axis('equal')
ax[0, 1].axis('equal')
ax[0, 2].axis('equal')
ax[1, 0].axis('equal')
ax[1, 1].axis('equal')



ax[0, 0].set_title('rescaled vel-dispersion')
ax[0, 1].set_title('rescaled ellipticity')
ax[0, 2].set_title('rescaled orientation')
ax[1, 0].set_title('rescaled redshift')
ax[1, 1].set_title( 'rescaled magnification')

ax[1, 2].set_visible(False)



plt.show()


#####################################


#
#
#
#
# from keras.wrappers.scikit_learn import KerasClassifier
# from keras.utils import np_utils
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import KFold
# from sklearn.preprocessing import LabelEncoder
# from sklearn.pipeline import Pipeline
#
#
# estimator = KerasClassifier(build_fn=create_model, epochs=20, batch_size= 16, verbose=1)
# kfold = KFold(n_splits=10, shuffle=True, random_state=4)
# results = cross_val_score(estimator, test_data, test_target, cv=kfold)
# print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))





