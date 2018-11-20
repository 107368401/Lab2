# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
# import keras's Sequential model
from keras.models import Sequential
from keras.layers import Dense,Dropout
from sklearn import preprocessing

map_characters = {0: 'abraham_grampa_simpson', 
                1: 'apu_nahasapeemapetilon', 
                2: 'bart_simpson', 
                3: 'charles_montgomery_burns', 
                4: 'chief_wiggum', 
                5: 'comic_book_guy', 
                6: 'edna_krabappel', 
                7: 'homer_simpson', 
                8: 'kent_brockman', 
                9: 'krusty_the_clown', 
                10: 'lenny_leonard',
                11: 'lisa_simpson', 
                12: 'marge_simpson', 
                13: 'mayor_quimby',
                14: 'milhouse_van_houten', 
                15: 'moe_szyslak', 
                16: 'ned_flanders', 
                17: 'nelson_muntz', 
                18: 'principal_skinner', 
                19: 'sideshow_bob'}

pic_size = 64
batch_size = 128
epochs = 200
num_classes = len(map_characters)
pictures_per_class = 1000
test_size = 0.15

def load_pictures(BGR):
    pics = []
    labels = []
    for k, char in map_characters.items():
        pictures = [k for k in glob.glob('./characters/%s/*' % char)]
        nb_pic = round(pictures_per_class/(1-test_size)) if round(pictures_per_class/(1-test_size))<len(pictures) else len(pictures)
        for pic in np.random.choice(pictures, nb_pic):
            a = cv2.imread(pic)
            if BGR:
                a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
            a = cv2.resize(a, (pic_size,pic_size))
            pics.append(a)
            labels.append(k)
    return np.array(pics), np.array(labels) 

def get_dataset(save=False, load=False, BGR=False):
    if load:
        h5f = h5py.File('dataset.h5','r')
        X_train = h5f['X_train'][:]
        X_test = h5f['X_test'][:]
        h5f.close()    

        h5f = h5py.File('labels.h5','r')
        y_train = h5f['y_train'][:]
        y_test = h5f['y_test'][:]
        h5f.close()
        
    else:
        X, y = load_pictures(BGR)
        
        y = keras.utils.to_categorical(y, num_classes)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        if save:
            h5f = h5py.File('dataset.h5', 'w')
            h5f.create_dataset('X_train', data=X_train)
            h5f.create_dataset('X_test', data=X_test)
            h5f.close()

            h5f = h5py.File('labels.h5', 'w')
            h5f.create_dataset('y_train', data=y_train)
            h5f.create_dataset('y_test', data=y_test)
            h5f.close()
            
    X_train = X_train.astype('float32') / 255.
    X_test = X_test.astype('float32') / 255.
    print("Train", X_train.shape, y_train.shape)
    print("Test", X_test.shape, y_test.shape)

    if not load:
        dist = {k:tuple(d[k] for d in [dict(Counter(np.where(y_train==1)[1])), dict(Counter(np.where(y_test==1)[1]))]) 
                for k in range(num_classes)}
        print('\n'.join(["%s : %d train pictures & %d test pictures" % (map_characters[k], v[0], v[1]) 
            for k,v in sorted(dist.items(), key=lambda x:x[1][0], reverse=True)]))
    return X_train, X_test, y_train, y_test

def create_model_six_conv(input_shape):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(256, (3, 3), padding='same')) 
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    return model, opt


filepath="weights_8conv_%s.hdf5" % time.strftime("%Y%m%d") 
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max')

history = model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.1,
          verbose=1,
          shuffle=True,
          callbacks=checkpoint)
score = model.evaluate(X_test, y_test, verbose=1)