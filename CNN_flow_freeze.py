import numpy as np
import matplotlib.pyplot as plt
import h5py
import cPickle
import random
import os
import argparse
import progressbar
from scipy.stats import norm
from scipy.misc import imresize
from numpy import linalg as LA
from multiprocessing import Process
from skimage.io import imread, imshow
from skimage.transform import resize
from time import sleep

from keras.layers import Input, Dense, Lambda, Flatten, Reshape, Dropout, ZeroPadding2D
from keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, TimeDistributed
from keras.layers import Conv3D, MaxPooling3D
from keras.layers.merge import concatenate
from keras.models import Model, Sequential
from keras import backend as K
from keras import objectives, initializers
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import SGD, RMSprop
from keras.applications.vgg16 import VGG16

#%%
def label2vec(label, num_classes):
    """convert label into a one-hot vector"""
    vec = np.zeros((1, num_classes))
    vec[0, label] = 1
    return vec


def preprocess_image_batch(u_paths, v_paths, img_size=None, crop_size=None, color_mode="rgb", out=None):
    img_list = []
    """preprocess a batch of optical flow images"""
    for u_path_stack, z_path_stack in zip(u_paths, v_paths):
        img_stack_list = []
        path_stack = u_path_stack + z_path_stack
        for im_path in path_stack:
            img = imread(im_path, mode='L')
            if img_size:
                img = imresize(img,img_size)
            img = img.astype('float32')
            # We normalize the colors (in RGB space) with the empirical means on the training set
            # img[:, :] -= 123.68 * 0.299 + 116.779 * 0.587 + 103.939 * 0.114 
            img[:, :] -= 128.0
            #print "img shape", img.shape
            img_stack_list.append(img)
 
        img_stack = np.stack(img_stack_list, axis=-1)
        #img_stack.transpose((1, 2, 0))
        # print "img stack shape", img_stack.shape
        img_list.append(img_stack)

    try:
        #print "img_list shape", np.array(img_list).shape
        #print u_path_stack
        img_batch = np.stack(img_list, axis=0)
        #print "img batch shape", img_batch.shape
        #raw_input("")
    except:
        raise ValueError('when img_size and crop_size are None, images'
                ' in image_paths must have the same shapes.')
 
    if out is not None and hasattr(out, 'append'):
        out.append(img_batch)
    else:
        return img_batch


def my_init(shape, dtype=None):
    """customized initialization"""
    return K.random_normal(shape, mean=0.0, stddev=0.01, dtype=dtype)

def build_model(args,
                weights_path = None,
                output_feature = False):
    """build flow model"""
    X_flow  = Input(shape = (7, 7, 512), name = 'input_fusion')
    fusion = Conv2D(512, (7, 7), activation = 'relu', padding = 'same', 
                    kernel_initializer = my_init, name = 'fusion')(X_flow)
    flatten = Reshape((512 * 7 * 7, ))(fusion)
    dense1 = Dense(4096, trainable=False, activation = 'relu', kernel_initializer = my_init, name = 'dense_1')(flatten)
    dp1 = Dropout(args.dropout_rate, name = 'dp_1')(dense1)
    dense2 = Dense(4096, trainable=False, activation = 'relu', kernel_initializer = my_init, name = 'dense_2')(dp1)
    dp2 = Dropout(args.dropout_rate, name = 'dp_2')(dense2)
    softmax = Dense(args.num_classes, trainable=True, activation = 'softmax', kernel_initializer = my_init, name = 'new_dense_3')(dp2)

    CNN_features = Reshape((7, 7, 512))(flatten)
    if output_feature:
        model = Model(inputs = X_flow, outputs = [softmax, dense2])
    else:
        model = Model(inputs = X_flow, outputs = softmax)

    if weights_path:
        model.load_weights(weights_path, by_name = True)

    return model


def compute_confusion_matrix(softmax_output, Y_GT, num_classes):
    Y_bar = np.argmax(softmax_output, axis=1)
    # Y_GT = np.argmax(Y_test, axis=1)
    # print Y_bar
    N_test = len(Y_GT)
    ConfusionMat = np.zeros((num_classes, num_classes), dtype=np.float32)
    score = float(sum([1 for i in range(N_test) if Y_GT[i] == Y_bar[i]])) / N_test
    for i in range(N_test):
        ConfusionMat[Y_GT[i], Y_bar[i]] += 1.0
    print ConfusionMat
    for label in range(num_classes):
        ConfusionMat[label, :] /= sum(ConfusionMat[label, :])
    return (score, ConfusionMat)

def load_features(path, args):
    """load features"""
    X = np.load(path)
    return X[::args.reduce_factor]

def getBatchGenerator_y(y, args):
    """sample a batch"""
    labels0 = list(y)
    cur_batch_index = 0
    nb_batch = 0

    labels = labels0
    while 1:
        start_id = 0
        end_id = len(y)

        Y = [label2vec(label, args.num_classes) for label in labels[start_id:end_id]]
        Y = np.vstack(Y)

        return Y
#%%
if __name__ == '__main__':
    # args
    a=5
    parser = argparse.ArgumentParser(description='CNN for image')
    parser.add_argument('--seed', type=int, default=123, help='Random seed')
    parser.add_argument('--num-classes', type=int, default=2, help='Number of output labels')
    parser.add_argument('--max-nb', type=int, default=5000, help='Maximum number of instances of a class')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--dropout-rate', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num-epochs', type=int, default=100, help='Number of training epoches')
    parser.add_argument('--img-rows', type=int, default=224, help='Image height')
    parser.add_argument('--img-cols', type=int, default=224, help='Image width')
    parser.add_argument('--img-chns', type=int, default=20, help='Image channels')
    parser.add_argument('--reduce-factor', type=int, default=1, help='Dataset size reducing factor')
    parser.add_argument('--mode', type=int, default=1, help='Running mode, 0 - training, 1 - testing')
    args = parser.parse_args()

    WORKSPACE = '/mnt/Data/2/ActionCNN_simulation/two_stream_LSTM/hmdb51'
    
    # load data
    train_data_path = os.path.join(WORKSPACE, 'flow_train_data.pik')
    train_data = cPickle.load(open(train_data_path))
    X_u_train = train_data['flow_u'][::args.reduce_factor]
    X_v_train = train_data['flow_v'][::args.reduce_factor]

    #totalsplit = 4
    #split = 1
    #X_u_train = X_u_train[((split-1)*len(X_u_train)//totalsplit):(split*len(X_u_train)//totalsplit)]
    #X_v_train = X_v_train[((split-1)*len(X_v_train)//totalsplit):(split*len(X_v_train)//totalsplit)]

    print(len(X_u_train))

    Y_train = train_data['labels'][::args.reduce_factor]
    #Y_train = Y_train[((split-1)*len(Y_train)//totalsplit):(split*len(Y_train)//totalsplit)]
    print(len(Y_train))

    counts = [0] * args.num_classes
    for label in Y_train:
        counts[label] += 1
    print "train class counts:", counts
    Y_train1 = getBatchGenerator_y(Y_train, args)

    val_data_path = os.path.join(WORKSPACE, 'flow_test_data.pik')
    val_data = cPickle.load(open(val_data_path))
    X_u_val = val_data['flow_u'][::args.reduce_factor]
    X_v_val = val_data['flow_v'][::args.reduce_factor]
    Y_val = val_data['labels'][::args.reduce_factor]
    counts = [0] * args.num_classes
    for label in Y_val:
        counts[label] += 1
    print "val class counts:", counts
    Y_val1 = getBatchGenerator_y(Y_val, args)

    X_flow_train_path = os.path.join(WORKSPACE, 'feature_maps_flow_train.npy')
    X_flow_train = load_features(X_flow_train_path, args)
    
    X_flow_val_path = os.path.join(WORKSPACE, 'feature_maps_flow_val.npy')
    X_flow_val = load_features(X_flow_val_path, args)
    X_flow_val = np.squeeze(X_flow_val)  # YP

    # set up callbacks
    CHECKPOINT_DIR = WORKSPACE + '/checkpoints_flow/'
    if not os.path.exists(CHECKPOINT_DIR):
      os.makedirs(CHECKPOINT_DIR)
    CHECKPOINT_PATH = CHECKPOINT_DIR + 'weights_dropout{0}.hdf5'.format(args.dropout_rate)

    checkpointer = ModelCheckpoint(filepath = CHECKPOINT_PATH, 
                                   monitor='val_loss', 
                                   verbose = 1,
                                   save_best_only=True, 
                                   save_weights_only=True, 
                                   mode='auto')
    early_stopper = EarlyStopping(monitor='val_loss', patience = 10, verbose = 0, mode = 'auto')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor = 0.5, patience = 3, min_lr = 1e-6)

    # training or testing the network
    sgd = SGD(lr = args.lr, decay = 0.000, momentum = 0.9, nesterov = True)
    steps_train = int(np.ceil(len(Y_train) / float(args.batch_size)))
    steps_val = int(np.ceil(len(Y_val) / float(args.batch_size)))
    if args.mode == 0: # training
        model = build_model(args,CHECKPOINT_PATH)
        #print model.summary()
        model.compile(loss = 'categorical_crossentropy', 
                      optimizer = sgd,
                      metrics = ['acc'])

        model.fit(X_flow_train,Y_train1,
                  epochs = args.num_epochs, 
                  batch_size = args.batch_size,
                  verbose = 1,
                  validation_data = (X_flow_val, Y_val1),
                  callbacks = [checkpointer, early_stopper, reduce_lr])
        model.load_weights(CHECKPOINT_PATH)
        
        softmax_output = model.predict([X_flow_val], verbose = 1)
        score, ConfusionMat = compute_confusion_matrix(softmax_output, Y_val, args.num_classes)
        print 'ConfusionMat:',ConfusionMat
        print 'score:', score
        np.save(WORKSPACE + '/conf_mat_val_flow.npy', ConfusionMat)
        #Confusionmat = np.load('sss')
    else: # testing
        model = build_model(args, CHECKPOINT_PATH, output_feature = True)
        print model.summary()
        model.compile(loss = 'categorical_crossentropy', 
                      optimizer = sgd,
                      metrics = ['acc'])
        output_val = model.predict([X_flow_val], verbose = 1)
        print output_val[0].shape
        score, ConfusionMat = compute_confusion_matrix(output_val[0], Y_val, args.num_classes)
        print 'ConfusionMat:',ConfusionMat
        print 'score:', score
        # CNN_features_val = output_val[1]
        # np.save(WORKSPACE + '/feature_maps_flow_val.npy', CNN_features_val)
        # np.save(WORKSPACE + '/conf_mat_val_flow.npy', ConfusionMat)
        # output_train = model.predict(X_flow_train, verbose = 1)
        # CNN_features_train = output_train[1]
        # np.save(WORKSPACE + '/feature_maps_flow_train.npy', CNN_features_train)
        
 
        

