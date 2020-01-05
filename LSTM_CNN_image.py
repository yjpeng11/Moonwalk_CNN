'''
D(x, H(x)) H(x) E(x) G(z)
3D for H(x)
VAE for E and G
'''
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
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers import Conv3D, MaxPooling3D
from keras.layers.merge import concatenate
from keras.models import Model, Sequential
from keras import backend as K
from keras import objectives, initializers
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import SGD, RMSprop
from keras.applications.vgg16 import VGG16
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.preprocessing import sequence


def label2vec(label, num_classes):
    """convert label into a one-hot vector"""
    vec = np.zeros((1, num_classes))
    vec[0, label] = 1
    return vec


def getBatchGenerator(features, labels0, args, is_shuffle=True):
    cur_batch_index = 0
    N = len(features)
    while 1:
        labels = labels0
        files = features
        if cur_batch_index == 0 and is_shuffle:
            index_shuf = range(N)
            random.shuffle(index_shuf)
            files = []
            labels = []
            for i in index_shuf[:len(index_shuf)]:
                files.append(features[i])
                labels.append(labels0[i])
        start_id = cur_batch_index
        end_id = min(cur_batch_index + args.batch_size, N)

        X = []
        Y = []
        for i in range(end_id)[start_id:]:
            X.append(files[i])
            if args.returnseq:
                Yrep = [label2vec(labels[i], args.num_classes)]*args.max_len
            else:
                Yrep = [label2vec(labels[i], args.num_classes)]

            Y.append([Yrep])
        # print len(X[-1])

        cur_batch_index = end_id
        if cur_batch_index >= N:
            cur_batch_index = 0

        X = sequence.pad_sequences(X, maxlen=args.max_len, dtype='float32', padding='post')
        # Y = sequence.pad_sequences(Y, maxlen=args.max_len, padding='post')
        Y = np.squeeze(np.array(Y))
        # print X.shape

        yield X, Y


def my_init(shape, dtype=None):
    """customized initialization"""
    return K.random_normal(shape, mean=0.0, stddev=0.01, dtype=dtype)


def build_model(args, weights_path_LSTM=None,output_feature = False):
    """build flow model"""
    if args.returnseq:
        sequenceout = True
    else:
        sequenceout = False
    model = Sequential()
    video = Input(shape=(args.max_len, 4096))

    if sequenceout:
        model_seq = TimeDistributed(model, name='td_0')(video)
        encoder = LSTM(args.nodes, return_sequences=True, name='lstm_1')(model_seq)
        dropout = TimeDistributed(Dropout(0.75, name='dp_1'), name='td_1')(encoder)
        output = TimeDistributed(Dense(args.num_classes, activation='softmax', init=my_init, name='dense_6'), name='td_2')(
        dropout)
    else:
        model_seq = TimeDistributed(model, name='td_0')(video)
        # encoder = LSTM(args.nodes, return_sequences=True, name='lstm_1')(model_seq)
        encoder2 = LSTM(args.nodes, return_sequences=False, name='lstm_2')(model_seq)
        dropout = Dropout(0.75, name='dp_1')(encoder2)
        output = Dense(args.num_classes, activation='softmax', init=my_init, name='dense_6')(dropout)

    # model = Model(input=video, output=[output, encoder2])
    if output_feature:
        model = Model(input=video, output= [output, encoder2])
    else:
        model = Model(input=video, output=output)

    if weights_path_LSTM:
        model.load_weights(weights_path_LSTM, by_name=True)
    print model.summary()
    return model


def compute_confusion_matrix(softmax_output, Y_GT, num_classes):
    Y_bar = np.argmax(softmax_output, axis=1)
    # Y_bar = Y_bar[:,-1]
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
# def compute_confusion_matrix(softmax_output, Y_test, time_stamps, max_len, nLabels):

if __name__ == '__main__':
    # args
    parser = argparse.ArgumentParser(description='CNN for image')
    parser.add_argument('--seed', type=int, default=123, help='Random seed')
    parser.add_argument('--num-classes', type=int, default=15, help='Number of output labels')
    parser.add_argument('--max-nb', type=int, default=5000, help='Maximum number of instances of a class')
    parser.add_argument('--batch-size', type=int, default=2048, help='Batch size')
    parser.add_argument('--dropout-rate', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--lr', type=float, default=.001, help='Learning rate')
    parser.add_argument('--num-epochs', type=int, default=100, help='Number of training epoches')
    parser.add_argument('--img-rows', type=int, default=224, help='Image height')
    parser.add_argument('--img-cols', type=int, default=224, help='Image width')
    parser.add_argument('--img-chns', type=int, default=3, help='Image channels')
    parser.add_argument('--reduce-factor', type=int, default=1, help='Dataset size reducing factor')
    parser.add_argument('--mode', type=int, default=1, help='Running mode, 0 - training, 1 - testing')
    parser.add_argument('--nodes', type=int, default=3000, help='Nodes')
    parser.add_argument('--max_len', type=int, default=15, help='Max length')
    parser.add_argument('--returnseq', type=int, default=0, help='Max length')
    args = parser.parse_args()

    WORKSPACE = '/mnt/Data/2/ActionCNN_simulation/two_stream_LSTM/hmdb51'

    # load data
    train_data_Y_path = os.path.join(WORKSPACE, 'flow_train_data_Y.pik')
    Y_train = cPickle.load(open(train_data_Y_path))
    Y_train = Y_train['labels1']

    train_feature_path = os.path.join(WORKSPACE, 'flow_train_data_feature.pik')
    train_feature_all = cPickle.load(open(train_feature_path))
    train_feature = train_feature_all['feature']

    train_counts = [0] * args.num_classes
    for label in Y_train:
        train_counts[label] += 1

    val_data_Y_path = os.path.join(WORKSPACE, 'flow_test_data_Y.pik')
    Y_val = cPickle.load(open(val_data_Y_path))
    Y_val = Y_val['labels1']

    test_feature_path = os.path.join(WORKSPACE, 'flow_test_data_feature.pik')
    test_feature_all = cPickle.load(open(test_feature_path))
    test_feature = test_feature_all['feature']

    test_counts = [0] * args.num_classes
    for label in Y_val:
        test_counts[label] += 1

    print "train class counts:", train_counts
    print "train feature length:", (len(train_feature))
    print "Y train length:", (len(Y_train))
    print "val class counts:", test_counts
    print "test feature length:", len(test_feature)
    print "Y val length:", len(Y_val)

    # set up callbacks
    CHECKPOINT_DIR = WORKSPACE + '/checkpoints_image/'
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
    CHECKPOINT_PATH = CHECKPOINT_DIR + 'weights_dropout{0}.hdf5'.format(args.dropout_rate)

    checkpointer = ModelCheckpoint(filepath=CHECKPOINT_PATH,
                                   monitor='val_loss',
                                   verbose=1,
                                   save_best_only=True,
                                   save_weights_only=True,
                                   mode='auto')
    early_stopper = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
    # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-5)
    # training or testing the network
    sgd = SGD(lr=args.lr, decay=0.000, momentum=0.9, nesterov=True)
    # sgd = RMSprop(lr=0.000001)
    steps_train = int(np.ceil(len(Y_train) / float(args.batch_size)))
    steps_val = int(np.ceil(len(Y_val) / float(args.batch_size)))
    if args.mode == 0:  # training
        model = build_model(args,CHECKPOINT_PATH)
        print model.summary()
        model.compile(loss='categorical_crossentropy',
                      optimizer=sgd,
                      metrics=['acc'])

        model.fit_generator(generator=getBatchGenerator(train_feature, Y_train, args, is_shuffle=True),
                            steps_per_epoch = steps_train, epochs = args.num_epochs,
                            validation_data = getBatchGenerator(test_feature, Y_val, args, is_shuffle = True),
                            validation_steps = steps_val,
                            callbacks = [checkpointer, reduce_lr, early_stopper])

        softmax_output = model.predict_generator(generator=getBatchGenerator(
            test_feature, Y_val, args, is_shuffle=False), steps=steps_val, verbose=1)
        score, ConfusionMat = compute_confusion_matrix(softmax_output, Y_val, args.num_classes)
        print 'ConfusionMat:', ConfusionMat
        print 'score:', score
        # np.save(WORKSPACE + '/conf_mat_val_test.npy', ConfusionMat)
    else:
        output_feature = 1
        model = build_model(args, CHECKPOINT_PATH, output_feature)
        # print model.summary()
        model.compile(loss='categorical_crossentropy',
                      optimizer=sgd,
                      metrics=['acc'])
        modeloutput = model.predict_generator(generator=getBatchGenerator(
            test_feature, Y_val, args, is_shuffle=False), steps=steps_val, verbose=1)
        LSTM_fea = modeloutput[1]
        softmax_output = modeloutput[0]
        print softmax_output[0].shape
        score, ConfusionMat = compute_confusion_matrix(softmax_output, Y_val, args.num_classes)
        print 'ConfusionMat:', ConfusionMat
        print 'score:', score
        # CNN_features_val = output_val[1]
        # np.save(WORKSPACE + '/feature_maps_image_val.npy', CNN_features_val)
        # np.save(WORKSPACE + '/conf_mat_val_test.npy', ConfusionMat)

        # output_train = model.predict_generator(generator=getBatchGenerator(
        #     X_train, Y_train, args, is_shuffle = False), steps = steps_train, verbose = 1)
        # CNN_features_train = output_train[1]
        # np.save(WORKSPACE + '/feature_maps_image_train.npy', CNN_features_train)
