import os
import cPickle
import scipy.io as sio
import numpy as np


def label2vec(label, num_classes):
    """convert label into a one-hot vector"""
    vec = np.zeros((1, num_classes))
    vec[0, label] = 1
    return vec


def get_file_list(folder_dir):
    """get file list in a folder"""
    return [os.path.join(folder_dir, f) for f in os.listdir(folder_dir) \
            if os.path.isfile(os.path.join(folder_dir, f))] 


def prepare_feature(X_feat_list,basedir):
    """read, reconstruct & return data"""
    X_feat = []
    video_lengths = []
    video_ids = []
    cur_label_id = 0
    # chunk = 30;
    #for video_id, x, y in enumerate(zip(X_data, Y_data)):
    for x in (X_feat_list):
        path = basedir +'/' +x
        X = np.load(path)
        X_feat.append(X)

    print len(X_feat)
    return {'feature': X_feat}

## for image features
feature_path = "/mnt/Data/2/ActionCNN_simulation/two_stream_LSTM/hmdb51/"
train_list = os.listdir(feature_path + 'indivi_feat_train_image')
train_list.sort()
test_list = os.listdir(feature_path + 'indivi_feat_test_image')
test_list.sort()
print 'image CNN feature'
print len(train_list), len(test_list)

train_data = prepare_feature(train_list, feature_path + 'indivi_feat_train_image')
cPickle.dump(train_data, open("./hmdb51/train_data_feature_image.pik", "wb"), protocol = 2)
test_data = prepare_feature(test_list, feature_path + 'indivi_feat_test_image')
cPickle.dump(test_data, open("./hmdb51/test_data_feature_image.pik", "wb"), protocol = 2)

## for flow features
train_list = os.listdir(feature_path + 'indivi_feat_train_flow')
train_list.sort()
test_list = os.listdir(feature_path + 'indivi_feat_test_flow')
test_list.sort()
print 'flow CNN feature'
print len(train_list), len(test_list)

train_data = prepare_feature(train_list, feature_path + 'indivi_feat_train_flow')
cPickle.dump(train_data, open("./hmdb51/train_data_feature_flow.pik", "wb"), protocol = 2)
test_data = prepare_feature(test_list, feature_path + 'indivi_feat_test_flow')
cPickle.dump(test_data, open("./hmdb51/test_data_feature_flow.pik", "wb"), protocol = 2)

## for fusion features
train_list = os.listdir(feature_path + 'indivi_feat_train_fusion')
train_list.sort()
test_list = os.listdir(feature_path + 'indivi_feat_test_fusion')
test_list.sort()
print 'flow CNN feature'
print len(train_list), len(test_list)

train_data = prepare_feature(train_list, feature_path + 'indivi_feat_train_fusion')
cPickle.dump(train_data, open("./hmdb51/train_data_feature_fusion.pik", "wb"), protocol = 2)
test_data = prepare_feature(test_list, feature_path + 'indivi_feat_test_fusion')
cPickle.dump(test_data, open("./hmdb51/test_data_feature_fusion.pik", "wb"), protocol = 2)
