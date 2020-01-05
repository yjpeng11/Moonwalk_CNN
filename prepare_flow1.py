import os
import cPickle
import scipy.io as sio
import numpy as np

NUM_CLASSES = 3
name2id = {'causal':1,'moonwalk':2, 'backward':3}  #
data_path = "/mnt/Data/4/ActionSimulation2/Moonwalk/causal/causal_moonwalk_backward/LSTM/test01_causal_moonwalk_backward_leftright.mat"

def label2vec(label, num_classes):
    """convert label into a one-hot vector"""
    vec = np.zeros((1, num_classes))
    vec[0, label] = 1
    return vec


def get_file_list(folder_dir):
    """get file list in a folder"""
    return [os.path.join(folder_dir, f) for f in os.listdir(folder_dir) \
            if os.path.isfile(os.path.join(folder_dir, f))] 


def prepare_data(X_data, Y_data, num_classes, name2id):
    """read, reconstruct & return data"""
    if len(X_data) != len(Y_data):
        print "Error: different number of instances in X and Y!"
        return None
    X_image = []
    X_flow_u = []
    X_flow_v = []
    video_lengths = []
    Y = []
    video_ids = []
    cur_label_id = 0
    #for video_id, x, y in enumerate(zip(X_data, Y_data)):
    for x, y in zip(X_data, Y_data):
        x_image_prefix = x[1][0]
        x_flow_u_prefix = x[2][0]
        x_flow_v_prefix = x[3][0]
        x_image_path = get_file_list(x_image_prefix)
        x_image_path.sort()
        #print x_image_path
        x_flow_u_path = get_file_list(x_flow_u_prefix)
        x_flow_v_path = get_file_list(x_flow_v_prefix)
        x_flow_u_path.sort()
        x_flow_v_path.sort()
        length = min(len(x_image_path), len(x_flow_u_path))
        cnt = length / 10
        for t in xrange(0, cnt * 10, 10):
            X_image.append(x_image_path[t:(t+10)])
            #print "----------------------"
            #print x_image_path[t:(t+10)]
            X_flow_u.append(x_flow_u_path[t:(t+10)])
            X_flow_v.append(x_flow_v_path[t:(t+10)])
        video_lengths.append(cnt)		
        label_name = y[0][0]
        if label_name not in name2id:
            print "error: can not find label", label_name 
            return None
        label = name2id[label_name] - 1
		# Y += [label2vec(label, num_classes)] * length
        Y += ([label] * cnt)
        #video_ids += ([video_id] * cnt)
        #print "-----------------------------------------------------"

    print len(X_image), len(X_flow_u), len(X_flow_v), len(video_lengths), sum(video_lengths), len(Y)

    return {'image': X_image,
            'flow_u': X_flow_u,
            'flow_v': X_flow_v, 
            'lengths': video_lengths, 
		    'labels': Y}
            #'video_ids': video_ids}

data = sio.loadmat(data_path)
train_data = prepare_data(data['trainx'][0], data['trainy'][0], NUM_CLASSES, name2id)
cPickle.dump(train_data, open("./hmdb51/flow_train_data.pik", "wb"), protocol = 2)
test_data = prepare_data(data['testx'][0], data['testy'][0], NUM_CLASSES, name2id)
cPickle.dump(test_data, open("./hmdb51/flow_test_data.pik", "wb"), protocol = 2)