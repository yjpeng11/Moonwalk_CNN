# Weak integration of form and motion in two-stream CNNs for action recognition

Tensorflow implementation of "Two-stream Convolutional Neural Networks Fails to Understand Motion Congruency in Human Actions"

## Publication
#### Weak integration of form and motion in two-stream CNNs for action recognition
Yujia Peng, Tianmin Shu, and Hongjing Lu

## Getting started

Clone this repository 
```
git clone https://github.com/yjpeng11/Moonwalk_CNN.git
```

### Prerequisites
* Linux Ubuntu 16.04
* Python 2 and 3
* NVIDIA GPU + CUDA 9.0

### Step 1: Extracting model inputs through OpenCV

The OpenCV script extracts static image frames from videos and further generates optical flow images.

Create an environment with all packages from requirements_opencv.txt installed.
```
python -m virtualenv opencv
source cnn/bin/activate
pip install -r requirements_opencv.txt
```

To extract static image frames and optical flow data, run:
```
python3 opencv_opticalflow.py
```

### Step 2: Preprocessing model inputs

The script "folder2list_leftright.m" generates a .txt file with a list of videos.

The script "make_test_mat2_leftright.m" takes the .txt file as input to generate a .mat file with directories of the saved static images 
and optical flow images.

To process the list of files in .mat into python-friendly format, run:
```
python prepare_flow1.py
python prepare_flow1_label.py
```
The name2id and data_path at the beginning of scripts need to be changed accordingly.

### Step 3: Training and testing CNN models

The CNN scripts implement the spatial CNN of appearance, the temporal CNN of motion, and the two-stream CNN.
```
python -m virtualenv cnn
source cnn/bin/activate
pip install -r requirements_cnn.txt
```

The running of CNN scripts requires an environment with python 2.7. Create an environment with all packages from requirements_cnn.txt installed (Note: please double check the CUDA version on your machine).

#### Step 3.1: Training and testing CNN models

To train the spatial CNN:
```
python CNN_image.py --mode 0
```
To train the temporal CNN:
```
python CNN_flow.py --mode 0
```
To train the two-stream CNN:
```
python CNN_flow.py --mode 0
```

Change mode 0 to mode 1 for testing. The directory of WORKSPACE need to be changed accordingly.

#### Step 3.2: Training and testing the softmax layer of CNN models
To train the spatial CNN:
```
python CNN_image_freeze.py --mode 0
```
To train the temporal CNN:
```
python CNN_flow_freeze.py --mode 0
```
To train the two-stream CNN:
```
python CNN_fusion_freeze.py --mode 0
```

Change mode 0 to mode 1 for testing.

### Step 4: Training and testing LSTM models

#### Step 4.1: prepare data

#### Step 4.2: Extract CNN features for LSTM
To generate CNN image features of each individual video clip and save them under folders indivi_feat_train/test
```
python  LSTM_CNNfeature_image_indivi
```
To generate CNN flow features of each individual video clip and save them under folders indivi_feat_train/test
```
python  LSTM_CNNfeature_flow_indivi
```
To generate two-stream CNN features of each individual video clip and save them under folders indivi_feat_train/test
```
python  LSTM_CNNfeature_fusion_indivi 
```
To combine all features into a single input file
```
python  prepare_image_flow_feature
```

#### Step 4.3: Training and testing CNN models

To train the spatial CNN:
```
python LSTM_CNN_image.py --mode 0
```
To train the temporal CNN:
```
python LSTM_CNN_flow.py --mode 0
```
To train the two-stream CNN:
```
python LSTM_CNN_fusion.py --mode 0
```

Change mode 0 to mode 1 for testing.

#### Step 4.4: Training and testing the softmax layer of CNN models
To train the spatial CNN:
```
python LSTM_CNN_image_freeze.py --mode 0
```
To train the temporal CNN:
```
python LSTM_CNN_flow_freeze.py --mode 0
```
To train the two-stream CNN:
```
python LSTM_CNN_fusion_freeze.py --mode 0
```

Change mode 0 to mode 1 for testing.
