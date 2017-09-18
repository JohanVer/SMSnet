# SMSnet
Model-evaluator of Publication : "SMSnet: Semantic Motion Segmentation using Deep Convolutional Neural Networks"

VIDEO

##Models:
The models can be downloaded here:
This repo contains the models trained on City-Kitti-Motion using EFS and without EFS
Link1 (With EFS)
Link2 (Without EFS)

## Test databases
The databases can be downloaded here:
Link1 (Cityscapes test images)
Link2 (KITTI test images)

Each compressed archiv includes 8 lmdb databases:

Labels:
IMAGES_LMDB0: Labels with 20m range
IMAGES_LMDB1: Labels with 40m range
IMAGES_LMDB2: Labels with 60m range
LABELS_LMDB:  Labels with inf range

Images:
IMAGES_LMDB3: Left image corresponding to label
IMAGES_LMDB4: Previous left image

Flow:
Note: The flow is centered at 128 and scaled by 1/6.4 in order to fit into the UC1 LMDB format
IMAGES_LMDB5: Flow with EFS 
IMAGES_LMDB6: Flow

##Prerequisites:
1. Caffe
2. CUDA
3. OPENCV
4. HDF5
5. PROTOBUF

##How to use:
1. Clone repo : "git clone https://github.com/JohanVer/SMSnet.git"
2. Go to folder and create build dir: "cd SMSnet && mkdir build && mkdir datasets && mkdir models"
3. Download KITTI dataset from the provided link and extract it into "datasets"
4. Download Models and extract them into "models"
4. Go to build folder: "cd build"
5. Compile: "cmake .." then "make"
6. run program : "./caffe_test_ex"
