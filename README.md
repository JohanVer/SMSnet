# SMSnet
Model-evaluator of Publication : "SMSnet: Semantic Motion Segmentation using Deep Convolutional Neural Networks"

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/oTL7l7ZuQpM/0.jpg)](https://www.youtube.com/watch?v=oTL7l7ZuQpM)

## Models
The models can be downloaded here: [Models](http://aisdatasets.informatik.uni-freiburg.de/smsnet/models.tar.gz)
This link contains the models trained on City-Kitti-Motion using EFS and without EFS on inf range and 40m range

## Test databases
The databases can be downloaded here:
- [KITTI Test Dataset](http://aisdatasets.informatik.uni-freiburg.de/smsnet/datasets/kitti.tar.gz)
- [Cityscapes Test Dataset](http://aisdatasets.informatik.uni-freiburg.de/smsnet/datasets/city.tar.gz)

Each compressed archiv includes 8 lmdb databases:

Labels:
- IMAGES_LMDB0: Labels with 20m range
- IMAGES_LMDB1: Labels with 40m range
- IMAGES_LMDB2: Labels with 60m range
- LABELS_LMDB:  Labels with inf range

Images:
- IMAGES_LMDB3: Left image corresponding to label
- IMAGES_LMDB4: Previous left image

Flow:
Note: The flow is centered at 128 and scaled by 1/6.4 in order to fit into the UC1 LMDB format
- IMAGES_LMDB5: Flow with EFS 
- IMAGES_LMDB6: Flow

### Raw Annotations
The full moving car/ static car annotations for the training and validation sets can be downloaded here: [Annotations](http://aisdatasets.informatik.uni-freiburg.de/smsnet/cityscapes_motion_labels.tar.gz)

## Prerequisites
1. Caffe
2. CUDA
3. OPENCV
4. HDF5
5. PROTOBUF

## How to use

### Build caffe
This repo comes with a modified version of caffe
1. Clone repo
2. Go into extern/modcaffe
3. Create a build folder "mkdir build && cd build"
4. Compile: "cmake .." then "make -j8"

### SMSnet evaluator
1. Go to repo and create build dir: "mkdir build && mkdir datasets && mkdir models"
2. Download KITTI dataset from the provided link and extract it into "datasets"
3. Download Models and extract them into "models"
4. Go to build folder: "cd build"
5. Compile: "cmake .." then "make"
6. run program : "./caffe_test_ex" . The network and the KITTI database should load. The network predictions and the GT is visualized. To get the next prediction just press Enter.
   If you want to change the Dataset take a look in the code (main.cpp)

## Troubleshooting
If you get : "caffe/proto/caffe.pb.h: No such file or directory" try to go in your caffe dir (extern/modcaffe) and type:
1. protoc src/caffe/proto/caffe.proto --cpp_out=.
2. mkdir include/caffe/proto
3. mv src/caffe/proto/caffe.pb.h include/caffe/proto

# Cite
If you use this code or the datasets please make sure that you cite the following papers:

~~~~ 
@InProceedings{Vertens17Icra,
author = {Johan Vertens, Abhinav Valada and Wolfram Burgard},
title = {SMSnet: Semantic Motion Segmentation
using Deep Convolutional Neural Networks},
year = 2017,
month = sept,
url = {https://github.com/JohanVer/SMSnet},
address = {Vancouver, Canada}
}
~~~~

~~~~
@article{DBLP:journals/corr/CordtsORREBFRS16,
  author    = {Marius Cordts and
               Mohamed Omran and
               Sebastian Ramos and
               Timo Rehfeld and
               Markus Enzweiler and
               Rodrigo Benenson and
               Uwe Franke and
               Stefan Roth and
               Bernt Schiele},
  title     = {The Cityscapes Dataset for Semantic Urban Scene Understanding},
  journal   = {CoRR},
  volume    = {abs/1604.01685},
  year      = {2016},
  url       = {http://arxiv.org/abs/1604.01685},
  timestamp = {Wed, 07 Jun 2017 14:41:02 +0200},
  biburl    = {http://dblp.uni-trier.de/rec/bib/journals/corr/CordtsORREBFRS16},
  bibsource = {dblp computer science bibliography, http://dblp.org}
}
~~~~

~~~~
@ARTICLE{Geiger2013IJRR,
  author = {Andreas Geiger and Philip Lenz and Christoph Stiller and Raquel Urtasun},
  title = {Vision meets Robotics: The KITTI Dataset},
  journal = {International Journal of Robotics Research (IJRR)},
  year = {2013}
}
~~~~
