#ifndef HDF5_READER_H
#define HDF5_READER_H

#include "caffe/util/hdf5.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/fstream.hpp>
#include <iostream>

#include "hdf5.h"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layers/hdf5_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

class hdf5_reader
{
public:
    hdf5_reader(const boost::filesystem::path hdf5_paths, const std::string &dataset_name);
    bool loadPaths(const boost::filesystem::path &path_to_paths_file, std::vector<boost::filesystem::path> &list_of_paths);
    cv::Mat nextImage();
private:
    std::vector<boost::filesystem::path> hdf5_paths_;


    size_t current_idx;
    std::vector<caffe::Blob<float>*> blob_bottom_vec_;
    std::vector<caffe::Blob<float>*> blob_top_vec_;
    caffe::Blob<float>* const blob_top_data_;
    caffe::LayerParameter param;

    caffe::HDF5DataParameter* hdf5_data_param;
    std::unique_ptr<caffe::HDF5DataLayer<float>> layer_;

    int width, height, num_cols;
};

#endif // HDF5_READER_H
