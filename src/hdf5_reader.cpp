#include "../include/hdf5_reader.h"

bool hdf5_reader::loadPaths(const boost::filesystem::path &path_to_paths_file, std::vector<boost::filesystem::path> &list_of_paths){
    std::ifstream path_file;
    path_file.open(path_to_paths_file.string());

    std::string line;
    while (std::getline(path_file, line))
    {
        list_of_paths.push_back(boost::filesystem::path(line));
    }

    if(list_of_paths.size()){
        std::cout << "Loaded " << list_of_paths.size() << " file paths\n";
        return true;
    }
    else{
        std::cerr << "list size of path file is zero...\n";
        return false;
    }
}

hdf5_reader::hdf5_reader(const boost::filesystem::path hdf5_paths, const std::string &dataset_name):
    current_idx(0),
    blob_top_data_(new caffe::Blob<float>())
{
    blob_top_vec_.push_back(blob_top_data_);

    std::cout << "Using sample HDF5 data file " << hdf5_paths.string() << std::endl;

    param.add_top(dataset_name);
    std::cout << "Searching for dataset: " << dataset_name << " in hdf5 file ..." << std::endl;
    hdf5_data_param = param.mutable_hdf5_data_param();
    int batch_size = 1;
    hdf5_data_param->set_batch_size(batch_size);
    hdf5_data_param->set_source(hdf5_paths.string());


    layer_ = std::unique_ptr<caffe::HDF5DataLayer<float> > (new caffe::HDF5DataLayer<float>(param));
    layer_->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);


    height = this->blob_top_data_->height();
    width = this->blob_top_data_->width();
    num_cols = this->blob_top_data_->channels();

    std::cout << "Channels: " << num_cols << " Height:" << height << " Width: " << width  << std::endl;
}

cv::Mat hdf5_reader::nextImage(){

    layer_->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    cv::Mat out;
    if(num_cols == 2){
        out = cv::Mat(height, width, CV_32FC2);
    }else if(num_cols == 1){
        out = cv::Mat(height, width, CV_32FC1);
    }

    for (int j = 0; j < num_cols; ++j) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                int idx = (j * height * width +
                           h * width + w);
                if(num_cols == 2){
                    cv::Vec2f &p_m = out.at<cv::Vec2f>(h, w);
                    p_m[j] = this->blob_top_data_->cpu_data()[idx];
                }else if(num_cols ==1){
                    float &p_m = out.at<float>(h, w);
                    p_m = this->blob_top_data_->cpu_data()[idx];
                }
            }
        }
    }
    return out;
}
