#ifndef CLASSIFY_H
#define CLASSIFY_H

#include <caffe/caffe.hpp>

#ifdef DEPTH_COLOR
#include <opencv2/face/facerec.hpp>
#endif

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <boost/timer/timer.hpp>

namespace classify{

class PredictionStatistic{
public:
    PredictionStatistic(){
        true_pos_= 0;
        false_pos_ = 0;
        false_neg_ = 0;
        iou_ = 0;
        precision_ = 0;
        recall_ = 0;
    }

    unsigned int true_pos_;
    unsigned int false_pos_;
    unsigned int false_neg_;
    double iou_;
    double precision_;
    double recall_;
};

class Classifier {
public:
    Classifier(const std::string& model_file,
               const std::string& trained_file,
               const std::map<unsigned char,std::string> &mean_files);

    std::vector<std::vector<cv::Mat> > Classify(const std::vector<cv::Mat> &img, std::vector<std::string> req_layer);

    int calcMeanStatistic(const std::map<size_t, PredictionStatistic> &stat, std::set<size_t> &ignore_list, double &mean_iou, double &mean_prec, double &mean_rec, double &mean_fp, double &mean_fn);

    void calculateStatistics(const cv::Mat &net_im, const cv::Mat &gt, std::set<size_t> &ignore_list, std::map<size_t,PredictionStatistic> &statistics, double &acc, double &error);

    void createColorCodedLabels(const cv::Mat &labels, cv::Mat &color_coded, std::map<unsigned char, cv::Vec3b> color_map);

    uint getNumberOfClasses(){
        return num_output_classes_;
    }

    cv::Size getInputSize(){
        return input_geometry_;
    }

    void Argmax(const std::vector<cv::Mat> &channels, cv::Mat &argmax);

    cv::Mat depthImage(cv::Mat &in);

    cv::Mat flowImage(cv::Mat &in);

private:
    void SetMean(const cv::Scalar &mean, uint8_t index);

    void SetMean(const std::map<unsigned char, std::string> &mean_file);

    std::vector<std::vector<cv::Mat> > Predict(const std::vector<cv::Mat> &img, std::vector<std::string> req_layers);

    void WrapInputLayer(std::vector<cv::Mat>* input_channels, size_t num_input);

    void WrapOutputLayer(std::vector< cv::Mat> *output_channels, std::string l_name);

    void Preprocess(const cv::Mat& img,
                    std::vector<cv::Mat>* input_channels, uint8_t input_index);


private:
    std::shared_ptr<caffe::Net<float> > net_;
    cv::Size input_geometry_;
    int num_channels_;
    uint num_output_classes_;
    std::map<uint8_t, cv::Mat> mean_;
    cv::Mat label_image_;
    std::vector<std::string> labels_;
};


}


#endif // CLASSIFY_H
