#include "../include/classify.h"

using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

namespace classify{

Classifier::Classifier(const string& model_file,
                       const string& trained_file,
                       const std::map<unsigned char, std::string>& mean_files) {
#ifdef CPU_ONLY
    Caffe::set_mode(Caffe::CPU);
#else
    Caffe::set_mode(Caffe::GPU);
    Caffe::SetDevice(0);
#endif
    /* Load the network. */
    net_.reset(new Net<float>(model_file, TEST));
    net_->CopyTrainedLayersFrom(trained_file);

//    CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

    Blob<float>* input_layer = net_->input_blobs()[0];
    num_channels_ = input_layer->channels();
    CHECK(num_channels_ == 3) << "Input layer should have 3 channels.";
    input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

    Blob<float>* output_layer = net_->output_blobs()[0];
    num_output_classes_ = output_layer->channels();
    std::cout << "Network has : " << num_output_classes_ << " output classes...." << std::endl;
    /* Load the binaryproto mean file. */
    SetMean(mean_files);
    //SetMean(cv::Scalar(81.1208038, 102.18515,120.44704));

    std::cout << "Neural Network is setup..." << std::endl;

}

size_t countPixelBelongingToClass(const cv::Mat &img, size_t class_nr){
    size_t counter = 0;
    for(auto y = 0; y < img.rows; y++){
        for(auto x = 0; x < img.cols; x++){
            unsigned char pixel = img.at<uchar>(y, x);
            if(pixel == class_nr) counter++;
        }
    }
    return counter;
}

int Classifier::calcMeanStatistic(const std::map<size_t, PredictionStatistic> &stat, std::set<size_t> &ignore_list, double &mean_iou, double &mean_prec, double &mean_rec, double &mean_fp, double &mean_fn){
    mean_iou = 0.0;
    mean_prec = 0.0;
    mean_rec = 0.0;
    mean_fp = 0.0;
    mean_fn = 0.0;
    size_t valid_counter = 0;
    for(auto i = stat.begin(); i != stat.end(); i++){
        double iou = i->second.iou_;
        double precision =i->second.precision_;
        double recall =i->second.recall_;
        double fp =i->second.false_pos_;
        double fn =i->second.false_neg_;
        if(ignore_list.find(i->first) == ignore_list.end()){
            if(iou != -1){
                mean_iou += iou;
                //std::cout << "IOU " << i->first << " : " << iou << std::endl;
                valid_counter++;

                mean_prec += precision;
                //std::cout << "Precision: " << i->first << " : " << precision << std::endl;

                mean_rec += recall;
                //std::cout << "Recall: " << i->first << " : " << recall << std::endl;

                mean_fp += fp;

                mean_fn += fn;
            }
        }
    }

    if(valid_counter){
        mean_iou /= valid_counter;
        mean_rec /= valid_counter;
        mean_prec /= valid_counter;
        mean_fp /= valid_counter;
        mean_fn /= valid_counter;
    }
    return valid_counter;
}


void Classifier::calculateStatistics(const cv::Mat &net_im, const cv::Mat &gt, std::set<size_t> &ignore_list, std::map<size_t,PredictionStatistic> &statistics, double &acc, double &error){

    // Calculate accuracy and error
    acc = 0;
    error = 0;
    for(auto y = 0; y < net_im.rows; y++){
        for(auto x = 0; x < net_im.cols; x++){
            unsigned char net_pixel = net_im.at<uchar>(y, x);
            unsigned char gt_pixel = gt.at<uchar>(y, x);
            if(net_pixel == gt_pixel){
                acc++;
            }else{
                error++;
            }
        }
    }
    acc = acc / (net_im.cols * net_im.rows);
    error = error / (net_im.cols * net_im.rows);

    // Get true positive, false positive, false negative
    for(auto cl = 0; cl < num_output_classes_; cl++){
        size_t cl_in_img = 0;

        PredictionStatistic class_stat;
        for(auto y = 0; y < net_im.rows; y++){
            for(auto x = 0; x < net_im.cols; x++){
                unsigned char net_pixel = net_im.at<uchar>(y, x);
                unsigned char gt_pixel = gt.at<uchar>(y, x);

                if(gt_pixel == cl){
                    cl_in_img++;
                }

                if(net_pixel == cl || gt_pixel == cl){
                    if(gt_pixel != 0 && gt_pixel != 11){
                        if(net_pixel == cl && gt_pixel == cl){
                            // True positive
                            class_stat.true_pos_ += 1;
                        }
                        else if(net_pixel == cl && gt_pixel != cl){
                            // False positive
                            class_stat.false_pos_ += 1;
                        }
                        else if(net_pixel !=cl && gt_pixel == cl){
                            // False negative
                            class_stat.false_neg_ += 1;
                        }
                    }
                }else{
                    // True negative
                }
            }
        }

        // Calc IoU
        //        double iou_test = (double) class_stat.true_pos_ /((double) countPixelBelongingToClass(net_im, cl) + (double) countPixelBelongingToClass(gt, cl) - (double) class_stat.true_pos_);
        double div = ((double)class_stat.true_pos_ + (double)class_stat.false_pos_ + (double)class_stat.false_neg_);
        //std::cout <<"cl: " << cl <<   "div : " << div << " tp: " << (double) class_stat.true_pos_ << " fn: " << (double) class_stat.false_neg_ << " cl_in_img: " << cl_in_img << " fp: " << class_stat.false_pos_ << std::endl;
        double tp_temp = (double) class_stat.true_pos_;
        //if(div != 0 && tp_temp != 0 && cl_in_img > 0){
        if(div != 0 ){
            class_stat.iou_ = tp_temp / div;
        }else{
            class_stat.iou_ = -1;
        }
        // Calc precision
        div = (double) (class_stat.true_pos_ + class_stat.false_pos_);
        if(div > 0  ){
            class_stat.precision_ = (double)class_stat.true_pos_ / div;
        }else{
            class_stat.precision_ = -1;
        }

        // Calc recall
        div = (double) (class_stat.true_pos_ + class_stat.false_neg_);
        if(div > 0 ){
            class_stat.recall_ = (double)class_stat.true_pos_ / div;
        }
        else{
            class_stat.recall_ = -1;
        }


        statistics[cl] = class_stat;
    }
}

void Classifier::createColorCodedLabels(const cv::Mat &labels, cv::Mat &color_coded, std::map<unsigned char, cv::Vec3b> color_map){
    for(auto y = 0; y < labels.rows; y++){
        for(auto x = 0; x < labels.cols; x++){
            unsigned char label = labels.at<uchar>(y, x);
            if(color_map.find(label) !=  color_map.end()){
                cv::Vec3b color = color_map.at(label);
                (color_coded.at<cv::Vec3b>(y, x))[0] = color[0];
                (color_coded.at<cv::Vec3b>(y, x))[1] = color[1];
                (color_coded.at<cv::Vec3b>(y, x))[2] = color[2];
            }else{
                (color_coded.at<cv::Vec3b>(y, x))[0] = 0;
                (color_coded.at<cv::Vec3b>(y, x))[1] = 0;
                (color_coded.at<cv::Vec3b>(y, x))[2] = 0;
            }
        }
    }
}

void Classifier::Argmax(const std::vector<cv::Mat> &channels, cv::Mat &argmax){

    size_t width = channels.front().cols;
    size_t height = channels.front().rows;
    argmax = cv::Mat(height, width, CV_8UC1, cv::Scalar(0));

    for(size_t y = 0; y < height; y++){
        for(size_t x = 0; x < width; x++){

            unsigned char best_i = 0;
            float best_val = std::numeric_limits<float>::lowest();

            for(size_t ch = 0; ch < channels.size(); ch++){
                float val = channels.at(ch).at<float>(y, x);
                if(val > best_val){
                    best_val = val;
                    best_i = ch;
                }		
            }
            argmax.at<unsigned char>(y,x) = best_i;
        }
    }

  double max= 0;
        double min = 0;
        cv::minMaxLoc(argmax, &min, &max);
}

/* Return the top N predictions. */
std::vector<std::vector<cv::Mat> > Classifier::Classify(const std::vector<cv::Mat>& imgs, std::vector<std::string> req_layer) {
    const std::vector<std::vector<cv::Mat> > &output = Predict(imgs, req_layer);
    num_output_classes_ = output.front().size();
    return output;

}

void Classifier::SetMean(const cv::Scalar &mean, uint8_t index){
    mean_[index] = cv::Mat(input_geometry_, CV_32FC3, mean);
}

/* Load the mean file in binaryproto format. */
void Classifier::SetMean(const std::map<unsigned char,std::string>& mean_files) {
    for(auto i = mean_files.begin(); i != mean_files.end(); i++){

        BlobProto blob_proto;
        ReadProtoFromBinaryFileOrDie(i->second.c_str(), &blob_proto);

        /* Convert from BlobProto to Blob<float> */
        Blob<float> mean_blob;
        mean_blob.FromProto(blob_proto);
        //CHECK_EQ(mean_blob.channels(), num_channels_)
        //        << "Number of channels of mean file doesn't match input layer.";

        /* The format of the mean file is planar 32-bit float BGR or grayscale. */
        std::vector<cv::Mat> channels;
        float* data = mean_blob.mutable_cpu_data();
        for (int k = 0; k < num_channels_; ++k) {
            /* Extract an individual channel. */
            cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
            channels.push_back(channel);
            data += mean_blob.height() * mean_blob.width();
        }

        /* Merge the separate channels into a single image. */
        cv::Mat mean;
        //std::reverse(channels.begin(),channels.end());
        cv::merge(channels, mean);

        mean_[i->first] = mean;
    }
}

std::vector<std::vector<cv::Mat> > Classifier::Predict(const std::vector<cv::Mat>& imgs, std::vector<std::string> req_layers) {
    std::cout << "Forward net with " << imgs.size() << " inputs" << std::endl;
    for(auto j = 0; j < imgs.size(); j++){
        Blob<float>* input_layer = net_->input_blobs()[j];
        input_layer->Reshape(1, imgs.at(j).channels(),
                             input_geometry_.height, input_geometry_.width);
    }

    std::cout << "Input dimensions: " << input_geometry_  <<  std::endl;
    /* Forward dimension change to all layers. */
    net_->Reshape();

    for(auto j = 0; j < imgs.size(); j++){
        // CHECK
        std::vector<cv::Mat> input_channels;
        WrapInputLayer(&input_channels, j);
        Preprocess(imgs.at(j), &input_channels, j);
    }

    boost::timer::cpu_timer timer;
    net_->ForwardPrefilled();
    std::cout << "Forward took: " << timer.format() << '\n';

    std::vector<std::vector<cv::Mat> > final;
    for(auto r = 0; r < req_layers.size(); r++){
        //std::cout << "Wrapping output..." << std::endl;
        std::vector< cv::Mat> output_channels;
        WrapOutputLayer(&output_channels, req_layers.at(r));
        //std::cout << "Argmax channels..." << std::endl;
        //for(auto i = 0; i< output_channels.size(); i++){
        //    std::cout << "Channel " << i << " width: " << output_channels.at(i).cols << " height: " << output_channels.at(i).rows << std::endl;
        //}
        final.push_back(output_channels);
    }
    return final;
}


void Classifier::WrapInputLayer( std::vector<cv::Mat>* input_channels, size_t num_input) {
    Blob<float>* input_layer = net_->input_blobs()[num_input];

    int width = input_layer->width();
    int height = input_layer->height();
    float* input_data = input_layer->mutable_cpu_data();
    for (int i = 0; i < input_layer->channels(); ++i) {
        const cv::Mat channel(height, width, CV_32FC1, input_data);
        input_channels->push_back(channel);
        input_data += width * height;
    }
}

void Classifier::WrapOutputLayer(std::vector< cv::Mat>* output_channel, std::string l_name) {
    boost::shared_ptr<Blob<float> > output_layer = net_->blob_by_name(l_name);
    //Blob<float>* output_layer = net_->output_blobs()[0];

    int width = output_layer->width();
    int height = output_layer->height();

    float* output_data = output_layer->mutable_cpu_data();
std::cout << "Ouput layer channels: " << output_layer->channels() << std::endl;
    for (int i = 0; i < output_layer->channels(); ++i) {
        cv::Mat channel(height, width, CV_32F, output_data);
        output_channel->push_back(channel);
        output_data += width * height;
    }
}

void Classifier::Preprocess(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels, uint8_t input_index) {
    /* Convert the input image to the input image format of the network. */
    cv::Mat sample;
    sample = img;

    cv::Mat sample_resized;
    if (sample.size() != input_geometry_)
        cv::resize(sample, sample_resized, input_geometry_);
    else
        sample_resized = sample;

    cv::Mat sample_float;
    switch(sample_resized.channels()){
    case 1:
        sample_resized.convertTo(sample_float, CV_32FC1);
        break;

    case 2:
        sample_resized.convertTo(sample_float, CV_32FC2);
        break;

    case 3:
        sample_resized.convertTo(sample_float, CV_32FC3);
        break;
    }

    cv::Mat sample_normalized;
    if(mean_.find(input_index) != mean_.end()){
        cv::Mat mean;
        cv::resize(mean_.at(input_index), mean, img.size());
        cv::subtract(sample_float, mean, sample_normalized);
    }else{
        sample_normalized = sample_float;
    }

    cv::split(sample_normalized, *input_channels);

    CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
          == net_->input_blobs()[input_index]->cpu_data())
            << "Input channels are not wrapping the input layer of the network.";
}

cv::Mat Classifier::depthImage( cv::Mat &in){
#ifdef DEPTH_COLOR

    //in = -1 * in;
    double min;
    double max;
    cv::minMaxIdx(in, &min, &max);
    cv::Mat adjMap;
    //std::cout << "Max " << max << " Min " << min << std::endl;
    // expand your range to 0..255. Similar to histEq();
    in.convertTo(adjMap,CV_8UC1);//, 255 / (max-min), -min);

    // this is great. It converts your grayscale image into a tone-mapped one,
    // much more pleasing for the eye
    // function is found in contrib module, so include contrib.hpp
    // and link accordingly
    cv::Mat falseColorsMap;
    cv::applyColorMap(adjMap, falseColorsMap, cv::COLORMAP_JET);

    return falseColorsMap;
#endif

}

cv::Mat Classifier::flowImage(cv::Mat &in){
    //extraxt x and y channels
    cv::Mat xy[2]; //X,Y
    cv::split(in, xy);

    //calculate angle and magnitude
    cv::Mat magnitude, angle;
    cv::cartToPolar(xy[0], xy[1], magnitude, angle, true);

    //translate magnitude to range [0;1]
    double mag_max;
    cv::minMaxLoc(magnitude, 0, &mag_max);
    magnitude.convertTo(magnitude, -1, 1.0 / mag_max);

    //build hsv image
    cv::Mat _hsv[3], hsv;
    _hsv[0] = angle;
    _hsv[1] = cv::Mat::ones(angle.size(), CV_32F);
    _hsv[2] = magnitude;
    cv::merge(_hsv, 3, hsv);

    //convert to BGR and show
    cv::Mat bgr;//CV_32FC3 matrix
    cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
    return bgr;
}

}
