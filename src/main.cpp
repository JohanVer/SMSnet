#include "../include/classify.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
//#include <glog/logging.h>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/fstream.hpp>
#include <caffe/util/db.hpp>
//#include <caffe/util/format.hpp>
#include "boost/scoped_ptr.hpp"
#include "google/protobuf/text_format.h"
#include "../include/flow_color_coder.h"
#include "../include/hdf5_reader.h"

//#define STATIC_BIG_CLASS

cv::Mat createDisparityChange(const cv::Mat &disp1, const cv::Mat &disp2, const cv::Mat &flow);

std::map<unsigned char, cv::Vec3b> net_colors_;
std::set<size_t> ignore_list;

std::map<unsigned char, std::string> mean_files;
std::vector<std::string> images_paths_file;
std::vector<std::string> output_layer_names;

flow_color_coder flow_coder;

const bool display_mode = true;

const std::string model_file   = "/home/vertensj/iros_deploy/motion/deploy_2class_kitti.prototxt";
const std::string trained_file = "/home/vertensj/deepmotion/caffe_software/caffe/models/iRos17/motion/city_kitti/2_class/80/best45k.caffemodel";

const bool is_not_txt = true;

const std::string labels_paths_file = "/home/vertensj/final_datasets/iRos_datasets/motion_test/kitti/kitti_motion_test/labels_lmdb";

const std::string color_mapping_file = "";

std::map<uchar, std::string> hdf5_channel_names;
int number_input_images;

void setupVars(){
    // Images
    // First has to correspond to label
    images_paths_file.push_back("/home/vertensj/final_datasets/iRos_datasets/motion_test/kitti/kitti_motion_test/images_lmdb3");
    
    // Flow (either ego-flow subtracted or raw flow) In the LMDB Databases the flow is scaled by 1/6.4 and centered at 128 to fit in the unsigned char format
    images_paths_file.push_back("/home/vertensj/final_datasets/iRos_datasets/motion_test/kitti/kitti_motion_test/images_lmdb5");

    // Mean file assignment
    mean_files[0] = "/home/vertensj/final_datasets/iRos_datasets/motion_train/city_kitti_cars_motion//mean3.binaryproto";

    // Requested layer names (first has to correspond to label)
    output_layer_names.push_back("out");

    // This class numbers will be ignored for evaluation
    ignore_list.insert(0);

    // Number of images which should be passed to network
    number_input_images = 2;
}

void setupColorMaps(){
    net_colors_[2] = cv::Vec3b(0,0,255);
    net_colors_[3] = cv::Vec3b(255,0,0);
    net_colors_[4] = cv::Vec3b(0,255,0);
}

void setupMapping3c(boost::filesystem::path source, std::map<unsigned char, cv::Vec3b> &mapping){
    mapping.clear();

    std::ifstream mapping_file;
    mapping_file.open(source.string());

    std::string line;
    while (std::getline(mapping_file, line)){
        size_t separator = line.find("=");
        std::string id = line.substr(0,separator);
        size_t end_1 = line.find(",");
        std::string r = line.substr(separator+1,end_1);
        size_t end_2 = line.find(",", end_1 +1);
        std::string g = line.substr(end_1+1,end_2);
        std::string b = line.substr(end_2+1);


        cv::Vec3b map_keys;
        map_keys[0] = std::stoi(b);
        map_keys[1] = std::stoi(g);
        map_keys[2] = std::stoi(r);

        size_t value_i = std::stoi(id);

        std::cout << "Read mapping: " << value_i << " --> " << map_keys << std::endl;
        mapping[value_i] = map_keys;
    }
}

bool loadPaths(const boost::filesystem::path &path_to_paths_file, std::vector<boost::filesystem::path> &list_of_paths){
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

cv::Mat datumToMat(caffe::Datum &datum){
    cv::Mat mat;
    if(datum.channels() == 1){
        mat =  cv::Mat(datum.height(), datum.width(), CV_8UC1, cv::Scalar(0));
    }else if(datum.channels() == 3){
        mat =  cv::Mat(datum.height(), datum.width(), CV_8UC3, cv::Scalar(0,0,0));
    }
    size_t pix_per_channel = datum.height() * datum.width();

    std::vector<cv::Mat> channels;
    for(auto ch = 0; ch < datum.channels(); ch++){
        cv::Mat channel =  cv::Mat(datum.height(), datum.width(), CV_8UC1, cv::Scalar(0));

        for(auto k = 0; k < pix_per_channel; k++){
            uint8_t label = (uint8_t) datum.data()[k + ch * pix_per_channel ];
            channel.at<uint8_t>(k) = label;
        }
        channels.push_back(channel);
    }
    cv::merge(channels, mat);
    // TODO: clone necessary ?
    return mat.clone();
}


void computeLateAverages(std::vector<std::map<size_t, classify::PredictionStatistic> > &all_statistics, size_t num_classes){
    for(auto cl = 0; cl < num_classes ; cl++){
	long total_tp = 0;
	long total_fp = 0;
	long total_fn = 0;
        for(auto i = 0; i < all_statistics.size(); i++){
            std::map<size_t, classify::PredictionStatistic> &stat_im = all_statistics.at(i);
            total_tp += stat_im.at(cl).true_pos_;
            total_fp += stat_im.at(cl).false_pos_;
            total_fn += stat_im.at(cl).false_neg_;
        }
	
	long div = total_tp + total_fp + total_fn;
	std::cout << "TP: " <<  total_tp << " FP: " << total_fp << " FN: " << total_fn << std::endl;
	if(div != 0){
		double iou = double(total_tp) / double(div);
		std::cout << "------------Mean IoU for class: " << cl << " is: " << iou << "------------------" << std::endl;
	}

    }
}


template <class T, class V>
void mapLabels(T mapping, const cv::Mat &in, cv::Mat &out){
    for(auto y = 0; y < in.rows; y++){
        for(auto x = 0; x < in.cols; x++){
            V pixel = in.at<V>(y, x);
            if(mapping.find(pixel) == mapping.end()){
                out.at<unsigned char>(y, x) = 0;
            }else{
                out.at<unsigned char>(y, x) = mapping.at(pixel);
            }
        }
    }
}

int main(int argc, char** argv) {

    ::google::InitGoogleLogging(argv[0]);

    if(color_mapping_file.empty()){
        setupColorMaps();
    }else{
        setupMapping3c(boost::filesystem::path(color_mapping_file), net_colors_);
    }

    setupVars();

    if(display_mode){
        cv::namedWindow("Network output", CV_WINDOW_AUTOSIZE|CV_WINDOW_FREERATIO);
        cv::namedWindow("Reference output", CV_WINDOW_AUTOSIZE|CV_WINDOW_FREERATIO);
        cv::namedWindow("Label output", CV_WINDOW_AUTOSIZE|CV_WINDOW_FREERATIO);
    }
    std::cout << "Loading network..." << std::endl;
    classify::Classifier classifier(model_file, trained_file, mean_files);
    std::cout << "Total number of classes: " << classifier.getNumberOfClasses() << std::endl;

    std::vector<std::map<size_t, classify::PredictionStatistic> > all_statistics;

    std::vector<std::vector<boost::filesystem::path> > feed_images;
    std::vector<boost::filesystem::path> feed_labels;
    std::vector<std::unique_ptr<caffe::db::DB> > db_images;

    std::vector<std::unique_ptr<hdf5_reader> > hdf5_readers;

    std::unique_ptr<caffe::db::DB> db_labels;
    std::vector<std::unique_ptr<caffe::db::Cursor> >images_cursor;
    caffe::db::Cursor *labels_cursor;

    int adapted_width = -1;
    int adapted_height = -1;

    double mean_iou_global = 0;
    double mean_prec_global = 0;
    double mean_rec_global = 0;
    double mean_fp_global = 0;
    double mean_fn_global = 0;
    size_t images_size = 0;
    size_t labels_size = 0;

    size_t save_counter = 0;

    size_t total_valid_counter = 0;

    size_t input_size = number_input_images; //images_paths_file.size();

    std::map<uchar, std::string> database_types;

    if(!is_not_txt){
        std::cout << "Reading from txt files..." << std::endl;
        feed_images.resize(images_paths_file.size());
        for(auto i = 0; i < images_paths_file.size(); i++){
            loadPaths(boost::filesystem::path(images_paths_file.at(i)), feed_images.at(i));
        }
        loadPaths(boost::filesystem::path(labels_paths_file), feed_labels);
        images_size = feed_images.front().size();
    }else{
        size_t lmdb_count = 0;
        size_t hdf5_count = 0;
        std::cout << "Reading from databases..." << std::endl;
        for(auto i = 0; i < images_paths_file.size(); i++){
            boost::filesystem::path path = boost::filesystem::path(images_paths_file.at(i));
            if(boost::filesystem::is_directory(path)){
                database_types[i] = "lmdb";
                lmdb_count++;
            }else if(path.filename().string().find("hdf") != std::string::npos){
                hdf5_count++;
                database_types[i] = "hdf5";
            }
        }

        std::cout << "LMDB files detected: " << lmdb_count << std::endl;
        std::cout << "HDF5 files detected: " << hdf5_count << std::endl;

        size_t h_c = 0;
        for(auto db_idx = 0; db_idx < images_paths_file.size(); db_idx++){
            if(database_types[db_idx] == "lmdb"){
                std::cout << "Open LMDB: " << images_paths_file.at(db_idx) << std::endl;
                db_images.push_back(std::unique_ptr<caffe::db::DB>(caffe::db::GetDB("lmdb")));
                db_images.back()->Open(images_paths_file.at(db_idx),caffe::db::Mode::READ);
                images_cursor.push_back(std::unique_ptr<caffe::db::Cursor>(db_images.back()->NewCursor()));
                images_cursor.back()->SeekToFirst();


            }else if(database_types[db_idx] == "hdf5"){
                std::cout << "Open HDF5: " << images_paths_file.at(db_idx) << std::endl;
                hdf5_readers.push_back(std::unique_ptr<hdf5_reader>(new hdf5_reader(boost::filesystem::path(images_paths_file.at(db_idx)), hdf5_channel_names[h_c])));
                h_c++;
            }
        }

        // First lmdb database has to exist -> counting for images
        // count images
        while(images_cursor.front()->valid()){
            images_cursor.front()->Next();
            images_size++;
        }
        images_cursor.front()->SeekToFirst();

        // Labels is assumed to be always a lmdb database
        if(!labels_paths_file.empty()){
            db_labels = std::unique_ptr<caffe::db::DB>(caffe::db::GetDB("lmdb"));
            db_labels->Open(labels_paths_file,caffe::db::Mode::READ);
            labels_cursor = db_labels->NewCursor();
            labels_cursor->SeekToFirst();
            // count images
            while(labels_cursor->valid()){
                labels_cursor->Next();
                labels_size++;
            }
            labels_cursor->SeekToFirst();
            assert(labels_size == images_size);
        }
    }

    cv::Size input_shape = classifier.getInputSize();

    std::cout << "Image size: " << images_size << std::endl;
    for(auto i = 0; i < images_size; i++){
        std::vector<cv::Mat> imgs;
        if(!is_not_txt){
            std::cout << "---------- Prediction for "
                      << feed_images.front().at(i).filename().string() << " ----------" << std::endl;

            for(auto j = 0; j < input_size; ++j){
                cv::Mat rez;
                int divisor = 64;
                cv::Mat org = cv::imread(feed_images.at(j).at(i).string(), 1);
                if(org.cols % divisor == 0 && org.rows % divisor ==0){
                    std::cout << "Size statisfies size divisor of " << divisor << std::endl;
                    cv::resize(org, rez, input_shape);
                }else{
                    adapted_width = std::ceil(input_shape.width/divisor) * divisor;
                    adapted_height = std::ceil(input_shape.height/divisor) * divisor;
                    cv::resize(org, rez, cv::Size(adapted_width, adapted_height));
                    std::cout << "Input does not statisfy the divisor: " << divisor << " adapted width: " << adapted_width << " adapted height: " << adapted_height << std::endl;
                }
                imgs.push_back(rez);
                CHECK(!imgs.back().empty()) << "Unable to decode image " << feed_images.at(j).at(i).string();
            }
        }else{
            uchar lm_c = 0;
            uchar h5_c = 0;
            for(auto j = 0; j < input_size; ++j){
                if(database_types[j] == "lmdb"){
                    caffe::Datum datum;
                    datum.ParseFromString(images_cursor.at(lm_c)->value());
                    imgs.push_back(datumToMat(datum));
                    images_cursor.at(lm_c)->Next();
                    lm_c++;
                }

                if(database_types[j] == "hdf5"){
                    imgs.push_back(hdf5_readers.at(h5_c)->nextImage());
                    h5_c++;
                }
            }
        }

        std::cout << "Channels 1 : " << imgs.front().channels() << std::endl;

        // Classify image
        cv::Size org_size = imgs.front().size();
        std::vector<std::vector<cv::Mat> > final_raw = classifier.Classify(imgs, output_layer_names);


        cv::Mat out_argmax;
        cv::Mat net_blended;
        cv::Mat label_blended;
        cv::Mat net_color_coded;
        cv::Mat label_color_coded;

        classifier.Argmax(final_raw.front(), out_argmax);

#ifdef STATIC_BIG_CLASS
        std::map<unsigned char, unsigned char> relabel_mapping;
        relabel_mapping[0]=0;
        relabel_mapping[1]=3;
        relabel_mapping[2]=2;
        relabel_mapping[3]=3;
        cv::Mat relabelled = cv::Mat(org_size.height, org_size.width, CV_8UC1, cv::Scalar(0));
        mapLabels<std::map<unsigned char, unsigned char>, unsigned char>(relabel_mapping, out_argmax, relabelled);
        out_argmax = relabelled;
#endif

        cv::Size output_size = out_argmax.size();
        // Blend color_coded net image with original image
        ////        cv::resize(out_argmax, out_argmax, org_size, CV_INTER_NN);
        double max= 0;
        double min = 0;
        cv::minMaxLoc(out_argmax, &min, &max);
        float alpha = 0.45;
        double beta = (1.0 - alpha);
        net_color_coded = cv::Mat(org_size.height, org_size.width, CV_8UC3, cv::Scalar(0,0,0));
        classifier.createColorCodedLabels(out_argmax,net_color_coded, net_colors_);
        cv::addWeighted( net_color_coded, alpha, imgs.front(), beta, 0.0, net_blended);


        if(!labels_paths_file.empty()){
            std::cout << "Output size: " << out_argmax.size() << std::endl;
            // Compare network output with GT and calculate metrics
            cv::Mat label_image;
            cv::Mat label_resized;
            if(!is_not_txt){
                label_image = cv::imread(feed_labels.at(i).string().c_str(), 0);
                cv::resize(label_image, label_resized, out_argmax.size(), cv::INTER_NEAREST);
            }else{
                labels_cursor->value();
                caffe::Datum datum;
                datum.ParseFromString(labels_cursor->value());
                const std::string data = datum.data();

                label_image = datumToMat(datum);
                //cv::resize(label_image, label_resized, out_argmax.size(), cv::INTER_NEAREST);
                label_resized = label_image;
                labels_cursor->Next();
            }

#ifdef STATIC_BIG_CLASS
            cv::Mat relabelled_label = cv::Mat(org_size.height, org_size.width, CV_8UC1, cv::Scalar(0));
            mapLabels<std::map<unsigned char, unsigned char>, unsigned char>(relabel_mapping, label_resized, relabelled_label);
            label_resized = relabelled_label;
#endif

            std::map<size_t, classify::PredictionStatistic> statistics;
            double accuracy = 0;
            double error = 0;
            classifier.calculateStatistics(out_argmax, label_resized, ignore_list, statistics, accuracy, error);
            all_statistics.push_back(statistics);

            cv::Mat resized;
            if(display_mode){
                // Blend original label with original image
                label_color_coded = cv::Mat(org_size.height, org_size.width, CV_8UC3, cv::Scalar(0,0,0));
                classifier.createColorCodedLabels(label_resized,label_color_coded, net_colors_);
                cv::addWeighted( label_color_coded, alpha, imgs.front(), beta, 0.0, label_blended);
            }
            // Show output
            for(auto i = statistics.begin(); i != statistics.end(); i++){
                std::cout << "----------------------------------------------------------------------------" << std::endl;
                std::cout << "Precision for class: " << i->first << " is: " << i->second.precision_ << std::endl;
                std::cout << "Recall for class: " << i->first << " is: " << i->second.recall_ << std::endl;
                std::cout << "IoU for class: " << i->first << " is: " << i->second.iou_ << std::endl;
                std::cout << "TP for class: " << i->first << " are: " << i->second.true_pos_ << std::endl;
                std::cout << "FP for class: " << i->first << " are: " << i->second.false_pos_ << std::endl;
                std::cout << "FN for class: " << i->first << " are: " << i->second.false_neg_ << std::endl;
            }
            double mean_image_iou = 0;
            double mean_image_prec = 0;
            double mean_image_rec = 0;
            double mean_image_fp = 0;
            double mean_image_fn = 0;
	    /*
            int valid_counter =  classifier.calcMeanStatistic(statistics, ignore_list, mean_image_iou, mean_image_prec, mean_image_rec, mean_image_fp, mean_image_fn);
            if(valid_counter){
                mean_iou_global += mean_image_iou;
                mean_prec_global += mean_image_prec;
                mean_rec_global += mean_image_rec;
                mean_fp_global += mean_image_fp;
                mean_fn_global += mean_image_fn;
                std::cout << "Mean IoU of Image: " << mean_image_iou << std::endl;
                std::cout << "Mean Precision of Image: " << mean_image_prec << std::endl;
                std::cout << "Mean Recall of Image: " << mean_image_rec << std::endl;
                std::cout << "Mean FP of Image: " << mean_image_fp << std::endl;
                std::cout << "Mean FN of Image: " << mean_image_fn << std::endl;

                total_valid_counter +=1;
            }
            */
        }

        if(display_mode){
            cv::imshow("Network output", net_blended);
            if(!labels_paths_file.empty()) cv::imshow("Reference output", label_blended);
            //cv::imshow("Current", imgs.front());
            cv::imshow("Label output", net_color_coded);
            //save_counter++;

            std::vector<cv::Mat> flow_channels;
	    cv::split(imgs.at(1), flow_channels);

            cv::Mat &magnitude = flow_channels.at(2);
            cv::Mat magnitude_norm;
            magnitude.convertTo(magnitude_norm, CV_8UC1);
            cv::imshow("Flow Magnitude", magnitude_norm);
            
            int k = cv::waitKey(0);
        }
        std::cout << "Processed : " << i <<  " of: " << images_size << " , percentage: "  << ((double) i / (double) images_size) * 100.0 << std::endl;
    }

    computeLateAverages(all_statistics, classifier.getNumberOfClasses());

    return 0;
}

cv::Mat createDisparityChange(const cv::Mat &disp1, const cv::Mat &disp2, const cv::Mat &flow){
    cv::Mat t_disp = cv::Mat(disp1.rows, disp1.cols, CV_32FC1,0.0);
    for(auto x = 0; x < flow.cols; x++){
        for(auto y = 0; y < flow.rows; y++){
            const cv::Vec2f &f_p = flow.at<cv::Vec2f>(y,x);
            // Compute new pixel position
            int n_x = x + f_p[0];
            int n_y = y + f_p[1];

            if(n_x >= 0 && n_x < disp1.cols && n_y >= 0 && n_y < disp1.rows){
                t_disp.at<float>(n_y, n_x) = disp1.at<float>(y,x);
            }
        }
    }

    cv::Mat disp_change = cv::Mat(disp2.rows, disp2.cols, CV_32FC1);
    for(auto x = 0; x < flow.cols; x++){
        for(auto y = 0; y < flow.rows; y++){
            disp_change.at<float>(y,x) = std::fabs(disp2.at<float>(y,x) - t_disp.at<float>(y,x));
        }
    }
    return disp_change;
}
