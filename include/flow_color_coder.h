#ifndef FLOW_COLOR_CODER_H
#define FLOW_COLOR_CODER_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

 #define MAXCOLS 60
// the "official" threshold - if the absolute value of either
// flow component is greater, it's considered unknown
#define UNKNOWN_FLOW_THRESH 1e9

// value to use to represent unknown flow
#define UNKNOWN_FLOW 1e10

class flow_color_coder
{
public:
    flow_color_coder();

    void motionToColor(const cv::Mat &in, cv::Mat &out, float maxmotion);

    void computeColor(float fx, float fy, cv::Vec<unsigned char, 3> &pix);

    void makecolorwheel();

    void setcols(int r, int g, int b, int k);

    int verbose;
    int ncols;
    int colorwheel[MAXCOLS][3];
};

#endif // FLOW_COLOR_CODER_H
