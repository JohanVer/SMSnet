#include "../include/flow_color_coder.h"

// return whether flow vector is unknown
bool unknown_flow(float u, float v) {
    return (fabs(u) >  UNKNOWN_FLOW_THRESH)
            || (fabs(v) >  UNKNOWN_FLOW_THRESH)
            || std::isnan(u) || std::isnan(v);
}

bool unknown_flow(float *f) {
    return unknown_flow(f[0], f[1]);
}

void flow_color_coder::setcols(int r, int g, int b, int k)
{
    colorwheel[k][0] = r;
    colorwheel[k][1] = g;
    colorwheel[k][2] = b;
}

flow_color_coder::flow_color_coder()
{
    ncols = 0;
    verbose = 1;
}

void flow_color_coder::makecolorwheel()
{
    // relative lengths of color transitions:
    // these are chosen based on perceptual similarity
    // (e.g. one can distinguish more shades between red and yellow
    //  than between yellow and green)
    int RY = 15;
    int YG = 6;
    int GC = 4;
    int CB = 11;
    int BM = 13;
    int MR = 6;
    ncols = RY + YG + GC + CB + BM + MR;
    //printf("ncols = %d\n", ncols);
    if (ncols > MAXCOLS)
        exit(1);

    int i;
    int k = 0;
    for (i = 0; i < RY; i++) setcols(255,	   255*i/RY,	 0,	       k++);
    for (i = 0; i < YG; i++) setcols(255-255*i/YG, 255,		 0,	       k++);
    for (i = 0; i < GC; i++) setcols(0,		   255,		 255*i/GC,     k++);
    for (i = 0; i < CB; i++) setcols(0,		   255-255*i/CB, 255,	       k++);
    for (i = 0; i < BM; i++) setcols(255*i/BM,	   0,		 255,	       k++);
    for (i = 0; i < MR; i++) setcols(255,	   0,		 255-255*i/MR, k++);
}

void flow_color_coder::computeColor(float fx, float fy, cv::Vec<unsigned char, 3> &pix)
{
    if (ncols == 0)
        makecolorwheel();

    float rad = sqrt(fx * fx + fy * fy);
    float a = atan2(-fy, -fx) / M_PI;
    float fk = (a + 1.0) / 2.0 * (ncols-1);
    int k0 = (int)fk;
    int k1 = (k0 + 1) % ncols;
    float f = fk - k0;
    //f = 0; // uncomment to see original color wheel
    for (int b = 0; b < 3; b++) {
        float col0 = colorwheel[k0][b] / 255.0;
        float col1 = colorwheel[k1][b] / 255.0;
        float col = (1 - f) * col0 + f * col1;
        if (rad <= 1)
            col = 1 - rad * (1 - col); // increase saturation with radius
        else
            col *= .75; // out of range
        pix[2 - b] = (int)(255.0 * col);
    }
}

void flow_color_coder::motionToColor(const cv::Mat &in, cv::Mat &out, float maxmotion){
    cv::Size sh = in.size();
    int width = sh.width, height = sh.height;
    out = cv::Mat(height, width, CV_8UC3, cv::Scalar(0,0,0));
    int x, y;
    // determine motion range:
    float maxx = -999, maxy = -999;
    float minx =  999, miny =  999;
    float maxrad = -1;
    for (y = 0; y < height; y++) {
        for (x = 0; x < width; x++) {
            cv::Vec<float, 2> input_px = in.at<cv::Vec<float, 2>>(y, x);
            float fx = input_px[0];
            float fy = input_px[1];

            if (unknown_flow(fx, fy))
                continue;
            maxx = std::max(maxx, fx);
            maxy = std::max(maxy, fy);
            minx = std::min(minx, fx);
            miny = std::min(miny, fy);
            float rad = sqrt(fx * fx + fy * fy);
            maxrad = std::max(maxrad, rad);
        }
    }
    //printf("max motion: %.4f  motion range: u = %.3f .. %.3f;  v = %.3f .. %.3f\n",
    //       maxrad, minx, maxx, miny, maxy);
    std::cout << "max_motion: " << maxrad << " motion_range: u= " << minx << " .. " << maxx << " v= " << miny << " .. " << maxy << std::endl;


    if (maxmotion > 0) // i.e., specified on commandline
        maxrad = maxmotion;

    if (maxrad == 0) // if flow == 0 everywhere
        maxrad = 1;

    if (verbose)
        fprintf(stderr, "normalizing by %g\n", maxrad);

    for (y = 0; y < height; y++) {
        for (x = 0; x < width; x++) {
            cv::Vec<float, 2> input_px = in.at<cv::Vec<float, 2>>(y, x);
            float fx = input_px[0];
            float fy = input_px[1];

            cv::Vec<unsigned char, 3> &output_px = out.at<cv::Vec<unsigned char, 3>>(y, x);
            if (unknown_flow(fx, fy)) {
                output_px[0] = output_px[1] = output_px[2] = 0;
            } else {
                computeColor(fx/maxrad, fy/maxrad, output_px);
            }
        }
    }
}
