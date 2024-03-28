#ifndef UTILITY_H
#define UTILITY_H

#include <cmath>
#include <math.h>
#include <fstream>
#include <iostream>
#include <string.h>
#include <vector>
#include <opencv2/features2d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>

#include "definitions.h"

// =====================================================================================================================
// UTILITY_TOOLS: useful functions for debugging, writing data to files, displaying images, etc.
//
// ChangeLogs
//    Chien  23-01-18    Initially created.
//    Chien  23-01-19    Add bilinear interpolation    
//
//> (c) LEMS, Brown University
//> Chiang-Heng Chien (chiang-heng_chien@brown.edu)
// ======================================================================================================================

class Utility {

public:

    //> Make the class shareable as a pointer
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Utility> Ptr;

    Utility();
    void get_dG_2D(cv::Mat &Gx_2d, cv::Mat &Gy_2d, int w, double sigma);
    double Bilinear_Interpolation(Frame::Ptr Frame, cv::Point2d P);
    void Display_Feature_Correspondences(cv::Mat Img1, cv::Mat Img2, \
                                         std::vector<cv::KeyPoint> KeyPoint1, std::vector<cv::KeyPoint> KeyPoint2, \
                                         std::vector<cv::DMatch> Good_Matches );
    std::string cvMat_Type(int type);

    template<typename T>
    T Uniform_Random_Number_Generator(T range_from, T range_to);
};

#endif