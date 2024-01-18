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
// IO_TOOLS: useful functions for debugging, writing data to files, displaying images, etc.
//
// ChangeLogs
//    Chien  23-01-19    Initially created.
//
//> (c) LEMS, Brown University
//> Chiang-Heng Chien (chiang-heng_chien@brown.edu)
// ======================================================================================================================

namespace IO_TOOLS {
/*
void Write_Final_Edge_Pair_Results( int Num_Of_Edgles_HYPO1, int* host_Hypothesis_Edgel_Pair_Index ) 
{
  std::ofstream GPU_Result_File;
  std::string Output_File_Path = OUTPUT_WRITE_FOLDER + "GPU_Final_Result.txt";
  GPU_Result_File.open(Output_File_Path);
  if (!GPU_Result_File) {
    std::cerr << "Unable to open GPU_Final_Result file!\n";
  }
  else {
    for (int i = 0; i < Num_Of_Edgles_HYPO1; i++)
        GPU_Result_File << i << "\t" << host_Hypothesis_Edgel_Pair_Index[i] << "\n";
  }
  GPU_Result_File.close();
}*/

//> Display images and features via OpenCV
void Display_Feature_Correspondences(cv::Mat Img1, cv::Mat Img2, \
                                     std::vector<cv::KeyPoint> KeyPoint1, std::vector<cv::KeyPoint> KeyPoint2, \
                                     std::vector<cv::DMatch> Good_Matches ) 
{
    //> Credit: matcher_simple.cpp from the official OpenCV
    cv::namedWindow("matches", 1);
    cv::Mat img_matches;
    cv::drawMatches(Img1, KeyPoint1, Img2, KeyPoint2, Good_Matches, img_matches);
    cv::imshow("matches", img_matches);
    cv::waitKey(0);
}

}

#endif