#ifndef UTILITY_CPP
#define UTILITY_CPP

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <random>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <glog/logging.h>

#include "Dataset.h"
#include "definitions.h"

// =======================================================================================================
// utility Dataset: Fetch data from dataset specified in the configuration file
//
// ChangeLogs
//    Chien  23-01-21    Initially created.
//
//> (c) LEMS, Brown University
//> Chiang-Heng Chien (chiang-heng_chien@brown.edu)
// =======================================================================================================

Utility::Utility() {}

void Utility::get_dG_2D(cv::Mat &Gx_2d, cv::Mat &Gy_2d, int w, double sigma) {
  
  cv::Mat G     = cv::Mat::ones(Gx_2d.cols, Gx_2d.rows, CV_64F);
  cv::Mat dG    = cv::Mat::ones(Gx_2d.cols, Gx_2d.rows, CV_64F);

  //> get zero mean Gaussian
  unsigned Index_Row = 0, Index_Col = 0;
  for (int iy = -w; iy <= w; iy++) {
    Index_Col = 0;
    for (int ix = -w; ix <= w; ix++) {
      G.at<double>(Index_Row, Index_Col) = (exp(-(ix*ix)/(2*sigma*sigma)) / std::sqrt(2*M_PI)) / sigma;
      Index_Col++;
    }
    Index_Row++;
  }

  //> get zero mean Gaussian derivative
  Index_Row = 0, Index_Col = 0;
  for (int iy = -w; iy <= w; iy++) {
    Index_Col = 0;
    for (int ix = -w; ix <= w; ix++) {
      dG.at<double>(Index_Row, Index_Col) = -(ix) * (exp(-(ix*ix)/(2*sigma*sigma)) / std::sqrt(2*M_PI)) / (sigma*sigma*sigma);
      Index_Col++;
    }
    Index_Row++;
  }

  //get_Zero_Mean_Gaussian(G, w, sigma);
  //get_Zero_Mean_Gaussian_Derivative(dG, w, sigma);

  for (int ri = 0; ri < Gx_2d.rows; ri++) {
    for (int ci = 0; ci < Gx_2d.cols; ci++) {
      Gx_2d.at<double>(ri, ci) = dG.at<double>(ri, ci) * G.at<double>(ci, ri);   //> G_y.*dG_x
      Gy_2d.at<double>(ri, ci) = dG.at<double>(ci, ri) * G.at<double>(ri, ci);   //> dG_y.*G_x
    }
  }
}

double Utility::Bilinear_Interpolation(Frame::Ptr Frame, cv::Point2d P) {

  //> y2 Q12--------Q22
  //      |          |
  //      |    P     |
  //      |          |
  //  y1 Q11--------Q21
  //      x1         x2
  cv::Point2d Q12 (floor(P.x), floor(P.y));
  cv::Point2d Q22 (ceil(P.x), floor(P.y));
  cv::Point2d Q11 (floor(P.x), ceil(P.y));
  cv::Point2d Q21 (ceil(P.x), ceil(P.y));

  double f_x_y1 = ((Q21.x-P.x)/(Q21.x-Q11.x))*Frame->Depth.at<float>(Q11.y, Q11.x) + ((P.x-Q11.x)/(Q21.x-Q11.x))*Frame->Depth.at<float>(Q21.y, Q21.x);
  double f_x_y2 = ((Q21.x-P.x)/(Q21.x-Q11.x))*Frame->Depth.at<float>(Q12.y, Q12.x) + ((P.x-Q11.x)/(Q21.x-Q11.x))*Frame->Depth.at<float>(Q22.y, Q22.x);
  return ((Q12.y-P.y)/(Q12.y-Q11.y))*f_x_y1 + ((P.y-Q11.y)/(Q12.y-Q11.y))*f_x_y2;
}

//> Display images and features via OpenCV
void Utility::Display_Feature_Correspondences(cv::Mat Img1, cv::Mat Img2, \
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

std::string Utility::cvMat_Type(int type) {
  //> Credit: https://stackoverflow.com/questions/10167534/how-to-find-out-what-type-of-a-mat-object-is-with-mattype-in-opencv

  std::string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:         r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

template<typename T>
T Utility::Uniform_Random_Number_Generator(T range_from, T range_to) {
  
  std::random_device                  rand_dev;
  std::mt19937                        generator(rand_dev());
  std::uniform_int_distribution<T>    distr(range_from, range_to);
  return distr(generator);
}

#endif