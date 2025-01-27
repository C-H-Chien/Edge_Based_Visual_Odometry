#ifndef DATASET_H
#define DATASET_H

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <yaml-cpp/yaml.h>

#include "definitions.h"
#include "Frame.h"
#include "utility.h"

// =======================================================================================================
// class Dataset: Fetch data from dataset specified in the configuration file
//
// ChangeLogs
//    Lopez 25-01-26     Modified for euroc dataset support.
//    Chien  23-01-17    Initially created.
//
//> (c) LEMS, Brown University
//> Chiang-Heng Chien (chiang-heng_chien@brown.edu)
// =======================================================================================================

class Dataset {
    
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Dataset> Ptr;
    Dataset(YAML::Node, bool);

    void PrintDatasetInfo();
    void DetectEdges(int num_images);
    void UndistortEdgePoints(const cv::Mat& edgeImage, const std::vector<double>& intrinsics, 
                         const std::vector<double>& distCoeffs, cv::Mat& undistortedPoints);
    // bool Init_Fetch_Data();
    // Frame::Ptr get_Next_Frame();

    // unsigned Total_Num_Of_Imgs;
    // int Current_Frame_Index;
    // double fx, fy, cx, cy;

private:

    YAML::Node config_file;

    std::string Dataset_Type;
    std::string Dataset_Path;
    std::string Sequence_Name;

    std::vector<int> cam0_resolution;
    int cam0_rate_hz;
    std::string cam0_model;
    std::vector<double> cam0_intrinsics;
    std::string cam0_dist_model;
    std::vector<double> cam0_dist_coeffs;

    std::vector<int> cam1_resolution;
    int cam1_rate_hz;
    std::string cam1_model;
    std::vector<double> cam1_intrinsics;
    std::string cam1_dist_model;
    std::vector<double> cam1_dist_coeffs;

    std::vector<std::vector<double>> rotation_matrix;
    std::vector<double> translation_vector;

    // Eigen::Matrix3d Calib;        
    // Eigen::Matrix3d Inverse_Calib; 

    // std::vector<std::string> Img_Path_List;
    // std::vector<std::string> Depth_Path_List;
    // std::vector<std::string> Img_Time_Stamps;

    // bool has_Depth;
    // cv::Mat grad_Depth_xi_;
    // cv::Mat grad_Depth_eta_;
    bool compute_grad_depth = false;
    cv::Mat Gx_2d, Gy_2d;
    cv::Mat Small_Patch_Radius_Map;

    Utility::Ptr utility_tool = nullptr;
};

#endif 
