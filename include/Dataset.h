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
#include "./toed/cpu_toed.hpp"

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
    void PerformEdgeBasedVO(const cv::Mat& left_map, const cv::Mat& right_map, std::vector<cv::Point2f> left_edge_coords);
    int FindBestMatchSSD(const cv::Mat& left_patch, const std::vector<cv::Mat>& right_patches);
    void ExtractPatches(int patch_size, const cv::Mat& binary_map, const std::vector<cv::Point2f>& selected_edges, 
                             cv::Mat& visualization, std::vector<cv::Mat>& patches);
    void UndistortEdges(const cv::Mat& dist_edges, cv::Mat& undist_edges, 
                             std::vector<cv::Point2f>& edge_locations,
                             const std::vector<double>& intr, 
                             const std::vector<double>& dist_coeffs);
    std::vector<cv::Point2f> ExtractEpipolarEdges(const Eigen::Vector3d& epipolar_line, const cv::Mat& binary_map);
    std::vector<cv::Point2f> SelectRandomEdges(const std::vector<cv::Point2f>& edges, size_t num_points);
    std::vector<Eigen::Vector3d> ComputeEpipolarLine(const Eigen::Matrix3d& fund_mat, const std::vector<cv::Point2f>& edges);
    void VisualizeOverlay(const std::string& extract_undist_path, const std::string& undistort_extract_path);

    // bool Init_Fetch_Data();
    // Frame::Ptr get_Next_Frame();

    unsigned Total_Num_Of_Imgs;
    // int Current_Frame_Index;
    // double fx, fy, cx, cy;

    std::vector<cv::Mat> undistorted_left_img;
    std::vector<cv::Mat> undistorted_right_img;
    int img_height, img_width;

    std::vector<cv::Point2d> left_third_order_edges_locations;
    std::vector<double> left_third_order_edges_orientation;
    std::vector<cv::Point2d> right_third_order_edges_locations;
    std::vector<double> right_third_order_edges_orientation;

private:

    YAML::Node config_file;

    std::string dataset_type;
    std::string dataset_path;
    std::string sequence_name;

    std::vector<int> left_res;
    int left_rate;
    std::string left_model;
    std::vector<double> left_intr;
    std::string left_dist_model;
    std::vector<double> left_dist_coeffs;

    std::vector<int> right_res;
    int right_rate;
    std::string right_model;
    std::vector<double> right_intr;
    std::string right_dist_model;
    std::vector<double> right_dist_coeffs;

    std::vector<std::vector<double>> R21;
    std::vector<double> T21;
    std::vector<std::vector<double>> F21;

    std::vector<std::vector<double>> R12;
    std::vector<double> T12;
    std::vector<std::vector<double>> F12;
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

    //> CH: shared pointer to the class of third-order edge detector
    std::shared_ptr< ThirdOrderEdgeDetectionCPU > TOED = nullptr;
};

#endif 