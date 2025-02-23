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
//    Lopez  25-01-26    Modified for euroc dataset support.
//    Chien  23-01-17    Initially created.
//
//> (c) LEMS, Brown University
//> Chiang-Heng Chien (chiang-heng_chien@brown.edu), Saul Lopez Lucas (saul_lopez_lucas@brown.edu)
// =======================================================================================================

class Dataset {

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Dataset> Ptr;
    Dataset(YAML::Node, bool);

    void PerformEdgeBasedVO();
    
    unsigned Total_Num_Of_Imgs;
    int img_height, img_width;

    std::vector<cv::Point2d> left_third_order_edges_locations;
    std::vector<double> left_third_order_edges_orientation;
    std::vector<cv::Point2d> right_third_order_edges_locations;
    std::vector<double> right_third_order_edges_orientation;

    std::vector<Eigen::Matrix3d> unaligned_GT_Rot;
    std::vector<Eigen::Vector3d> unaligned_GT_Transl;
    std::vector<Eigen::Matrix3d> aligned_GT_Rot;
    std::vector<Eigen::Vector3d> aligned_GT_Transl;

    // bool Init_Fetch_Data();
    // Frame::Ptr get_Next_Frame();
    // int Current_Frame_Index;
    // double fx, fy, cx, cy;

private:
    YAML::Node config_file;

    std::string dataset_type;
    std::string dataset_path;
    std::string sequence_name;
    std::string GT_file_name;

    //> Used only for the EuRoC dataset
    Eigen::Matrix3d rot_frame2body_left;
    Eigen::Vector3d transl_frame2body_left;

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

    std::vector<std::vector<double>> rot_mat_21;
    std::vector<double> trans_vec_21;
    std::vector<std::vector<double>> fund_mat_21;

    std::vector<std::vector<double>> rot_mat_12;
    std::vector<double> trans_vec_12;
    std::vector<std::vector<double>> fund_mat_12;

    std::vector<cv::Point2d> matched_left_edges;
    std::vector<cv::Point2d> matched_right_edges;
    std::vector<double> matched_left_orientations;
    std::vector<double> matched_right_orientations;
    std::vector<double> left_edge_depths; 
    std::vector<Eigen::Vector3d> left_edge_3D_orientations;

    // void PrintDatasetInfo();
    void DisplayMatches(const cv::Mat& left_image, const cv::Mat& right_image, const cv::Mat& left_binary_map, const cv::Mat& right_binary_map, std::vector<cv::Point2d> left_edge_coords, std::vector<cv::Point2d> right_edge_coords, std::vector<double> left_edge_orientations, std::vector<double> right_edge_orientations);
    void CalculateMatches(const std::vector<cv::Point2d>& selected_left_edges, const std::vector<double>& selected_left_orientations, const std::vector<cv::Point2d>& left_edge_coords, const std::vector<double>& left_edge_orientations, const std::vector<cv::Point2d>& right_edge_coords, const std::vector<double>& right_edge_orientations, const std::vector<cv::Mat>& left_patches, const std::vector<Eigen::Vector3d>& epipolar_lines_right, const cv::Mat& left_image, const cv::Mat& right_image, const Eigen::Matrix3d& fundamental_matrix_12, cv::Mat& right_visualization);
    void CalculateOrientations();
    void CalculateDepths();
    int CalculateNCCPatch(const cv::Mat& left_patch, const std::vector<cv::Mat>& right_patches);
    void ExtractPatches(int patch_size, const cv::Mat& image, const std::vector<cv::Point2d>& selected_edges, std::vector<cv::Mat>& patches);
    void UndistortEdges(const cv::Mat& dist_edges, cv::Mat& undist_edges, std::vector<cv::Point2f>& edge_locations, const std::vector<double>& intr, const std::vector<double>& dist_coeffs);
    void DisplayOverlay(const std::string& extract_undist_path, const std::string& undistort_extract_path);
    std::pair<std::vector<cv::Point2d>, std::vector<double>> ExtractEpipolarEdges(int patch_size, const Eigen::Vector3d& epipolar_line, const std::vector<cv::Point2d>& edge_locations, const std::vector<double>& edge_orientations);  
    std::vector<Eigen::Vector3d> CalculateEpipolarLine(const Eigen::Matrix3d& fund_mat, const std::vector<cv::Point2d>& edges);
    std::pair<std::vector<cv::Point2d>, std::vector<double>> PickRandomEdges(int patch_size, const std::vector<cv::Point2d>& edges, const std::vector<double>& orientations, size_t num_points, int img_width, int img_height);
    Eigen::Matrix3d ConvertToEigenMatrix(const std::vector<std::vector<double>>& matrix);
    std::vector<std::pair<cv::Mat, cv::Mat>> LoadImages(const std::string& csv_path, const std::string& left_path, const std::string& right_path, int num_images);

    void Load_GT_Poses();
    std::vector<double> GT_time_stamps;

    bool compute_grad_depth = false;
    cv::Mat Gx_2d, Gy_2d;
    cv::Mat Small_Patch_Radius_Map;
    Utility::Ptr utility_tool = nullptr;

    //> CH: shared pointer to the class of third-order edge detector
    std::shared_ptr< ThirdOrderEdgeDetectionCPU > TOED = nullptr;

    // Eigen::Matrix3d Calib;        
    // Eigen::Matrix3d Inverse_Calib; 

    // std::vector<std::string> Img_Path_List;
    // std::vector<std::string> Depth_Path_List;
    // std::vector<std::string> Img_Time_Stamps;

    // bool has_Depth;
    // cv::Mat grad_Depth_xi_;
    // cv::Mat grad_Depth_eta_;
};

#endif 