#ifndef DATASET_H
#define DATASET_H
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <yaml-cpp/yaml.h>
#include <opencv2/opencv.hpp>
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

struct RecallMetrics {
  double epi_distance_recall;
  double max_disparity_recall;
  double epi_shift_recall;
  double epi_cluster_recall;
  double patch_recall;
  double ncc_recall;
  double lowe_recall;


  std::vector<int> epi_input_counts;
  std::vector<int> epi_output_counts;
  std::vector<int> disp_input_counts;
  std::vector<int> disp_output_counts;
  std::vector<int> shift_input_counts;
  std::vector<int> shift_output_counts;
  std::vector<int> clust_input_counts;
  std::vector<int> clust_output_counts;
  std::vector<int> ncc_input_counts;
  std::vector<int> ncc_output_counts;
};


extern cv::Mat merged_visualization_global;
class Dataset {
public:
   EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
   typedef std::shared_ptr<Dataset> Ptr;
   Dataset(YAML::Node, bool);

   void PerformEdgeBasedVO();
   static void onMouse(int event, int x, int y, int, void*);

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

   std::vector<Eigen::Matrix3d> unaligned_GT_Rot;
   std::vector<Eigen::Vector3d> unaligned_GT_Transl;
   std::vector<Eigen::Matrix3d> aligned_GT_Rot;
   std::vector<Eigen::Vector3d> aligned_GT_Transl;
   std::vector<std::tuple<cv::Point2d, cv::Point2d, double>> gt_edge_data;

   std::vector<std::pair<double, double>> ncc_one_vs_err;
   std::vector<std::pair<double, double>> ncc_two_vs_err;

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

    // void PrintDatasetInfo();
    void DisplayMatches(const cv::Mat& left_map, const cv::Mat& right_map, std::vector<cv::Point2f> left_edge_coords);
    void CalculateMatches(const std::vector<cv::Point2f>& selected_left_edges, const std::vector<cv::Mat>& left_patches, const std::vector<Eigen::Vector3d>& 
        epipolar_lines_right,const cv::Mat& left_map, const cv::Mat& right_map, const Eigen::Matrix3d& fundamental_matrix_12, cv::Mat& right_visualization);     
    int CalculateSSDPatch(const cv::Mat& left_patch, const std::vector<cv::Mat>& right_patches);
    void ExtractPatches(int patch_size, const cv::Mat& binary_map, const std::vector<cv::Point2f>& selected_edges, std::vector<cv::Mat>& patches);
    void UndistortEdges(const cv::Mat& dist_edges, cv::Mat& undist_edges, std::vector<cv::Point2f>& edge_locations, const std::vector<double>& intr, const std::vector<double>& dist_coeffs);
    void DisplayOverlay(const std::string& extract_undist_path, const std::string& undistort_extract_path);
    std::vector<cv::Point2f> ExtractEpipolarEdges(int patch_size, const Eigen::Vector3d& epipolar_line, const cv::Mat& binary_map);     
    std::vector<Eigen::Vector3d> CalculateEpipolarLine(const Eigen::Matrix3d& fund_mat, const std::vector<cv::Point2f>& edges);
    std::vector<cv::Point2f> PickRandomEdges(int patch_size, const std::vector<cv::Point2f>& edges, size_t num_points, int img_width, int img_height);
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
};


#endif