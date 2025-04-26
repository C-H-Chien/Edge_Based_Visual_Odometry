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
  std::vector<int> patch_input_counts;
  std::vector<int> patch_output_counts;
  std::vector<int> ncc_input_counts;
  std::vector<int> ncc_output_counts;
  std::vector<int> lowe_input_counts;
  std::vector<int> lowe_output_counts;

  double per_image_epi_precision;
  double per_image_disp_precision;
  double per_image_shift_precision;
  double per_image_clust_precision;
};

struct PatchMatch{
    cv::Point2d coord;
    double orientation;
    double final_score;
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
   std::vector<std::tuple<cv::Point2d, cv::Point2d, double>> gt_edge_data;

   std::vector<std::pair<double, double>> ncc_one_vs_err;
   std::vector<std::pair<double, double>> ncc_two_vs_err;

private:
   YAML::Node config_file;

   std::string dataset_type;
   std::string dataset_path;
   std::string output_path;
   std::string sequence_name;

   std::string GT_file_name;

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
   double focal_length;
   double baseline;
   double max_disparity;

   std::vector<cv::Point2d> matched_left_edges;
   std::vector<cv::Point2d> matched_right_edges;
   std::vector<double> matched_left_orientations;
   std::vector<double> matched_right_orientations;
   std::vector<double> left_edge_depths;

   void write_ncc_vals_to_files( int img_index );

   void PrintDatasetInfo();
   RecallMetrics DisplayMatches(const cv::Mat& left_image, const cv::Mat& right_image, const cv::Mat& left_binary_map, const cv::Mat& right_binary_map, std::vector<cv::Point2d> right_edge_coords, std::vector<double> right_edge_orientations);
   RecallMetrics CalculateMatches(const std::vector<cv::Point2d>& selected_left_edges, const std::vector<cv::Point2d>& selected_ground_truth_right_edges, 
      const std::vector<double>& selected_left_orientations, const std::vector<cv::Point2d>& left_edge_coords, const std::vector<double>& left_edge_orientations, 
      const std::vector<cv::Point2d>& right_edge_coords, const std::vector<double>& right_edge_orientations, const std::vector<cv::Mat>& left_patch_set_one, const std::vector<cv::Mat>& left_patch_set_two,
      const std::vector<Eigen::Vector3d>& epipolar_lines_right, const cv::Mat& left_image, const cv::Mat& right_image, const Eigen::Matrix3d& fundamental_matrix_12, 
      cv::Mat& right_visualization);
   int ComputeNCCScores(const cv::Mat& left_patch_one, const cv::Mat& left_patch_two, const std::vector<cv::Mat>& right_patch_set_one, const std::vector<cv::Mat>& right_patch_set_two, double ncc_threshold);
   double ComputeNCC(const cv::Mat& patch_one, const cv::Mat& patch_two);
   std::pair<std::vector<cv::Point2d>, std::vector<cv::Point2d>> CalculateOrthogonalShifts(const std::vector<cv::Point2d>& edge_points, const std::vector<double>& orientations, double shift_magnitude);
   std::vector<std::pair<std::vector<cv::Point2d>, std::vector<double>>> ClusterEpipolarShiftedEdges(std::vector<cv::Point2d>& valid_corrected_edges, std::vector<double>& valid_corrected_orientations); 
   std::vector<Eigen::Vector3d> ReprojectOrientations(const std::vector<Eigen::Vector3d>& tangent_vectors, std::vector<Eigen::Matrix3d> rot_mat_list);
   std::vector<Eigen::Vector3d> ReconstructOrientations();
   void CalculateDepths();
   void ExtractPatches(
    int patch_size,
    const cv::Mat& image,
    const std::vector<cv::Point2d>& edges,
    const std::vector<double>& orientations,
    const std::vector<cv::Point2d>* right_edges, 
    const std::vector<cv::Point2d>& shifted_one,
    const std::vector<cv::Point2d>& shifted_two,
    std::vector<cv::Point2d>& filtered_edges_out,
    std::vector<double>& filtered_orientations_out,
    std::vector<cv::Point2d>* filtered_right_edges_out,
    std::vector<cv::Mat>& patch_set_one_out,
    std::vector<cv::Mat>& patch_set_two_out);
   std::pair<std::vector<cv::Point2d>, std::vector<double>> ExtractEpipolarEdges(const Eigen::Vector3d& epipolar_line, const std::vector<cv::Point2d>& edge_locations, const std::vector<double>& edge_orientations, double distance_threshold); 
   std::vector<Eigen::Vector3d> CalculateEpipolarLine(const Eigen::Matrix3d& fund_mat, const std::vector<cv::Point2d>& edges);
   std::tuple<std::vector<cv::Point2d>, std::vector<double>, std::vector<cv::Point2d>> PickRandomEdges(int patch_size, const std::vector<cv::Point2d>& edges, const std::vector<cv::Point2d>& ground_truth_right_edges, const std::vector<double>& orientations, size_t num_points, int img_width, int img_height);
   Eigen::Matrix3d ConvertToEigenMatrix(const std::vector<std::vector<double>>& matrix);
   std::vector<std::pair<cv::Mat, cv::Mat>> LoadEuRoCImages(const std::string& csv_path, const std::string& left_path, const std::string& right_path, int num_images);
   std::vector<std::pair<cv::Mat, cv::Mat>> LoadETH3DImages(const std::string &stereo_pairs_path, int num_images);
   std::vector<double> LoadMaximumDisparityValues(const std::string& stereo_pairs_path, int num_images);
   std::vector<cv::Mat> LoadETH3DMaps(const std::string &stereo_pairs_path, int num_maps);
   void WriteDisparityToBinary(const std::string& filepath, const cv::Mat& disparity_map);
   cv::Mat ReadDisparityFromBinary(const std::string& filepath);
   cv::Mat LoadDisparityFromCSV(const std::string& path);
   void WriteEdgesToBinary(const std::string& filepath,
                           const std::vector<cv::Point2d>& locations,
                           const std::vector<double>& orientations); 
   void ReadEdgesFromBinary(const std::string& filepath,
                                   std::vector<cv::Point2d>& locations,
                                   std::vector<double>& orientations);
   void ProcessEdges(const cv::Mat& image,
                     const std::string& filepath,
                     std::shared_ptr<ThirdOrderEdgeDetectionCPU>& toed,
                     std::vector<cv::Point2d>& locations,
                     std::vector<double>& orientations);
   void VisualizeGTRightEdge(const cv::Mat &left_image, const cv::Mat &right_image, const std::vector<std::pair<cv::Point2d, cv::Point2d>> &edge_pairs);
   void CalculateGTRightEdge(const std::vector<cv::Point2d> &left_third_order_edges_locations, const std::vector<double> &left_third_order_edges_orientation, const cv::Mat &disparity_map, const cv::Mat &left_image, const cv::Mat &right_image);
   cv::Point2d PerformEpipolarShift( cv::Point2d original_edge_location, double edge_orientation, std::vector<double> epipolar_line_coeffs, bool& b_pass_epipolar_tengency_check);
   void Load_GT_Poses( std::string GT_Poses_File_Path );
   std::vector<double> GT_time_stamps;
   std::vector<double> Img_time_stamps;
   void Align_Images_and_GT_Poses();

   bool compute_grad_depth = false;
   cv::Mat Gx_2d, Gy_2d;
   cv::Mat Small_Patch_Radius_Map;
   Utility::Ptr utility_tool = nullptr;

   //> CH: shared pointer to the class of third-order edge detector
   std::shared_ptr< ThirdOrderEdgeDetectionCPU > TOED = nullptr;
};


#endif