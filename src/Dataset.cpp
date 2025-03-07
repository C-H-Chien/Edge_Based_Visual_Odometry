#ifndef DATASET_CPP
#define DATASET_CPP
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <glog/logging.h>
#include <time.h>
#include <filesystem>
#include <sys/time.h>
#include <random>
#include <unordered_set>
#include <vector>
#include <chrono>
#include "Dataset.h"
#include "definitions.h"


// =======================================================================================================
// Class Dataset: Fetch data from dataset specified in the configuration file
//
// ChangeLogs
//    Lopez  25-01-26    Modified for euroc dataset support.
//    Chien  23-01-17    Initially created.
//
//> (c) LEMS, Brown University
//> Chiang-Heng Chien (chiang-heng_chien@brown.edu), Saul Lopez Lucas (saul_lopez_lucas@brown.edu)
// =======================================================================================================


Dataset::Dataset(YAML::Node config_map, bool use_GCC_filter) : config_file(config_map), compute_grad_depth(use_GCC_filter) {
   dataset_path = config_file["dataset_dir"].as<std::string>();
   sequence_name = config_file["sequence_name"].as<std::string>();
   dataset_type = config_file["dataset_type"].as<std::string>();

   if (dataset_type == "EuRoC") {
       try {
           GT_file_name = config_file["state_GT_estimate_file_name"].as<std::string>();

           YAML::Node left_cam = config_file["left_camera"];
           YAML::Node right_cam = config_file["right_camera"];
           YAML::Node stereo = config_file["stereo"];
           YAML::Node frame_to_body = config_file["frame_to_body"];

           left_res = left_cam["resolution"].as<std::vector<int>>();
           left_intr = left_cam["intrinsics"].as<std::vector<double>>();
           left_dist_coeffs = left_cam["distortion_coefficients"].as<std::vector<double>>();

           right_res = right_cam["resolution"].as<std::vector<int>>();
           right_intr = right_cam["intrinsics"].as<std::vector<double>>();
           right_dist_coeffs = right_cam["distortion_coefficients"].as<std::vector<double>>();

           if (stereo["R21"] && stereo["T21"] && stereo["F21"]) {
               for (const auto& row : stereo["R21"]) {
                   rot_mat_21.push_back(row.as<std::vector<double>>());
               }

               trans_vec_21 = stereo["T21"].as<std::vector<double>>();

               for (const auto& row : stereo["F21"]) {
                   fund_mat_21.push_back(row.as<std::vector<double>>());
               }
           } else {
               std::cerr << "ERROR: Missing left-to-right stereo parameters (R21, T21, F21) in YAML file!" << std::endl;
           }

           if (stereo["R12"] && stereo["T12"] && stereo["F12"]) {
               for (const auto& row : stereo["R12"]) {
                   rot_mat_12.push_back(row.as<std::vector<double>>());
               }
               trans_vec_12 = stereo["T12"].as<std::vector<double>>();

               for (const auto& row : stereo["F12"]) {
                   fund_mat_12.push_back(row.as<std::vector<double>>());
               }
           } else {
               std::cerr << "ERROR: Missing right-to-left stereo parameters (R12, T12, F12) in YAML file!" << std::endl;
           }

            if (frame_to_body["rotation"] && frame_to_body["translation"]) {
                rot_frame2body_left = Eigen::Map<Eigen::Matrix3d>(frame_to_body["rotation"].as<std::vector<double>>().data()).transpose();
                transl_frame2body_left = Eigen::Map<Eigen::Vector3d>(frame_to_body["translation"].as<std::vector<double>>().data());
            } else {
                LOG_ERROR("Missing relative rotation and translation from the left camera to the body coordinate (should be given by cam0/sensor.yaml)");
            }

        } catch (const YAML::Exception &e) {
            std::cerr << "ERROR: Could not parse YAML file! " << e.what() << std::endl;
        }
    }
    else if (dataset_type == "ETH3D")
       try {
           YAML::Node left_cam = config_file["left_camera"];
           YAML::Node right_cam = config_file["right_camera"];
           YAML::Node stereo = config_file["stereo"];

           left_res = left_cam["resolution"].as<std::vector<int>>();
           left_intr = left_cam["intrinsics"].as<std::vector<double>>();

           right_res = right_cam["resolution"].as<std::vector<int>>();
           right_intr = right_cam["intrinsics"].as<std::vector<double>>();

           if (stereo["R21"] && stereo["T21"] && stereo["F21"]) {
               for (const auto& row : stereo["R21"]) {
                   rot_mat_21.push_back(row.as<std::vector<double>>());
               }

               trans_vec_21 = stereo["T21"].as<std::vector<double>>();

               for (const auto& row : stereo["F21"]) {
                   fund_mat_21.push_back(row.as<std::vector<double>>());
               }
           } else {
               std::cerr << "ERROR: Missing left-to-right stereo parameters (R21, T21, F21) in YAML file!" << std::endl;
           }

           if (stereo["R12"] && stereo["T12"] && stereo["F12"]) {
               for (const auto& row : stereo["R12"]) {
                   rot_mat_12.push_back(row.as<std::vector<double>>());
               }
               trans_vec_12 = stereo["T12"].as<std::vector<double>>();

               for (const auto& row : stereo["F12"]) {
                   fund_mat_12.push_back(row.as<std::vector<double>>());
               }
           } else {
               std::cerr << "ERROR: Missing right-to-left stereo parameters (R12, T12, F12) in YAML file!" << std::endl;
           }
        } catch (const YAML::Exception &e) {
            std::cerr << "ERROR: Could not parse YAML file! " << e.what() << std::endl;
        }
    
   Total_Num_Of_Imgs = 0;
}


void Dataset::PerformEdgeBasedVO() {
    int num_images = 437;
    std::vector<std::pair<cv::Mat, cv::Mat>> image_pairs;

    if (dataset_type == "EuRoC"){
        std::string left_path = dataset_path + "/" + sequence_name + "/mav0/cam0/data/";
        std::string right_path = dataset_path + "/" + sequence_name + "/mav0/cam1/data/";
        std::string image_csv_path = dataset_path + "/" + sequence_name + "/mav0/cam0/data.csv";
        std::string ground_truth_path = dataset_path + "/" + sequence_name + "/mav0/state_groundtruth_estimate0/data.csv";

        image_pairs = LoadEuRoCImages(image_csv_path, left_path, right_path, num_images);

        Load_GT_Poses(ground_truth_path);
        Align_Images_and_GT_Poses();
    }
    else if (dataset_type == "ETH3D"){
        std::string stereo_pairs_path = dataset_path + "/" + sequence_name + "/stereo_pairs";
        image_pairs = LoadETH3DImages(stereo_pairs_path, num_images);

        // for (size_t i = 0; i < image_pairs.size(); ++i) {
        // cv::Mat left_image = image_pairs[i].first;
        // cv::Mat right_image = image_pairs[i].second;

        // cv::Mat merged_visualization;
        // cv::hconcat(left_image, right_image, merged_visualization);

        // cv::imshow("Stereo Image Pair-Chronological Order Check", merged_visualization);

        // std::cout << "Displaying stereo pair " << i + 1 << " of " << image_pairs.size() << std::endl;

        // char key = cv::waitKey(0); 
        // if (key == 27) break;
        // }
        // cv::destroyAllWindows();
    }
    
   for (const auto& pair : image_pairs) {
       const cv::Mat& left_img = pair.first;
       const cv::Mat& right_img = pair.second; {
       cv::Mat left_calib = (cv::Mat_<double>(3, 3) << left_intr[0], 0, left_intr[2], 0, left_intr[1], left_intr[3], 0, 0, 1);
       cv::Mat right_calib = (cv::Mat_<double>(3, 3) << right_intr[0], 0, right_intr[2], 0, right_intr[1], right_intr[3], 0, 0, 1);
       cv::Mat left_dist_coeff_mat(left_dist_coeffs);
       cv::Mat right_dist_coeff_mat(right_dist_coeffs);
    
       cv::Mat left_undistorted, right_undistorted;
       cv::undistort(left_img, left_undistorted, left_calib, left_dist_coeff_mat);
       cv::undistort(right_img, right_undistorted, right_calib, right_dist_coeff_mat);

       //> CH: stack all the undistorted images
       if (Total_Num_Of_Imgs == 0) {
           img_height = left_undistorted.rows;
           img_width  = left_undistorted.cols;
           //> CH: initiate a TOED constructor
           TOED = std::shared_ptr<ThirdOrderEdgeDetectionCPU>(new ThirdOrderEdgeDetectionCPU( img_height, img_width ));
       }

       //> CH: get third-order edges
       //> (i) left undistorted image
       std::cout << "Processing third-order edges on the left image... " << std::endl;
       TOED->get_Third_Order_Edges( left_undistorted );
       left_third_order_edges_locations = TOED->toed_locations;
       left_third_order_edges_orientation = TOED->toed_orientations;
       std::cout << "Number of third-order edges on the left image: " << TOED->Total_Num_Of_TOED << std::endl;

       //> (ii) right undistorted image
       std::cout << "Processing third-order edges on the right image... " << std::endl;
       TOED->get_Third_Order_Edges( right_undistorted );
       right_third_order_edges_locations = TOED->toed_locations;
       right_third_order_edges_orientation = TOED->toed_orientations;
       std::cout << "Number of third-order edges on the right image: " << TOED->Total_Num_Of_TOED << std::endl;

       Total_Num_Of_Imgs++;

       // Initialize empty binary maps for edges
       cv::Mat left_edge_map = cv::Mat::zeros(left_undistorted.size(), CV_8UC1);
       cv::Mat right_edge_map = cv::Mat::zeros(right_undistorted.size(), CV_8UC1);

       // Convert edge locations to binary maps
       for (const auto& edge : left_third_order_edges_locations) {
           if (edge.x >= 0 && edge.x < left_edge_map.cols && edge.y >= 0 && edge.y < left_edge_map.rows) {
               left_edge_map.at<uchar>(cv::Point(edge.x, edge.y)) = 255;
           }
       }

       for (const auto& edge : right_third_order_edges_locations) {
           if (edge.x >= 0 && edge.x < right_edge_map.cols && edge.y >= 0 && edge.y < right_edge_map.rows) {
               right_edge_map.at<uchar>(cv::Point(edge.x, edge.y)) = 255;
           }
       }

    //    DisplayMatches(left_undistorted, right_undistorted, left_edge_map, right_edge_map, left_third_order_edges_locations, right_third_order_edges_locations, left_third_order_edges_orientation, right_third_order_edges_orientation);
       }
   }
}

void Dataset::DisplayMatches(const cv::Mat& left_image, const cv::Mat& right_image, const cv::Mat& left_binary_map, const cv::Mat& right_binary_map,
   std::vector<cv::Point2d> left_edge_coords, std::vector<cv::Point2d> right_edge_coords, std::vector<double> left_edge_orientations, std::vector<double> right_edge_orientations) {
   cv::Mat left_visualization, right_visualization;
   cv::cvtColor(left_binary_map, left_visualization, cv::COLOR_GRAY2BGR);
   cv::cvtColor(right_binary_map, right_visualization, cv::COLOR_GRAY2BGR);

   std::pair<std::vector<cv::Point2d>, std::vector<double>> selected_data = PickRandomEdges(7, left_edge_coords, left_edge_orientations, 5, left_res[0], left_res[1]);

   std::vector<cv::Point2d> selected_left_edges = selected_data.first;
   std::vector<double> selected_left_orientations = selected_data.second;

   for (const auto& point : selected_left_edges) {
       cv::circle(left_visualization, point, 5, cv::Scalar(0, 0, 255), cv::FILLED);
   }

   std::vector<cv::Mat> left_patches;
   ExtractPatches(7, left_image, selected_left_edges, left_patches);

   Eigen::Matrix3d fundamental_matrix_21 = ConvertToEigenMatrix(fund_mat_21);
   Eigen::Matrix3d fundamental_matrix_12 = ConvertToEigenMatrix(fund_mat_12);
   std::vector<Eigen::Vector3d> epipolar_lines_right = CalculateEpipolarLine(fundamental_matrix_21, selected_left_edges);

   CalculateMatches(selected_left_edges, selected_left_orientations, left_edge_coords, left_edge_orientations, right_edge_coords, right_edge_orientations, left_patches, epipolar_lines_right, left_image, right_image,
       fundamental_matrix_12, right_visualization);


    CalculateDepths();
    std::vector<Eigen::Vector3d> tangent_vectors = ReconstructOrientations();
    std::vector<Eigen::Vector3d> reprojected_orientations = ReprojectOrientations(tangent_vectors, aligned_GT_Rot);

   std::cout << "Reprojected Orientations:" << std::endl;
    for (size_t i = 0; i < reprojected_orientations.size(); i++) {
        std::cout << "Reprojected Orientation " << (reprojected_orientations[i])(0) << "\t" << (reprojected_orientations[i])(1) << "\t" << (reprojected_orientations[i])(2) << std::endl;
    }


   cv::Mat merged_visualization;
   cv::hconcat(left_visualization, right_visualization, merged_visualization);
   cv::imshow("Edge Matching Using NCC & Bidirectional Consistency", merged_visualization);
   cv::waitKey(0);
}

void Dataset::CalculateMatches(const std::vector<cv::Point2d>& selected_left_edges, const std::vector<double>& selected_left_orientations, const std::vector<cv::Point2d>& left_edge_coords, const std::vector<double>& left_edge_orientations, const std::vector<cv::Point2d>& right_edge_coords, const std::vector<double>& right_edge_orientations, const std::vector<cv::Mat>& left_patches, const std::vector<Eigen::Vector3d>& epipolar_lines_right, const cv::Mat& left_image, const cv::Mat& right_image, const Eigen::Matrix3d& fundamental_matrix_12, cv::Mat& right_visualization) {
   matched_left_edges.clear();
   matched_right_edges.clear();
   matched_left_orientations.clear();
   matched_right_orientations.clear();
  
   double tol = 2.0;

   for (size_t i = 0; i < selected_left_edges.size(); i++) {
       const auto& left_edge = selected_left_edges[i];
       const auto& left_orientation = selected_left_orientations[i];
       const auto& left_patch = left_patches[i];
       const auto& epipolar_line = epipolar_lines_right[i];
       double a = epipolar_line(0);
       double b = epipolar_line(1);
       double c = epipolar_line(2);

       if (b != 0) {
           cv::Point2d pt1(0, -c / b);
           cv::Point2d pt2(right_visualization.cols, -(c + a * right_visualization.cols) / b);
           cv::line(right_visualization, pt1, pt2, cv::Scalar(255, 200, 100), 1);
       }

       std::pair<std::vector<cv::Point2d>, std::vector<double>> right_candidates = ExtractEpipolarEdges(7, epipolar_line, right_edge_coords, right_edge_orientations);
       std::vector<cv::Point2d> right_candidate_edges = right_candidates.first;
       std::vector<double> right_candidate_orientations = right_candidates.second;

       std::vector<cv::Mat> right_patches;
       ExtractPatches(7, right_image, right_candidate_edges, right_patches);
      
       if (!left_patch.empty() && !right_patches.empty()) {
           int best_right_match_idx = CalculateNCCPatch(left_patch, right_patches);
          
           if (best_right_match_idx != -1) {
               cv::Point2d best_right_match = right_candidate_edges[best_right_match_idx];
               double best_right_orientation = right_candidate_orientations[best_right_match_idx];

               std::vector<Eigen::Vector3d> right_to_left_epipolar = CalculateEpipolarLine(fundamental_matrix_12, {best_right_match});
               Eigen::Vector3d epipolar_line_left = right_to_left_epipolar[0];

               std::pair<std::vector<cv::Point2d>, std::vector<double>> left_candidates = ExtractEpipolarEdges(7, epipolar_line_left, left_edge_coords, left_edge_orientations);
               std::vector<cv::Point2d> left_candidate_edges = left_candidates.first;
               std::vector<double> left_candidate_orientations = left_candidates.second;

               std::vector<cv::Mat> left_candidate_patches;
               ExtractPatches(7, left_image, left_candidate_edges, left_candidate_patches);
              
               if (!left_candidate_patches.empty()) {
                   int best_left_match_idx = CalculateNCCPatch(right_patches[best_right_match_idx], left_candidate_patches);
                  
                   if (best_left_match_idx != -1) {
                       cv::Point2d best_left_match = left_candidate_edges[best_left_match_idx];

                       if (cv::norm(best_left_match - left_edge) < tol) {
                           cv::circle(right_visualization, best_right_match, 5, cv::Scalar(0, 0, 255), cv::FILLED);
                           matched_left_edges.push_back(left_edge);
                           matched_left_orientations.push_back(left_orientation);
                           matched_right_edges.push_back(best_right_match);
                           matched_right_orientations.push_back(best_right_orientation);
                       }
                   }
               }
           }
       }
   }
}

std::vector<Eigen::Vector3d> Dataset::ReprojectOrientations(const std::vector<Eigen::Vector3d>& tangent_vectors, std::vector<Eigen::Matrix3d> rot_mat_list){
    // std::cout << "Vector Sizes:" << std::endl;
    // std::cout << "tangent_vectors.size(): " << tangent_vectors.size() << std::endl;
    // std::cout << "rot_mat_list.size(): " << rot_mat_list.size() << std::endl;
    // std::cout << "matched_left_edges.size(): " << matched_left_edges.size() << std::endl;
    // std::cout << "left_edge_depths.size(): " << left_edge_depths.size() << std::endl;

    if (tangent_vectors.size() != matched_left_edges.size() ||
       tangent_vectors.size() != left_edge_depths.size()) {
       std::cerr << "ERROR: Mismatch in vector sizes!" << std::endl;
       return {};
   }

   Eigen::Vector3d e3(0, 0, 1);
   std::vector<Eigen::Vector3d> reprojected_orientations;

   for (size_t i = 0; i < tangent_vectors.size(); i++) {
       Eigen::Vector3d T = rot_mat_list[i] * tangent_vectors[i];

       Eigen::Vector3d small_gamma(matched_left_edges[i].x, matched_left_edges[i].y, 1.0);
       Eigen::Vector3d big_gamma = left_edge_depths[i] * small_gamma;

       Eigen::Vector3d numerator = (e3.transpose() * big_gamma) * T - (e3.transpose() * T) * big_gamma;
       double denominator = numerator.norm();

       Eigen::Vector3d t = numerator / denominator;
       reprojected_orientations.push_back(t);
   }
   return reprojected_orientations;
}

//FIXED
std::vector<Eigen::Vector3d> Dataset::ReconstructOrientations() {
   if (matched_left_edges.size() != matched_right_edges.size() ||
       matched_left_edges.size() != matched_left_orientations.size() ||
       matched_right_edges.size() != matched_right_orientations.size()) {
       std::cerr << "ERROR: Mismatch in number of edge matches and orientations!" << std::endl;
       return {};
   }

   Eigen::Matrix3d R21 = ConvertToEigenMatrix(rot_mat_21);
  
   Eigen::Matrix3d K_left;
   K_left << left_intr[0], 0, left_intr[2],
        0, left_intr[1], left_intr[3],
        0, 0, 1;
  
   Eigen::Matrix3d K_right;
   K_right << right_intr[0], 0, right_intr[2],
        0, right_intr[1], right_intr[3],
        0, 0, 1;

   Eigen::Matrix3d K_left_inv = K_left.inverse();
   Eigen::Matrix3d K_right_inv = K_right.inverse();

   std::vector<Eigen::Vector3d> reconstructed_orientations;

   for (size_t i = 0; i < matched_left_edges.size(); i++) {
       Eigen::Vector3d gamma_one(matched_left_edges[i].x, matched_left_edges[i].y, 1.0);
       Eigen::Vector3d gamma_two(matched_right_edges[i].x, matched_right_edges[i].y, 1.0);

       Eigen::Vector3d gamma_one_meter = K_left_inv * gamma_one;
       Eigen::Vector3d gamma_two_meter = K_right_inv * gamma_two;

       double theta_one = matched_left_orientations[i];
       double theta_two = matched_right_orientations[i];

       Eigen::Vector3d t_one(std::cos(theta_one), std::sin(theta_one), 0);
       Eigen::Vector3d t_two(std::cos(theta_two), std::sin(theta_two), 0);

       Eigen::Vector3d r_t = R21 * t_one;
       Eigen::Vector3d r_gamma = R21 * gamma_one_meter;

       Eigen::Vector3d t_cross_r_t = t_two.cross(r_t);
       Eigen::Vector3d t_cross_r_gamma = t_two.cross(r_gamma);

       Eigen::Vector3d numerator = -(gamma_two_meter.dot(t_cross_r_t)) * gamma_one_meter + (gamma_two_meter.dot(t_cross_r_gamma)) * t_one;
       double denominator = numerator.norm();

       Eigen::Vector3d T1 = numerator / denominator;

       reconstructed_orientations.push_back(T1);
   }
   return reconstructed_orientations;
}

//FIXED
void Dataset::CalculateDepths() {
   if (matched_left_edges.size() != matched_right_edges.size()) {
       std::cerr << "ERROR: Number of left and right edge matches do not match!" << std::endl;
       return;
   }

   Eigen::Matrix3d R = ConvertToEigenMatrix(rot_mat_21);
   Eigen::Vector3d T;

   for (int i = 0; i < 3; i++) {
       T(i) = trans_vec_21[i];
   }

   Eigen::Matrix3d K_left;
   K_left << left_intr[0], 0, left_intr[2],
        0, left_intr[1], left_intr[3],
        0, 0, 1;
  
   Eigen::Matrix3d K_right;
   K_right << right_intr[0], 0, right_intr[2],
        0, right_intr[1], right_intr[3],
        0, 0, 1;

   Eigen::Matrix3d K_left_inv = K_left.inverse();
   Eigen::Matrix3d K_right_inv = K_right.inverse();

   Eigen::Vector3d e1(1, 0, 0);
   Eigen::Vector3d e3(0, 0, 1);

   left_edge_depths.clear();

   for (size_t i = 0; i < matched_left_edges.size(); i++) {
       Eigen::Vector3d gamma(matched_left_edges[i].x, matched_left_edges[i].y, 1.0);
       Eigen::Vector3d gamma_bar(matched_right_edges[i].x, matched_right_edges[i].y, 1.0);

       Eigen::Vector3d gamma_meter = K_left_inv * gamma;
       Eigen::Vector3d gamma_bar_meter = K_right_inv * gamma_bar;

       double e1_gamma_bar = (e1.transpose() * gamma_bar_meter)(0, 0);
       double e3_R_gamma = (e3.transpose() * R * gamma_meter)(0, 0);
       double e1_R_gamma = (e1.transpose() * R * gamma_meter)(0, 0);
       double e1_T = (e1.transpose() * T)(0, 0);
       double e3_T = (e3.transpose() * T)(0, 0);

       double numerator = (e1_T * e3_R_gamma) - (e3_T * e1_R_gamma);
       double denominator = (e3_R_gamma * e1_gamma_bar) - e1_R_gamma;

       if (std::abs(denominator) > 1e-6) {
           double rho = numerator / denominator;
           left_edge_depths.push_back(rho);
       } else {
           std::cerr << "WARNING: Skipping depth computation for edge " << i << " due to near-zero denominator!" << std::endl;
           left_edge_depths.push_back(0.0);
       }
   }

   std::cout << "Computed depths for " << left_edge_depths.size() << " edges:\n";
   for (size_t i = 0; i < left_edge_depths.size(); i++) {
       std::cout << "Edge " << i + 1 << ": Depth = " << left_edge_depths[i] << " meters\n";
   }
}

//UPDATED
int Dataset::CalculateNCCPatch(const cv::Mat& left_patch, const std::vector<cv::Mat>& right_patches) {
   if (right_patches.empty()) return -1;

   int best_match_idx = -1;
   double best_score = -1.0;
  
   for (size_t i = 0; i < right_patches.size(); i++) {
       cv::Mat result;
       cv::matchTemplate(right_patches[i], left_patch, result, cv::TM_CCORR_NORMED);
       double score = result.at<float>(0, 0);
      
       if (score > best_score) {
           best_score = score;
           best_match_idx = static_cast<int>(i);
       }
   }

   if (best_score < 0.85) return -1;
   return best_match_idx;
}

//UPDATED
void Dataset::ExtractPatches(int patch_size, const cv::Mat& image, const std::vector<cv::Point2d>& selected_edges, std::vector<cv::Mat>& patches) {
   int half_patch = patch_size / 2;
   patches.clear();

   for (const auto& edge : selected_edges) {
       double x = edge.x;
       double y = edge.y;

       if (x - half_patch >= 0 && x + half_patch < image.cols &&
           y - half_patch >= 0 && y + half_patch < image.rows) {

           cv::Mat patch;
           cv::Point2f center (static_cast<float>(x), static_cast<float>(y));
           cv::Size size (patch_size, patch_size);
           cv::getRectSubPix(image, size, center, patch);
          
           if (patch.type() != CV_32F) {
               patch.convertTo(patch, CV_32F);
           }

           patches.push_back(patch);
       }
       else {
           std::cerr << "WARNING: Skipped patch at (" << x << ", " << y << ") due to boundary constraints!" << std::endl;
       }
   }
}

//UPDATED
std::pair<std::vector<cv::Point2d>, std::vector<double>> Dataset::ExtractEpipolarEdges(int patch_size, const Eigen::Vector3d& epipolar_line, const std::vector<cv::Point2d>& edge_locations, const std::vector<double>& edge_orientations) {
   std::vector<cv::Point2d> extracted_edges;
   std::vector<double> extracted_orientations;

   double threshold = 3.0;

   if (edge_locations.size() != edge_orientations.size()) {
       throw std::runtime_error("Edge locations and orientations size mismatch.");
   }

    for (size_t i = 0; i < edge_locations.size(); ++i) {
       const auto& edge = edge_locations[i];
       double x = edge.x;
       double y = edge.y;

       double distance = std::abs(epipolar_line(0) * x + epipolar_line(1) * y + epipolar_line(2))
                         / std::sqrt((epipolar_line(0) * epipolar_line(0)) + (epipolar_line(1) * epipolar_line(1)));

       if (distance < threshold) {
           extracted_edges.push_back(edge);
           extracted_orientations.push_back(edge_orientations[i]);
       }
   }

   return {extracted_edges, extracted_orientations};
}

//UPDATED
std::vector<Eigen::Vector3d> Dataset::CalculateEpipolarLine(const Eigen::Matrix3d& fund_mat, const std::vector<cv::Point2d>& edges) {
   std::vector<Eigen::Vector3d> epipolar_lines;

   for (const auto& point : edges) {
       Eigen::Vector3d homo_point(point.x, point.y, 1.0); 

       Eigen::Vector3d epipolar_line = fund_mat * homo_point;

       epipolar_lines.push_back(epipolar_line);

       std::cout << "Epipolar Line Equation for Point (" << point.x << ", " << point.y << "): "
                 << epipolar_line(0) << "x + "
                 << epipolar_line(1) << "y + "
                 << epipolar_line(2) << " = 0" << std::endl;
   }

   return epipolar_lines;
}

//UPDATED
std::pair<std::vector<cv::Point2d>, std::vector<double>> Dataset::PickRandomEdges(int patch_size, const std::vector<cv::Point2d>& edges, const std::vector<double>& orientations, size_t num_points, int img_width, int img_height) {
  std::vector<double> valid_orientations;
  std::vector<cv::Point2d> valid_edges;
  int half_patch = patch_size / 2;

  if (edges.size() != orientations.size()) {
       throw std::runtime_error("Edge locations and orientations size mismatch.");
   }

  for (size_t i = 0; i < edges.size(); ++i) {
       const auto& edge = edges[i];
       if (edge.x >= half_patch && edge.x < img_width - half_patch &&
           edge.y >= half_patch && edge.y < img_height - half_patch) {
           valid_edges.push_back(edge);
           valid_orientations.push_back(orientations[i]);
       }
   }

  num_points = std::min(num_points, valid_edges.size());

  std::vector<cv::Point2d> selected_points;
  std::vector<double> selected_orientations;
  std::unordered_set<int> used_indices;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dis(0, valid_edges.size() - 1);

  while (selected_points.size() < num_points) {
      int index = dis(gen);
      if (used_indices.find(index) == used_indices.end()) {
          selected_points.push_back(valid_edges[index]);
          selected_orientations.push_back(valid_orientations[index]);
          used_indices.insert(index);
      }
  }

  return {selected_points, selected_orientations};
}

std::vector<std::pair<cv::Mat, cv::Mat>> Dataset::LoadEuRoCImages(const std::string& csv_path, const std::string& left_path, const std::string& right_path,
   int num_images) {
   std::ifstream csv_file(csv_path);
   if (!csv_file.is_open()) {
       std::cerr << "ERROR: Could not open the CSV file located at " << csv_path << "!" << std::endl;
       return {};
   }

   std::vector<std::pair<cv::Mat, cv::Mat>> image_pairs;
   std::string line;
   bool first_line = true;

   while (std::getline(csv_file, line) && image_pairs.size() < num_images) {
       if (first_line) {
           first_line = false;
           continue;
       }

       std::istringstream line_stream(line);
       std::string timestamp;
       std::getline(line_stream, timestamp, ',');

       //> stack timestamps of images
       Img_time_stamps.push_back( std::stod(timestamp) );
      
       std::string left_img_path = left_path + timestamp + ".png";
       std::string right_img_path = right_path + timestamp + ".png";
      
       cv::Mat left_img = cv::imread(left_img_path, cv::IMREAD_GRAYSCALE);
       cv::Mat right_img = cv::imread(right_img_path, cv::IMREAD_GRAYSCALE);
      
       if (left_img.empty() || right_img.empty()) {
           std::cerr << "ERROR: Could not load the images: " << left_img_path << " or " << right_img_path << "!" << std::endl;
           continue;
       }
      
       image_pairs.emplace_back(left_img, right_img);
   }
  
   csv_file.close();
   return image_pairs;
}

std::vector<std::pair<cv::Mat, cv::Mat>> Dataset::LoadETH3DImages(const std::string &stereo_pairs_path, int num_images) {
    std::vector<std::pair<cv::Mat, cv::Mat>> image_pairs;

    std::vector<std::string> stereo_folders;
    for (const auto &entry : std::filesystem::directory_iterator(stereo_pairs_path)) {
        if (entry.is_directory()) {
            stereo_folders.push_back(entry.path().string());
        }
    }

    std::sort(stereo_folders.begin(), stereo_folders.end());

    for (int i = 0; i < std::min(num_images, static_cast<int>(stereo_folders.size())); ++i) {
        std::string folder_path = stereo_folders[i];

        std::string left_image_path = folder_path + "/im0.png";
        std::string right_image_path = folder_path + "/im1.png";

        cv::Mat left_image = cv::imread(left_image_path, cv::IMREAD_GRAYSCALE);
        cv::Mat right_image = cv::imread(right_image_path, cv::IMREAD_GRAYSCALE);

        if (!left_image.empty() && !right_image.empty()) {
            image_pairs.emplace_back(left_image, right_image);
        } else {
            std::cerr << "ERROR: Could not load images from folder: " << folder_path << std::endl;
        }
    }

    return image_pairs;
}

void Dataset::Load_GT_Poses( std::string GT_Poses_File_Path ) {
   std::ifstream gt_pose_file(GT_Poses_File_Path);
   if (!gt_pose_file.is_open()) {
       LOG_FILE_ERROR(GT_Poses_File_Path);
       exit(1);
   }

   std::string line;
   bool b_first_line = true;
   if (dataset_type == "EuRoC") {
       Eigen::Matrix4d Transf_frame2body;
       Eigen::Matrix4d inv_Transf_frame2body;
       Transf_frame2body.setIdentity();
       Transf_frame2body.block<3,3>(0,0) = rot_frame2body_left;
       Transf_frame2body.block<3,1>(0,3) = transl_frame2body_left;
       inv_Transf_frame2body = Transf_frame2body.inverse();

       Eigen::Matrix4d Transf_Poses;
       Eigen::Matrix4d inv_Transf_Poses;
       Transf_Poses.setIdentity();

       Eigen::Matrix4d frame2world;

       while (std::getline(gt_pose_file, line)) {
           if (b_first_line) {
               b_first_line = false;
               continue;
           }

           std::stringstream ss(line);
           std::string gt_val;
           std::vector<double> csv_row_val;

           while (std::getline(ss, gt_val, ',')) {
               try {
                   csv_row_val.push_back(std::stod(gt_val));
               } catch (const std::invalid_argument& e) {
                   std::cerr << "Invalid argument: " << e.what() << " for value (" << gt_val << ") from the file " << GT_Poses_File_Path << std::endl;
               } catch (const std::out_of_range& e) {
                    std::cerr << "Out of range exception: " << e.what() << " for value: " << gt_val << std::endl;
               }
           }

           GT_time_stamps.push_back(csv_row_val[0]);
           Eigen::Vector3d transl_val( csv_row_val[1], csv_row_val[2], csv_row_val[3] );
           Eigen::Quaterniond quat_val( csv_row_val[4], csv_row_val[5], csv_row_val[6], csv_row_val[7] );
           Eigen::Matrix3d rot_from_quat = quat_val.toRotationMatrix();

           Transf_Poses.block<3,3>(0,0) = rot_from_quat;
           Transf_Poses.block<3,1>(0,3) = transl_val;
           inv_Transf_Poses = Transf_Poses.inverse();

           frame2world = (inv_Transf_frame2body*inv_Transf_Poses).inverse();

           unaligned_GT_Rot.push_back(frame2world.block<3,3>(0,0));
           unaligned_GT_Transl.push_back(frame2world.block<3,1>(0,3));
       }
   }
   else {
       LOG_ERROR("Dataset type not supported!");
   }
}

void Dataset::Align_Images_and_GT_Poses() {
   std::vector<double> time_stamp_diff_val;
   std::vector<unsigned> time_stamp_diff_indx;
   for (double img_time_stamp : Img_time_stamps) {
       time_stamp_diff_val.clear();
       for ( double gt_time_stamp : GT_time_stamps) {
           time_stamp_diff_val.push_back(std::abs(img_time_stamp - gt_time_stamp));
       }
       auto min_diff = std::min_element(std::begin(time_stamp_diff_val), std::end(time_stamp_diff_val));
       int min_index;
       if (min_diff != time_stamp_diff_val.end()) {
           min_index = std::distance(std::begin(time_stamp_diff_val), min_diff);
       } else {
           LOG_ERROR("Empty vector for time stamp difference vector");
       }

       aligned_GT_Rot.push_back(unaligned_GT_Rot[min_index]);
       aligned_GT_Transl.push_back(unaligned_GT_Transl[min_index]);
   }

}

void Dataset::UndistortEdges(const cv::Mat& dist_edges, cv::Mat& undist_edges, std::vector<cv::Point2f>& edge_locations, const std::vector<double>& intr,
   const std::vector<double>& dist_coeffs) {
   cv::Mat calibration_matrix = (cv::Mat_<double>(3, 3) <<
                                 intr[0], 0, intr[2],
                                 0, intr[1], intr[3],
                                 0, 0, 1);

   cv::Mat dist_coeffs_matrix(dist_coeffs);

   std::vector<cv::Point2f> edge_points;
   cv::findNonZero(dist_edges, edge_points);

   std::vector<cv::Point2f> undist_edge_points;
   cv::undistortPoints(edge_points, undist_edge_points, calibration_matrix, dist_coeffs_matrix);

   undist_edges = cv::Mat::zeros(dist_edges.size(), CV_8UC1);

   edge_locations.clear();
   for (const auto& point : undist_edge_points) {
       int x = static_cast<int>(point.x * intr[0] + intr[2]);
       int y = static_cast<int>(point.y * intr[1] + intr[3]);

       if (x >= 0 && x < undist_edges.cols && y >= 0 && y < undist_edges.rows) {
           undist_edges.at<uchar>(y, x) = 255;
           edge_locations.emplace_back(x, y);
       }
   }
}

void Dataset::DisplayOverlay(const std::string& extract_undist_path, const std::string& undistort_extract_path) {
   cv::Mat extract_undist_img = cv::imread(extract_undist_path, cv::IMREAD_GRAYSCALE);
   cv::Mat undist_extract_img = cv::imread(undistort_extract_path, cv::IMREAD_GRAYSCALE);

   cv::Mat overlay;
   cv::cvtColor(extract_undist_img, overlay, cv::COLOR_GRAY2BGR);

   for (int y = 0; y < extract_undist_img.rows; y++) {
       for (int x = 0; x < extract_undist_img.cols; x++) {
           uchar extract_undistort = extract_undist_img.at<uchar>(y, x);
           uchar undistort_extract = undist_extract_img.at<uchar>(y, x);

           if (extract_undistort > 0 && undistort_extract > 0) {
               overlay.at<cv::Vec3b>(y, x) = cv::Vec3b(255, 0, 255);
           }
           else if (extract_undistort > 0) {
               overlay.at<cv::Vec3b>(y, x) = cv::Vec3b(255, 0, 0);
           }
           else if (undistort_extract > 0) {
               overlay.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 255);
           }
       }
   }

   cv::imshow("Edge Map Overlay - Blue (EU), Red (UE), Pink (Both)", overlay);
   cv::imwrite("Edge Map Overlay - Blue (EU), Red (UE), Pink (Both).png", overlay);
   cv::waitKey(0);
}


Eigen::Matrix3d Dataset::ConvertToEigenMatrix(const std::vector<std::vector<double>>& matrix) {
   Eigen::Matrix3d eigen_matrix;
   for (int i = 0; i < 3; i++) {
       for (int j = 0; j < 3; j++) {
           eigen_matrix(i, j) = matrix[i][j];
       }
   }
   return eigen_matrix;
}

// void Dataset::PrintDatasetInfo() {
//     //Stereo Intrinsic Parameters
//     std::cout << "Left Camera Resolution: " << left_res[0] << "x" << left_res[1] << std::endl;
//     std::cout << "\nRight Camera Resolution: " << right_res[0] << "x" << right_res[1] << std::endl;


//     std::cout << "\nLeft Camera Intrinsics: ";
//     for (const auto& value : left_intr) std::cout << value << " ";
//     std::cout << std::endl;


//     std::cout << "\nRight Camera Intrinsics: ";
//     for (const auto& value : right_intr) std::cout << value << " ";
//     std::cout << std::endl;


//     // Stereo Extrinsic Parameters (Left to Right)
//     std::cout << "\nStereo Extrinsic Parameters (Left to Right): \n";


//     std::cout << "\nRotation Matrix: \n";
//     for (const auto& row : rot_mat_21) {
//         for (const auto& value : row) {
//             std::cout << value << " ";
//         }
//         std::cout << std::endl;
//     }


//     std::cout << "\nTranslation Vector: \n";
//     for (const auto& value : trans_vec_21) std::cout << value << " ";
//     std::cout << std::endl;


//     std::cout << "\nFundamental Matrix: \n";
//     for (const auto& row : fund_mat_21) {
//         for (const auto& value : row) {
//             std::cout << value << " ";
//         }
//         std::cout << std::endl;
//     }


//     // Stereo Extrinsic Parameters (Right to Left)
//     std::cout << "\nStereo Extrinsic Parameters (Right to Left): \n";


//     std::cout << "\nRotation Matrix: \n";
//     for (const auto& row : rot_mat_12) {
//         for (const auto& value : row) {
//             std::cout << value << " ";
//         }
//         std::cout << std::endl;
//     }


//     std::cout << "\nTranslation Vector: \n";
//     for (const auto& value : trans_vec_12) std::cout << value << " ";
//     std::cout << std::endl;


//     std::cout << "\nFundamental Matrix: \n";
//     for (const auto& row : fund_mat_12) {
//         for (const auto& value : row) {
//             std::cout << value << " ";
//         }
//         std::cout << std::endl;
//     }
//     std::cout << "\n" << std::endl;
// }

#endif
