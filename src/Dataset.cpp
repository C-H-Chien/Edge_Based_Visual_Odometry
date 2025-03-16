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
cv::Mat merged_visualization_global;

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
           if (stereo["focal_length"] && stereo["baseline"]) {
            focal_length = stereo["focal_length"].as<double>();
            baseline = stereo["baseline"].as<double>();
            } else {
                std::cerr << "ERROR: Missing stereo parameters (focal_length, baseline) in YAML file!" << std::endl;
            }
        } catch (const YAML::Exception &e) {
            std::cerr << "ERROR: Could not parse YAML file! " << e.what() << std::endl;
        }
    
   Total_Num_Of_Imgs = 0;
}


void Dataset::PerformEdgeBasedVO() {
    int num_images = 5;
    std::vector<std::pair<cv::Mat, cv::Mat>> image_pairs;
    std::vector<cv::Mat> disparity_maps;
    std::vector<double> max_disparity_values;

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
        disparity_maps = LoadETH3DMaps(stereo_pairs_path, num_images);
        max_disparity_values = LoadMaximumDisparityValues(stereo_pairs_path, num_images);
    }
    
    for (size_t i = 0; i < image_pairs.size(); ++i) {
        const cv::Mat& left_img = image_pairs[i].first;
        const cv::Mat& right_img = image_pairs[i].second;
        const cv::Mat& disparity_map = disparity_maps[i]; 
        max_disparity = max_disparity_values[i];
        {

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

       std::cout << "Processing third-order edges on the right image... " << std::endl;
       TOED->get_Third_Order_Edges( right_undistorted );
       right_third_order_edges_locations = TOED->toed_locations;
       right_third_order_edges_orientation = TOED->toed_orientations;
       std::cout << "Number of third-order edges on the right image: " << TOED->Total_Num_Of_TOED << std::endl;

       Total_Num_Of_Imgs++;

       cv::Mat left_edge_map = cv::Mat::zeros(left_undistorted.size(), CV_8UC1);
       cv::Mat right_edge_map = cv::Mat::zeros(right_undistorted.size(), CV_8UC1);

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

       CalculateGTRightEdge(left_third_order_edges_locations, left_third_order_edges_orientation, disparity_map, left_edge_map, right_edge_map);
       DisplayMatches(left_undistorted, right_undistorted, left_edge_map, right_edge_map, right_third_order_edges_locations, right_third_order_edges_orientation);
       }
   }
}

void Dataset::onMouse(int event, int x, int y, int, void*) {
    if (event == cv::EVENT_MOUSEMOVE) {
        if (merged_visualization_global.empty()) return;

        int left_width = merged_visualization_global.cols / 2; 

        std::string coord_text;
        if (x < left_width) {
            coord_text = "Left Image: (" + std::to_string(x) + ", " + std::to_string(y) + ")";
        } else {
            int right_x = x - left_width;
            coord_text = "Right Image: (" + std::to_string(right_x) + ", " + std::to_string(y) + ")";
        }

        cv::Mat display = merged_visualization_global.clone();
        cv::putText(display, coord_text, cv::Point(x, y), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        cv::imshow("Edge Matching Using NCC & Bidirectional Consistency", display);
    }
}

void Dataset::DisplayMatches(const cv::Mat& left_image, const cv::Mat& right_image, const cv::Mat& left_binary_map, const cv::Mat& right_binary_map, std::vector<cv::Point2d> right_edge_coords, std::vector<double> right_edge_orientations) {
   cv::Mat left_visualization, right_visualization;
   cv::cvtColor(left_binary_map, left_visualization, cv::COLOR_GRAY2BGR);
   cv::cvtColor(right_binary_map, right_visualization, cv::COLOR_GRAY2BGR);

    std::vector<cv::Point2d> left_edge_coords;
    std::vector<cv::Point2d> ground_truth_right_edges;
    std::vector<double> left_edge_orientations;

    for (const auto& data : gt_edge_data) {
        left_edge_coords.push_back(std::get<0>(data)); 
        ground_truth_right_edges.push_back(std::get<1>(data)); 
        left_edge_orientations.push_back(std::get<2>(data)); 
    }

    std::vector<cv::Point2d> selected_left_edges;
    std::vector<double> selected_left_orientations;
    std::vector<cv::Point2d> selected_ground_truth_right_edges;

    // std::tie(selected_left_edges, selected_left_orientations, selected_ground_truth_right_edges) = PickRandomEdges(7, left_edge_coords, ground_truth_right_edges, 
    //     left_edge_orientations, 10, left_res[0], left_res[1]);

    // selected_left_edges = left_edge_coords;
    // selected_left_orientations = left_edge_orientations;
    // selected_ground_truth_right_edges = ground_truth_right_edges;

    selected_left_edges.push_back(cv::Point2d(706.002, 424.508));
    selected_left_orientations.push_back(-0.334778);
    selected_ground_truth_right_edges.push_back(cv::Point2d(692.526, 424.508));

    // std::cout << "Selected Left Edges:\n";
    // for (const auto& point : selected_left_edges) {
    //     std::cout << "X: " << point.x << ", Y: " << point.y << std::endl;
    // }

    // std::cout << "\nSelected Left Orientations:\n";
    // for (size_t i = 0; i < selected_left_orientations.size(); i++) {
    //     std::cout << "Orientation: " << selected_left_orientations[i] << std::endl;
    // }

    // std::cout << "\nSelected Ground Truth Right Edges:\n";
    // for (const auto& point : selected_ground_truth_right_edges) {
    //     std::cout << "X: " << point.x << ", Y: " << point.y << std::endl;
    // }

   for (const auto& point : selected_left_edges) {
       cv::circle(left_visualization, point, 5, cv::Scalar(0, 0, 255), cv::FILLED);
   }

   std::vector<cv::Mat> left_patches;
   ExtractPatches(7, left_image, selected_left_edges, left_patches);

   Eigen::Matrix3d fundamental_matrix_21 = ConvertToEigenMatrix(fund_mat_21);
   Eigen::Matrix3d fundamental_matrix_12 = ConvertToEigenMatrix(fund_mat_12);
   std::vector<Eigen::Vector3d> epipolar_lines_right = CalculateEpipolarLine(fundamental_matrix_21, selected_left_edges);

   CalculateMatches(selected_left_edges, selected_ground_truth_right_edges, selected_left_orientations, left_edge_coords, left_edge_orientations, right_edge_coords, 
    right_edge_orientations, left_patches, epipolar_lines_right, left_image, right_image, fundamental_matrix_12, right_visualization);

   cv::hconcat(left_visualization, right_visualization, merged_visualization_global);
   cv::namedWindow("Edge Matching Using NCC & Bidirectional Consistency");
   cv::setMouseCallback("Edge Matching Using NCC & Bidirectional Consistency", Dataset::onMouse);
   cv::imshow("Edge Matching Using NCC & Bidirectional Consistency", merged_visualization_global);
   cv::waitKey(0);
}

void Dataset::CalculateMatches(const std::vector<cv::Point2d>& selected_left_edges, const std::vector<cv::Point2d>& selected_ground_truth_right_edges,
   const std::vector<double>& selected_left_orientations, const std::vector<cv::Point2d>& left_edge_coords, const std::vector<double>& left_edge_orientations,
   const std::vector<cv::Point2d>& right_edge_coords, const std::vector<double>& right_edge_orientations, const std::vector<cv::Mat>& left_patches,
   const std::vector<Eigen::Vector3d>& epipolar_lines_right, const cv::Mat& left_image, const cv::Mat& right_image, const Eigen::Matrix3d& fundamental_matrix_12,
   cv::Mat& right_visualization) {
    matched_left_edges.clear();
    matched_right_edges.clear();
    matched_left_orientations.clear();
    matched_right_orientations.clear();
    double bidirectional_consistency_tol = 1.0;
    int true_positive = 0;
    int false_negative = 0;
    int true_negative = 0;

    int shift_true_positive = 0;
    int shift_false_negative = 0;
    int shift_true_negative = 0;

    int cluster_true_positive = 0;
    int cluster_true_negative = 0;
    int cluster_false_negative = 0;

    int NCC_true_negative = 0;
    
  for (size_t i = 0; i < selected_left_edges.size(); i++) {
      const auto& left_edge = selected_left_edges[i];
      const auto& ground_truth_right_edge = selected_ground_truth_right_edges[i];
      const auto& left_orientation = selected_left_orientations[i];
      const auto& left_patch = left_patches[i];
      const auto& epipolar_line = epipolar_lines_right[i];
      epipolar_distance_threshold = 0.5;

      double a = epipolar_line(0);
      double b = epipolar_line(1);
      double c = epipolar_line(2);

      std::cout << "Epipolar Line Coefficients:\n";
      std::cout << "a: " << a << ", b: " << b << ", c: " << c << std::endl;

      if (b != 0) {
          cv::Point2d pt1(0, -c / b);
          cv::Point2d pt2(right_visualization.cols, -(c + a * right_visualization.cols) / b);
          cv::line(right_visualization, pt1, pt2, cv::Scalar(255, 200, 100), 1);
      }

       std::pair<std::vector<cv::Point2d>, std::vector<double>> right_candidates_data = ExtractEpipolarEdges(epipolar_line, right_edge_coords, right_edge_orientations, epipolar_distance_threshold);
       std::vector<cv::Point2d> right_candidate_edges = right_candidates_data.first;
       std::vector<double> right_candidate_orientations = right_candidates_data.second;

       std::pair<std::vector<cv::Point2d>, std::vector<double>> test_right_candidates_data = ExtractEpipolarEdges(epipolar_line, right_edge_coords, right_edge_orientations, 3);
       std::vector<cv::Point2d> test_right_candidate_edges = test_right_candidates_data.first;

       /////////////////////////////// REDUCING CANDIDATE POOL W/ MAX DISPARITY/////////////////////////////
       std::vector<cv::Point2d> filtered_right_candidate_edges;
       std::vector<double> filtered_right_candidate_orientations;

       for (size_t j = 0; j < right_candidate_edges.size(); ++j) {
           double disparity = left_edge.x - right_candidate_edges[j].x;

           if (disparity <= max_disparity) {
               filtered_right_candidate_edges.push_back(right_candidate_edges[j]);
               filtered_right_candidate_orientations.push_back(right_candidate_orientations[j]);
           }
       }

       right_candidate_edges = std::move(filtered_right_candidate_edges);
       right_candidate_orientations = std::move(filtered_right_candidate_orientations);

       ///////////////////////////////EPIPOLAR DISTANCE THRESHOLD RECALL//////////////////////////
    //    bool match_found = false;

    //    for (const auto& candidate : right_candidate_edges) {
    //        if (cv::norm(candidate - ground_truth_right_edge) <= 0.5) {
    //            match_found = true;
    //            break;
    //        }
    //    }

    //    if (match_found) {
    //        true_positive++;
    //    } else {
    //            bool gt_right_edge_exists = false;
    //            for (const auto& test_candidate : test_right_candidate_edges) {
    //                if (cv::norm(test_candidate - ground_truth_right_edge) <= 0.5) {
    //                    gt_right_edge_exists = true;
    //                    break;
    //                }
    //            }

    //            if (!gt_right_edge_exists) {
    //                true_negative++;
    //            } else {
    //                false_negative++;
    //            }
    //     }

    //    double recall = 0.0;
    //    if ((true_positive + false_negative) > 0) {
    //        recall = static_cast<double>(true_positive) / (true_positive + false_negative);
    //    }
    //    std::cout <<  "TP: " << true_positive << ", TN: " <<true_negative << ", FN: " << false_negative << ", Recall: " << recall * 100.0 << "%\n";

       ///////////////////////////////EPIPOLAR SHIFT ON CANDIDATE EDGES//////////////////////////
        std::vector<cv::Point2d> valid_shifted_edges;
        std::vector<double> valid_shifted_orientations;
        std::vector<double> epipolar_coefficients = {a, b, c};

        for (size_t i = 0; i < right_candidate_edges.size(); ++i) {
            bool passes_tangency_check = false;

            cv::Point2d corrected_edge = Epipolar_Shift(right_candidate_edges[i], right_candidate_orientations[i], epipolar_coefficients, passes_tangency_check);

            valid_shifted_edges.push_back(corrected_edge);
            valid_shifted_orientations.push_back(right_candidate_orientations[i]);
        }

        ///////////////////////////////SHIFTED EDGES THRESHOLD RECALL//////////////////////////
    //    bool shift_match_found = false;
    //    for (const auto& corrected_edge : valid_shifted_edges) {
    //        if (cv::norm(corrected_edge - ground_truth_right_edge) <= 1.0) {
    //            shift_match_found = true;
    //            break;
    //        }
    //    }

    //    if (shift_match_found) {
    //        shift_true_positive++;
    //    } else {
    //            bool shift_gt_right_edge_exists = false;
    //            for (const auto& test_candidate : test_right_candidate_edges) {
    //                if (cv::norm(test_candidate - ground_truth_right_edge) <= 0.5) {
    //                    shift_gt_right_edge_exists = true;
    //                    break;
    //                }
    //            }

    //            if (!shift_gt_right_edge_exists) {
    //                shift_true_negative++;
    //            } else {
    //                shift_false_negative++;
    //            }
    //     }

    //    double shift_recall = 0.0;
    //    if ((shift_true_positive + shift_false_negative) > 0) {
    //        shift_recall = static_cast<double>(shift_true_positive) / (shift_true_positive + shift_false_negative);
    //    }

    //    std::cout << "Shift TP: " << shift_true_positive << ", Shift TN: " << shift_true_negative << ", Shift FN: " << shift_false_negative << ", Shift Recall: " << shift_recall * 100.0 << "%\n";

    std::vector<std::pair<std::vector<cv::Point2d>, std::vector<double>>> clustered_edges = 
        ClusterEpipolarShiftedEdges(valid_shifted_edges, valid_shifted_orientations);

    std::vector<cv::Point2d> cluster_center_edges;
    std::vector<double> cluster_orientation_edges;

    for (size_t i = 0; i < clustered_edges.size(); ++i) {
        const auto& cluster_edges = clustered_edges[i].first;
        const auto& cluster_orientations = clustered_edges[i].second;

        if (cluster_edges.empty()) continue;

        cv::Point2d sum_point(0.0, 0.0);
        double sum_orientation = 0.0;

        std::cout << "\nCluster #" << (i + 1) << ":\n";
        
        for (size_t j = 0; j < cluster_edges.size(); ++j) {
            std::cout << "  Edge #" << (j + 1) << ": X: " << cluster_edges[j].x 
                    << ", Y: " << cluster_edges[j].y << std::endl;
            sum_point += cluster_edges[j];
        }

        std::cout << "  Orientations:\n";
        for (size_t j = 0; j < cluster_orientations.size(); ++j) {
            std::cout << "    Edge #" << (j + 1) << ": Orientation: " << cluster_orientations[j] << std::endl;
            sum_orientation += cluster_orientations[j];
        }

        cv::Point2d avg_point = sum_point * (1.0 / cluster_edges.size());
        double avg_orientation = sum_orientation / cluster_orientations.size();

        cluster_center_edges.push_back(avg_point);
        cluster_orientation_edges.push_back(avg_orientation);

        std::cout << "  Cluster Center: X: " << avg_point.x 
                << ", Y: " << avg_point.y 
                << ", Average Orientation: " << avg_orientation << std::endl;
    }

    std::cout << "\nFinal Cluster Center Edges:\n";
    for (size_t i = 0; i < cluster_center_edges.size(); ++i) {
        std::cout << "Cluster #" << (i + 1) << ": X: " << cluster_center_edges[i].x 
                << ", Y: " << cluster_center_edges[i].y 
                << ", Orientation: " << cluster_orientation_edges[i] << std::endl;
    }

    ///////////////////////////////CLUSTER CENTER EDGES THRESHOLD RECALL//////////////////////////
    //    bool cluster_match_found = false;

    //    for (const auto& cluster_center : cluster_center_edges) {
    //        if (cv::norm(cluster_center - ground_truth_right_edge) <= 1.0) {
    //            cluster_match_found = true;
    //            break;
    //        }
    //    }

    //    if (cluster_match_found) {
    //        cluster_true_positive++;
    //    } else {
    //            bool cluster_gt_right_edge_exists = false;
    //            for (const auto& test_candidate : test_right_candidate_edges) {
    //                if (cv::norm(test_candidate - ground_truth_right_edge) <= 0.5) {
    //                    cluster_gt_right_edge_exists = true;
    //                    break;
    //                }
    //            }

    //            if (!cluster_gt_right_edge_exists) {
    //                cluster_true_negative++;
    //            } else {
    //                cluster_false_negative++;
    //            }
    //     }

    //    double cluster_recall = 0.0;
    //    if ((cluster_true_positive + cluster_false_negative) > 0) {
    //        cluster_recall = static_cast<double>(cluster_true_positive) / (cluster_true_positive + cluster_false_negative);
    //    }

    //    std::cout << "Cluster TP: " << cluster_true_positive << ", Cluster TN: " << cluster_true_negative << ", Cluster FN: " << cluster_false_negative << ", Cluster Recall: " << cluster_recall * 100.0 << "%\n";

       std::vector<cv::Mat> right_patches;
       ExtractPatches(7, right_image, cluster_center_edges, right_patches);
   }
}

std::vector<std::pair<std::vector<cv::Point2d>, std::vector<double>>> Dataset::ClusterEpipolarShiftedEdges(std::vector<cv::Point2d>& valid_shifted_edges, std::vector<double>& valid_shifted_orientations) {
    double cluster_threshold = 0.5;
    std::vector<std::pair<std::vector<cv::Point2d>, std::vector<double>>> clusters;
    
    if (valid_shifted_edges.empty() || valid_shifted_orientations.empty()) {
        return clusters;
    }

    // Sort edges and orientations together based on X-coordinate
    std::vector<std::pair<cv::Point2d, double>> edge_orientation_pairs;
    for (size_t i = 0; i < valid_shifted_edges.size(); ++i) {
        edge_orientation_pairs.emplace_back(valid_shifted_edges[i], valid_shifted_orientations[i]);
    }

    std::sort(edge_orientation_pairs.begin(), edge_orientation_pairs.end(),
              [](const std::pair<cv::Point2d, double>& a, const std::pair<cv::Point2d, double>& b) {
                  return a.first.x < b.first.x;
              });

    // Extract sorted edges and orientations
    valid_shifted_edges.clear();
    valid_shifted_orientations.clear();
    for (const auto& pair : edge_orientation_pairs) {
        valid_shifted_edges.push_back(pair.first);
        valid_shifted_orientations.push_back(pair.second);
    }

    // Initialize first cluster
    std::vector<cv::Point2d> current_cluster_edges;
    std::vector<double> current_cluster_orientations;
    current_cluster_edges.push_back(valid_shifted_edges[0]);
    current_cluster_orientations.push_back(valid_shifted_orientations[0]);

    // Iterate through sorted edges and cluster them
    for (size_t i = 1; i < valid_shifted_edges.size(); ++i) {
        double distance = cv::norm(valid_shifted_edges[i] - valid_shifted_edges[i - 1]); 
        double orientation_difference = std::abs(valid_shifted_orientations[i] - valid_shifted_orientations[i - 1]);

        if (distance <= cluster_threshold && orientation_difference < 5.0) {
            // Add to current cluster
            current_cluster_edges.push_back(valid_shifted_edges[i]);
            current_cluster_orientations.push_back(valid_shifted_orientations[i]);
        } else {
            // Store completed cluster and start a new one
            clusters.emplace_back(current_cluster_edges, current_cluster_orientations);
            current_cluster_edges.clear();
            current_cluster_orientations.clear();
            current_cluster_edges.push_back(valid_shifted_edges[i]);
            current_cluster_orientations.push_back(valid_shifted_orientations[i]);
        }
    }

    // Store the last cluster
    if (!current_cluster_edges.empty()) {
        clusters.emplace_back(current_cluster_edges, current_cluster_orientations);
    }

    return clusters;
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

    double NCC_threshold = 0.85; 
    double ratio_threshold = 1.1; 

    int best_match_idx = -1;
    int second_best_match_idx = -1;
    double best_score = -1.0;
    double second_best_score = -1.0;

    for (size_t i = 0; i < right_patches.size(); i++) {
        cv::Mat result;
        cv::matchTemplate(right_patches[i], left_patch, result, cv::TM_CCORR_NORMED);
        double score = result.at<float>(0, 0);


        if (score > best_score) {
            second_best_score = best_score;
            second_best_match_idx = best_match_idx;

            best_score = score;
            best_match_idx = static_cast<int>(i);
        } else if (score > second_best_score) {
            second_best_score = score;
            second_best_match_idx = static_cast<int>(i);
        }
    }

    if (best_match_idx != -1 && second_best_match_idx != -1) {
        double ratio = best_score / second_best_score;
        if (ratio < ratio_threshold) {  
            return -1;  
        }
    }

    if (best_score < NCC_threshold) return -1;

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
std::pair<std::vector<cv::Point2d>, std::vector<double>>Dataset::ExtractEpipolarEdges(const Eigen::Vector3d& epipolar_line, const std::vector<cv::Point2d>& edge_locations, const std::vector<double>& edge_orientations, double distance_threshold) {
   std::vector<cv::Point2d> extracted_edges;
   std::vector<double> extracted_orientations;

   if (edge_locations.size() != edge_orientations.size()) {
       throw std::runtime_error("Edge locations and orientations size mismatch.");
   }

    for (size_t i = 0; i < edge_locations.size(); ++i) {
       const auto& edge = edge_locations[i];
       double x = edge.x;
       double y = edge.y;

       double distance = std::abs(epipolar_line(0) * x + epipolar_line(1) * y + epipolar_line(2))
                         / std::sqrt((epipolar_line(0) * epipolar_line(0)) + (epipolar_line(1) * epipolar_line(1)));

       if (distance < distance_threshold) {
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

    //    std::cout << "Epipolar Line Equation for Point (" << point.x << ", " << point.y << "): "
    //              << epipolar_line(0) << "x + "
    //              << epipolar_line(1) << "y + "
    //              << epipolar_line(2) << " = 0" << std::endl;
   }

   return epipolar_lines;
}

//UPDATED
std::tuple<std::vector<cv::Point2d>, std::vector<double>, std::vector<cv::Point2d>> Dataset::PickRandomEdges(int patch_size, const std::vector<cv::Point2d>& edges, const std::vector<cv::Point2d>& ground_truth_right_edges, const std::vector<double>& orientations, size_t num_points, int img_width, int img_height) {
    std::vector<cv::Point2d> valid_edges;
    std::vector<double> valid_orientations;
    std::vector<cv::Point2d> valid_ground_truth_edges;
    int half_patch = patch_size / 2;

    if (edges.size() != orientations.size() || edges.size() != ground_truth_right_edges.size()) {
        throw std::runtime_error("Edge locations, orientations, and ground truth edges size mismatch.");
    }

    for (size_t i = 0; i < edges.size(); ++i) {
        const auto& edge = edges[i];
        if (edge.x >= half_patch && edge.x < img_width - half_patch &&
            edge.y >= half_patch && edge.y < img_height - half_patch) {
            valid_edges.push_back(edge);
            valid_orientations.push_back(orientations[i]);
            valid_ground_truth_edges.push_back(ground_truth_right_edges[i]);
        }
    }

    num_points = std::min(num_points, valid_edges.size());

    std::vector<cv::Point2d> selected_points;
    std::vector<double> selected_orientations;
    std::vector<cv::Point2d> selected_ground_truth_points;
    std::unordered_set<int> used_indices;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0, valid_edges.size() - 1);

    while (selected_points.size() < num_points) {
        int index = dis(gen);
        if (used_indices.find(index) == used_indices.end()) {
            selected_points.push_back(valid_edges[index]);
            selected_orientations.push_back(valid_orientations[index]);
            selected_ground_truth_points.push_back(valid_ground_truth_edges[index]);
            used_indices.insert(index);
        }
    }

    return {selected_points, selected_orientations, selected_ground_truth_points};
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

        // std::cout << "Loading Image Pair #" << (i + 1) << ":\n";
        // std::cout << "  Image 0 Path: " << left_image_path << std::endl;
        // std::cout << "  Image 1 Path: " << right_image_path << std::endl;

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


std::vector<cv::Mat> Dataset::LoadETH3DMaps(const std::string &stereo_pairs_path, int num_maps) {
    std::vector<cv::Mat> disparity_maps;
    std::vector<std::string> stereo_folders;

    for (const auto &entry : std::filesystem::directory_iterator(stereo_pairs_path)) {
        if (entry.is_directory()) {
            stereo_folders.push_back(entry.path().string());
        }
    }

    std::sort(stereo_folders.begin(), stereo_folders.end());

    for (int i = 0; i < std::min(num_maps, static_cast<int>(stereo_folders.size())); ++i) {
        std::string folder_path = stereo_folders[i];

        std::string disparity_map_path = folder_path + "/disparity_map.csv";

        std::ifstream file(disparity_map_path);
        if (!file.is_open()) {
            std::cerr << "ERROR: Could not open disparity map file: " << disparity_map_path << std::endl;
            continue;
        }

        std::vector<std::vector<float>> data;
        std::string line;
        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::vector<float> row;
            std::string value;
            while (std::getline(ss, value, ',')) {
                try {
                    float disparity_value = std::stof(value);

                    if (value == "nan" || value == "NaN") {
                        disparity_value = std::numeric_limits<float>::quiet_NaN();
                    } else if (value == "inf" || value == "Inf") {
                        disparity_value = std::numeric_limits<float>::infinity();
                    } else if (value == "-inf" || value == "-Inf") {
                        disparity_value = -std::numeric_limits<float>::infinity();
                    }

                    row.push_back(disparity_value);
                } catch (const std::exception &e) {
                    std::cerr << "WARNING: Invalid value in file: " << disparity_map_path << " -> " << value << std::endl;
                    row.push_back(std::numeric_limits<float>::quiet_NaN()); 
                }
            }
            if (!row.empty()) {
                data.push_back(row);
            }
        }
        file.close();

        if (!data.empty()) {
            int rows = data.size();
            int cols = data[0].size();
            cv::Mat disparity_map(rows, cols, CV_32F);

            for (int r = 0; r < rows; ++r) {
                for (int c = 0; c < cols; ++c) {
                    disparity_map.at<float>(r, c) = data[r][c];
                }
            }

            disparity_maps.push_back(disparity_map);
        }
    }

    return disparity_maps;
}

std::vector<double> Dataset::LoadMaximumDisparityValues(const std::string& stereo_pairs_path, int num_images) {
    std::vector<double> max_disparities;
    std::string csv_filename = stereo_pairs_path + "/maximum_disparity_values.csv";
    std::ifstream file(csv_filename);
    std::string line;

    if (!file.is_open()) {
        std::cerr << "ERROR: Could not open file " << csv_filename << "!"<<std::endl;
        return max_disparities; 
    }

    int count = 0;
    while (std::getline(file, line) && count < num_images) {
        max_disparities.push_back(std::stod(line));
        count++;
    }

    file.close();
    return max_disparities;
}

double Bilinear_Interpolation(const cv::Mat &meshGrid, cv::Point2d P) {
    cv::Point2d Q12(floor(P.x), floor(P.y));
    cv::Point2d Q22(ceil(P.x), floor(P.y));
    cv::Point2d Q11(floor(P.x), ceil(P.y));
    cv::Point2d Q21(ceil(P.x), ceil(P.y));

    if (Q11.x < 0 || Q11.y < 0 || Q21.x >= meshGrid.cols || Q21.y >= meshGrid.rows ||
        Q12.x < 0 || Q12.y < 0 || Q22.x >= meshGrid.cols || Q22.y >= meshGrid.rows) {
        return std::numeric_limits<double>::quiet_NaN(); 
    }

    double fQ11 = meshGrid.at<float>(Q11.y, Q11.x);
    double fQ21 = meshGrid.at<float>(Q21.y, Q21.x);
    double fQ12 = meshGrid.at<float>(Q12.y, Q12.x);
    double fQ22 = meshGrid.at<float>(Q22.y, Q22.x);

    double f_x_y1 = ((Q21.x - P.x) / (Q21.x - Q11.x)) * fQ11 + ((P.x - Q11.x) / (Q21.x - Q11.x)) * fQ21;
    double f_x_y2 = ((Q21.x - P.x) / (Q21.x - Q11.x)) * fQ12 + ((P.x - Q11.x) / (Q21.x - Q11.x)) * fQ22;
    return ((Q12.y - P.y) / (Q12.y - Q11.y)) * f_x_y1 + ((P.y - Q11.y) / (Q12.y - Q11.y)) * f_x_y2;
}

void Dataset::VisualizeGTRightEdge(const cv::Mat &left_image, const cv::Mat &right_image, const std::vector<std::pair<cv::Point2d, cv::Point2d>> &left_right_edges) {
    cv::Mat left_visualization, right_visualization;
    cv::cvtColor(left_image, left_visualization, cv::COLOR_GRAY2BGR);
    cv::cvtColor(right_image, right_visualization, cv::COLOR_GRAY2BGR);

    std::vector<cv::Scalar> vibrant_colors = {
        cv::Scalar(255, 0, 0),    
        cv::Scalar(0, 255, 0),   
        cv::Scalar(0, 0, 255),   
        cv::Scalar(255, 255, 0), 
        cv::Scalar(255, 0, 255),
        cv::Scalar(0, 255, 255),
        cv::Scalar(255, 165, 0), 
        cv::Scalar(128, 0, 128),
        cv::Scalar(0, 128, 255),
        cv::Scalar(255, 20, 147)
    };

    std::vector<std::pair<cv::Point2d, cv::Point2d>> sampled_pairs;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distr(0, left_right_edges.size() - 1);

    int num_samples = std::min(10, static_cast<int>(left_right_edges.size()));
    for (int i = 0; i < num_samples; ++i) {
        sampled_pairs.push_back(left_right_edges[distr(gen)]);
    }

    for (size_t i = 0; i < sampled_pairs.size(); ++i) {
        const auto &[left_edge, right_edge] = sampled_pairs[i];

        cv::Scalar color = vibrant_colors[i % vibrant_colors.size()];

        cv::circle(left_visualization, left_edge, 5, color, cv::FILLED); 
        cv::circle(right_visualization, right_edge, 5, color, cv::FILLED); 
    }

    cv::Mat merged_visualization;
    cv::hconcat(left_visualization, right_visualization, merged_visualization);

    cv::imshow("Ground Truth Disparity Edges Visualization", merged_visualization);
    cv::waitKey(0);
}

void Dataset::CalculateGTRightEdge(const std::vector<cv::Point2d> &left_third_order_edges_locations, const std::vector<double> &left_third_order_edges_orientation, const cv::Mat &disparity_map, const cv::Mat &left_image, const cv::Mat &right_image) {
    // std::cout << "Size of left_third_order_edges_locations: " << left_third_order_edges_locations.size() << std::endl;
    // std::cout << "Size of left_third_order_edges_orientation: " << left_third_order_edges_orientation.size() << std::endl;
    gt_edge_data.clear();

    for (size_t i = 0; i < left_third_order_edges_locations.size(); i++) {
        const cv::Point2d &left_edge = left_third_order_edges_locations[i];
        double orientation = left_third_order_edges_orientation[i];

        double disparity = Bilinear_Interpolation(disparity_map, left_edge);

        if (std::isnan(disparity) || std::isinf(disparity) || disparity < 0) {
            continue; 
        }

        cv::Point2d right_edge(left_edge.x - disparity, left_edge.y);
        gt_edge_data.emplace_back(left_edge, right_edge, orientation);
    }
}

cv::Point2d Dataset::Epipolar_Shift( cv::Point2d original_edge_location, double edge_orientation, std::vector<double> epipolar_line_coeffs, bool& b_pass_epipolar_tengency_check){
    cv::Point2d corrected_edge;

    assert(epipolar_line_coeffs.size() == 3);
    double EL_coeff_A = epipolar_line_coeffs[0];
    double EL_coeff_B = epipolar_line_coeffs[1];
    double EL_coeff_C = epipolar_line_coeffs[2];

    double a1_line  = -epipolar_line_coeffs[0] / epipolar_line_coeffs[1];
    double b1_line  = -1;
    double c1_line  = -epipolar_line_coeffs[2] / epipolar_line_coeffs[1];

    double a_edgeH2 = tan(edge_orientation);
    double b_edgeH2 = -1;
    double c_edgeH2 = -(a_edgeH2*original_edge_location.x - original_edge_location.y); //−(a⋅x2−y2)

    corrected_edge.x = (b1_line * c_edgeH2 - b_edgeH2 * c1_line) / (a1_line * b_edgeH2 - a_edgeH2 * b1_line);
    corrected_edge.y = (c1_line * a_edgeH2 - c_edgeH2 * a1_line) / (a1_line * b_edgeH2 - a_edgeH2 * b1_line);

    double epipolar_shift_displacement = cv::norm(corrected_edge - original_edge_location);
    double m_epipolar = -a1_line / b1_line;
    double angle_diff_rad = abs(edge_orientation - atan(m_epipolar));
    double angle_diff_deg = angle_diff_rad * (180.0 / M_PI);
    if (angle_diff_deg > 180){
        angle_diff_deg -= 180;
    }

    b_pass_epipolar_tengency_check = (epipolar_shift_displacement < 4 && abs(angle_diff_deg - 0) > 8 && abs(angle_diff_deg - 180) > 8) ? (true) : (false);

    if (!b_pass_epipolar_tengency_check) {
        return original_edge_location; 
    }
    
    return corrected_edge;
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

void Dataset::PrintDatasetInfo() {
    std::cout << "Left Camera Resolution: " << left_res[0] << "x" << left_res[1] << std::endl;
    std::cout << "\nRight Camera Resolution: " << right_res[0] << "x" << right_res[1] << std::endl;

    std::cout << "\nLeft Camera Intrinsics: ";
    for (const auto& value : left_intr) std::cout << value << " ";
    std::cout << std::endl;

    std::cout << "\nRight Camera Intrinsics: ";
    for (const auto& value : right_intr) std::cout << value << " ";
    std::cout << std::endl;

    std::cout << "\nStereo Extrinsic Parameters (Left to Right): \n";

    std::cout << "\nRotation Matrix: \n";
    for (const auto& row : rot_mat_21) {
        for (const auto& value : row) {
            std::cout << value << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "\nTranslation Vector: \n";
    for (const auto& value : trans_vec_21) std::cout << value << " ";
    std::cout << std::endl;

    std::cout << "\nFundamental Matrix: \n";
    for (const auto& row : fund_mat_21) {
        for (const auto& value : row) {
            std::cout << value << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "\nStereo Extrinsic Parameters (Right to Left): \n";

    std::cout << "\nRotation Matrix: \n";
    for (const auto& row : rot_mat_12) {
        for (const auto& value : row) {
            std::cout << value << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "\nTranslation Vector: \n";
    for (const auto& value : trans_vec_12) std::cout << value << " ";
    std::cout << std::endl;

    std::cout << "\nFundamental Matrix: \n";
    for (const auto& row : fund_mat_12) {
        for (const auto& value : row) {
            std::cout << value << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "\nStereo Camera Parameters: \n";
    std::cout << "Focal Length: " << focal_length << " pixels" << std::endl;
    std::cout << "Baseline: " << baseline << " meters" << std::endl;

    std::cout << "\n" << std::endl;
}

#endif
