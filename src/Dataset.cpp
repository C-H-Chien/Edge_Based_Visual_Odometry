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
            YAML::Node left_cam = config_file["left_camera"];
            YAML::Node right_cam = config_file["right_camera"];
            YAML::Node stereo = config_file["stereo"];

            // Parsing left camera parameters
            left_res = left_cam["resolution"].as<std::vector<int>>();
            left_rate = left_cam["rate_hz"].as<int>();
            left_model = left_cam["camera_model"].as<std::string>();
            left_intr = left_cam["intrinsics"].as<std::vector<double>>();
            left_dist_model = left_cam["distortion_model"].as<std::string>();
            left_dist_coeffs = left_cam["distortion_coefficients"].as<std::vector<double>>();

            // Parsing right camera parameters
            right_res = right_cam["resolution"].as<std::vector<int>>();
            right_rate = right_cam["rate_hz"].as<int>();
            right_model = right_cam["camera_model"].as<std::string>();
            right_intr = right_cam["intrinsics"].as<std::vector<double>>();
            right_dist_model = right_cam["distortion_model"].as<std::string>();
            right_dist_coeffs = right_cam["distortion_coefficients"].as<std::vector<double>>();

            // Parsing stereo extrinsic parameters
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
    }

    Total_Num_Of_Imgs = 0;

    // Calib = Eigen::Matrix3d::Identity();
    // Inverse_Calib = Eigen::Matrix3d::Identity();

	// Calib(0,0) = config_file["camera.fx"].as<double>();
	// Calib(1,1) = config_file["camera.fy"].as<double>();
	// Calib(0,2) = config_file["camera.cx"].as<double>();
	// Calib(1,2) = config_file["camera.cy"].as<double>();

    // Inverse_Calib(0,0) = 1.0 / Calib(0,0);
    // Inverse_Calib(1,1) = 1.0 / Calib(1,1);
    // Inverse_Calib(0,2) = -Calib(0,2) / Calib(0,0);
    // Inverse_Calib(1,2) = -Calib(1,2) / Calib(1,1);

    // Current_Frame_Index = 0;
    // has_Depth = false;

    // if (compute_grad_depth) {
    //     Gx_2d = cv::Mat::ones(GAUSSIAN_KERNEL_WINDOW_LENGTH, GAUSSIAN_KERNEL_WINDOW_LENGTH, CV_64F);
    //     Gy_2d = cv::Mat::ones(GAUSSIAN_KERNEL_WINDOW_LENGTH, GAUSSIAN_KERNEL_WINDOW_LENGTH, CV_64F);
    //     utility_tool->get_dG_2D(Gx_2d, Gy_2d, 4*DEPTH_GRAD_GAUSSIAN_SIGMA, DEPTH_GRAD_GAUSSIAN_SIGMA); 
    //     Small_Patch_Radius_Map = cv::Mat::ones(2*GCC_PATCH_HALF_SIZE+1, 2*GCC_PATCH_HALF_SIZE+1, CV_64F);
    // }
}

void Dataset::PerformEdgeBasedVO() {
    std::string left_path = dataset_path + "/" + sequence_name + "/mav0/cam0/data/";
    std::string right_path = dataset_path + "/" + sequence_name + "/mav0/cam1/data/";
    std::string csv_path = dataset_path + "/" + sequence_name + "/mav0/cam0/data.csv";

    std::vector<std::pair<cv::Mat, cv::Mat>> image_pairs = LoadImages(csv_path, left_path, right_path, 5);

    for (const auto& pair : image_pairs) {
        const cv::Mat& left_img = pair.first;
        const cv::Mat& right_img = pair.second; {
        cv::Mat left_calib = (cv::Mat_<double>(3, 3) << left_intr[0], 0, left_intr[2], 0, left_intr[1], left_intr[3], 0, 0, 1);
        cv::Mat right_calib = (cv::Mat_<double>(3, 3) << right_intr[0], 0, right_intr[2], 0, right_intr[1], right_intr[3], 0, 0, 1);
        cv::Mat left_dist_coeff_mat(left_dist_coeffs);
        cv::Mat right_dist_coeff_mat(right_dist_coeffs);
        
        //////////////UNDISTORT, THEN EXTRACT////////////////////

        cv::Mat left_undistorted, right_undistorted;
        cv::undistort(left_img, left_undistorted, left_calib, left_dist_coeff_mat);
        cv::undistort(right_img, right_undistorted, right_calib, right_dist_coeff_mat);
        
        // cv::Mat left_map, right_map;

        // cv::Canny(left_undistorted, left_map, 50, 150);
        // cv::Canny(right_undistorted, right_map, 50, 150);
        
        // std::vector<cv::Point2f> left_edge_coords;
        // cv::findNonZero(left_map, left_edge_coords);

        // DisplayMatches(left_map, right_map, left_edge_coords);

        //////////////EXTRACT, THEN UNDISTORT////////////////////

        // cv::Mat left_map, right_map;
        // cv::Canny(left_img, left_map, 50, 150);
        // cv::Canny(right_img, right_map, 50, 150);

        // // Undistort edges
        // cv::Mat left_undist_edges, right_undist_edges;
        // std::vector<cv::Point2f> left_edge_coords, right_edge_coords;

        // UndistortEdges(left_map, left_undist_edges, left_edge_coords, left_intr, left_dist_coeffs);
        // UndistortEdges(right_map, right_undist_edges, right_edge_coords, right_intr, right_dist_coeffs);

        //////////////THIRD ORDER DETECTION////////////////////

        //> CH: stack all the undistorted images
        undistorted_left_img.push_back(left_undistorted);
        undistorted_right_img.push_back(right_undistorted);
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

        std::vector<cv::Point2f> left_edge_coords;
        cv::findNonZero(left_edge_map, left_edge_coords);
        DisplayMatches(left_edge_map, right_edge_map, left_edge_coords);

        }
    }
}

void Dataset::DisplayMatches(const cv::Mat& left_map, const cv::Mat& right_map, std::vector<cv::Point2f> left_edge_coords) {
    cv::Mat left_visualization, right_visualization;
    cv::cvtColor(left_map, left_visualization, cv::COLOR_GRAY2BGR);
    cv::cvtColor(right_map, right_visualization, cv::COLOR_GRAY2BGR);

    std::vector<cv::Point2f> selected_left_edges = PickRandomEdges(7, left_edge_coords, 5, left_res[0], left_res[1]);

    for (const auto& point : selected_left_edges) {
        cv::circle(left_visualization, point, 5, cv::Scalar(0, 0, 255), cv::FILLED);
    }

    std::vector<cv::Mat> left_patches;
    ExtractPatches(7, left_map, selected_left_edges, left_patches);

    Eigen::Matrix3d fundamental_matrix_21 = ConvertToEigenMatrix(fund_mat_21);
    Eigen::Matrix3d fundamental_matrix_12 = ConvertToEigenMatrix(fund_mat_12);
    std::vector<Eigen::Vector3d> epipolar_lines_right = CalculateEpipolarLine(fundamental_matrix_21, selected_left_edges);
    
    CalculateMatches(selected_left_edges, left_patches, epipolar_lines_right, left_map, right_map, fundamental_matrix_12, right_visualization);

    cv::Mat merged_visualization;
    cv::hconcat(left_visualization, right_visualization, merged_visualization);
    cv::imshow("Edge Matching Using SSD, Lowe's Ratio Test, & Bidirectional Consistency", merged_visualization);
    cv::waitKey(0);
}

void Dataset::CalculateMatches(const std::vector<cv::Point2f>& selected_left_edges, const std::vector<cv::Mat>& left_patches, const std::vector<Eigen::Vector3d>& epipolar_lines_right,
    const cv::Mat& left_map, const cv::Mat& right_map, const Eigen::Matrix3d& fundamental_matrix_12, cv::Mat& right_visualization) {

    for (size_t i = 0; i < selected_left_edges.size(); i++) {
        const auto& left_edge = selected_left_edges[i];
        const auto& left_patch = left_patches[i];
        const auto& epipolar_line = epipolar_lines_right[i];

        std::vector<cv::Point2f> right_candidate_edges = ExtractEpipolarEdges(7, epipolar_line, right_map);

        if (right_candidate_edges.size() > 1) {
            cv::line(right_visualization, right_candidate_edges.front(), right_candidate_edges.back(), cv::Scalar(255, 200, 100), 1);
        }

        std::vector<cv::Mat> right_patches;
        ExtractPatches(7, right_map, right_candidate_edges, right_patches);
        
        if (!left_patch.empty() && !right_patches.empty()) {
            int best_right_match_idx = CalculateSSDPatch(left_patch, right_patches);
            
            if (best_right_match_idx != -1) {
                cv::Point2f best_right_match = right_candidate_edges[best_right_match_idx];

                std::vector<Eigen::Vector3d> right_to_left_epipolar = CalculateEpipolarLine(fundamental_matrix_12, {best_right_match});
                Eigen::Vector3d epipolar_line_left = right_to_left_epipolar[0];

                std::vector<cv::Point2f> left_candidate_edges = ExtractEpipolarEdges(7, epipolar_line_left, left_map);
                std::vector<cv::Mat> left_candidate_patches;
                ExtractPatches(7, left_map, left_candidate_edges, left_candidate_patches);
                
                if (!left_candidate_patches.empty()) {
                    int best_left_match_idx = CalculateSSDPatch(right_patches[best_right_match_idx], left_candidate_patches);
                    
                    if (best_left_match_idx != -1) {
                        cv::Point2f best_left_match = left_candidate_edges[best_left_match_idx];

                        if (best_left_match == left_edge) {
                            cv::circle(right_visualization, best_right_match, 5, cv::Scalar(0, 0, 255), cv::FILLED);
                        }
                    }
                }
            }
        }
    }
}

int Dataset::CalculateSSDPatch(const cv::Mat& left_patch, const std::vector<cv::Mat>& right_patches) {
    if (right_patches.empty()) return -1;

    int best_match_idx = -1;
    int second_best_match_idx = -1;
    double min_ssd = std::numeric_limits<double>::max();
    double second_min_ssd = std::numeric_limits<double>::max();

    for (size_t i = 0; i < right_patches.size(); i++) {
        const cv::Mat& right_patch = right_patches[i];

        if (left_patch.size() != right_patch.size()) continue;

        cv::Mat diff;
        cv::absdiff(left_patch, right_patch, diff);
        cv::Mat squared_diff;
        cv::multiply(diff, diff, squared_diff);

        double ssd = cv::sum(squared_diff)[0];

        if (ssd < min_ssd) {
            second_min_ssd = min_ssd;
            second_best_match_idx = best_match_idx;

            min_ssd = ssd;
            best_match_idx = static_cast<int>(i);
        } else if (ssd < second_min_ssd) {
            second_min_ssd = ssd;
            second_best_match_idx = static_cast<int>(i);
        }
    }

    if (best_match_idx != -1 && second_best_match_idx != -1) {
        double ratio = second_min_ssd / min_ssd;
        if (ratio > 1.6) {
            return best_match_idx;
        } else {
            return -1;  
        }
    }

    return best_match_idx; 
}

void Dataset::ExtractPatches(int patch_size, const cv::Mat& binary_map, const std::vector<cv::Point2f>& selected_edges, std::vector<cv::Mat>& patches) {
    int half_patch = patch_size / 2;
    patches.clear();

    for (const auto& edge : selected_edges) {
        int x = static_cast<int>(edge.x);
        int y = static_cast<int>(edge.y);

        if (x - half_patch >= 0 && x + half_patch < binary_map.cols &&
            y - half_patch >= 0 && y + half_patch < binary_map.rows) {

            cv::Rect patch_rect(x - half_patch, y - half_patch, patch_size, patch_size);
            cv::Mat patch = binary_map(patch_rect).clone();
            patches.push_back(patch);
        } 
        else {
            std::cerr << "WARNING: Skipped patch at (" << x << ", " << y << ") due to boundary constraints!" << std::endl;
        }
    }
}

std::vector<cv::Point2f> Dataset::ExtractEpipolarEdges(int patch_size, const Eigen::Vector3d& epipolar_line, const cv::Mat& binary_map) {
   std::vector<cv::Point2f> edges;

   int width = binary_map.cols;
   int height = binary_map.rows;

   int half_patch = patch_size / 2;
   for (int x = half_patch; x < width - half_patch; x++) {
       double y = (-epipolar_line(2) - epipolar_line(0) * x) / epipolar_line(1);

       int y_int = static_cast<int>(std::round(y));
       if (y_int >= half_patch && y_int < height - half_patch) {
           if (binary_map.at<uchar>(y_int, x) == 255) {
               edges.emplace_back(x, y_int);
           }
       }
   }

   return edges;
}

std::vector<Eigen::Vector3d> Dataset::CalculateEpipolarLine(const Eigen::Matrix3d& fund_mat, const std::vector<cv::Point2f>& edges) {
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

std::vector<cv::Point2f> Dataset::PickRandomEdges(int patch_size, const std::vector<cv::Point2f>& edges, size_t num_points, int img_width, int img_height) {
   std::vector<cv::Point2f> valid_edges;
   int half_patch = patch_size / 2;

   for (const auto& edge : edges) {
       if (edge.x >= half_patch && edge.x < img_width - half_patch &&
           edge.y >= half_patch && edge.y < img_height - half_patch) {
           valid_edges.push_back(edge);
       }
   }

   num_points = std::min(num_points, valid_edges.size());

   std::vector<cv::Point2f> selected_points;
   std::unordered_set<int> used_indices;
   std::random_device rd;
   std::mt19937 gen(rd());
   std::uniform_int_distribution<int> dis(0, valid_edges.size() - 1);

   while (selected_points.size() < num_points) {
       int idx = dis(gen);
       if (used_indices.find(idx) == used_indices.end()) {
           selected_points.push_back(valid_edges[idx]);
           used_indices.insert(idx);
       }
   }

   return selected_points;
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

std::vector<std::pair<cv::Mat, cv::Mat>> Dataset::LoadImages(const std::string& csv_path, const std::string& left_path, const std::string& right_path, 
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

// bool Dataset::Init_Fetch_Data() {
//     if (dataset_type == "tum") {
//         const std::string Associate_Path = dataset_path + sequence_name + ASSOCIATION_FILE_NAME;
//         stream_Associate_File.open(Associate_Path, std::ios_base::in);
//         if (!stream_Associate_File) { 
//             std::cerr << "ERROR: Dataset association file does not exist!" << std::endl; 
//             return false; 
//         }

//         std::string img_time_stamp, img_file_name, depth_time_stamp, depth_file_name;
//         std::string image_path, depth_path;
//         while (stream_Associate_File >> img_time_stamp >> img_file_name >> depth_time_stamp >> depth_file_name) {
            
//             image_path = dataset_path + sequence_name + img_file_name;
//             depth_path = dataset_path + sequence_name + depth_file_name;
//             Img_Path_List.push_back(image_path);
//             Depth_Path_List.push_back(depth_path);
//             Img_Time_Stamps.push_back(img_time_stamp);

//             Total_Num_Of_Imgs++;
//         }
//         has_Depth = true;
//     }
//     else if (dataset_type == "kitti") {
//         //TODO
//     }
//     return true;
// }

// Frame::Ptr Dataset::get_Next_Frame() {
    
//     std::string Current_Image_Path = Img_Path_List[Current_Frame_Index];
//     std::cout << "Image path: " << Current_Image_Path << std::endl;
//     cv::Mat gray_Image = cv::imread(Current_Image_Path, cv::IMREAD_GRAYSCALE);
//     if (gray_Image.data == nullptr) {
//         std::cerr << "ERROR: Cannot find image at index " << Current_Frame_Index << std::endl;
//         std::cerr << "Path: " << Current_Image_Path << std::endl;
//         return nullptr;
//     }

//     cv::Mat depth_Map;
//     if (has_Depth) {
//         std::string Current_Depth_Path = Depth_Path_List[Current_Frame_Index];
//         std::cout << "Depth path: " << Current_Depth_Path << std::endl;
//         depth_Map = cv::imread(Current_Depth_Path, cv::IMREAD_ANYDEPTH);
//         if (depth_Map.data == nullptr) {
//             std::cerr << "ERROR: Cannot find depth map at index " << Current_Frame_Index << std::endl;
//             std::cerr << "Path: " << Current_Depth_Path << std::endl;
//             return nullptr;
//         }
//         depth_Map.convertTo(depth_Map, CV_64F);
//         depth_Map /= 5000.0;
//     }

//     auto new_frame = Frame::Create_Frame();
//     new_frame->Image = gray_Image;
//     if (has_Depth) new_frame->Depth = depth_Map;
//     new_frame->K = Calib;
//     new_frame->inv_K = Inverse_Calib;
//     new_frame->ID = Current_Frame_Index;

//     if (compute_grad_depth) {

//         struct timeval tStart_gradient_depth;
//         struct timeval tEnd_gradient_dpeth;
//         unsigned long time_gradient_depth;

//         gettimeofday(&tStart_gradient_depth, NULL);
//         grad_Depth_eta_ = cv::Mat::ones(depth_Map.rows, depth_Map.cols, CV_64F);
//         grad_Depth_xi_  = cv::Mat::ones(depth_Map.rows, depth_Map.cols, CV_64F);
//         cv::filter2D( depth_Map, grad_Depth_xi_,  depth_Map.depth(), Gx_2d );
//         cv::filter2D( depth_Map, grad_Depth_eta_, depth_Map.depth(), Gy_2d );

//         gettimeofday(&tEnd_gradient_dpeth, NULL);
//         time_gradient_depth = ((tEnd_gradient_dpeth.tv_sec * 1000000) + tEnd_gradient_dpeth.tv_usec) - ((tStart_gradient_depth.tv_sec * 1000000) + tStart_gradient_depth.tv_usec);
//         printf("Time spent on computing gradient depths: %Lf (ms)\n", (long double)time_gradient_depth/1000.0);

//         grad_Depth_xi_  *= (-1);
//         grad_Depth_eta_ *= (-1);

//         new_frame->grad_Depth_eta = grad_Depth_eta_;
//         new_frame->grad_Depth_xi  = grad_Depth_xi_;
//         new_frame->need_depth_grad = true;
//     }

//     Current_Frame_Index++;

//     return new_frame;
// }

#endif