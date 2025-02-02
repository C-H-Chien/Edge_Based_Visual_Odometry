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

#include "Dataset.h"
#include "definitions.h"

// =======================================================================================================
// Class Dataset: Fetch data from dataset specified in the configuration file
//
// ChangeLogs
//    Lopez 25-01-26     Modified for euroc dataset support.
//    Chien  23-01-17    Initially created.
//
//> (c) LEMS, Brown University
//> Chiang-Heng Chien (chiang-heng_chien@brown.edu)
// =======================================================================================================

Dataset::Dataset(YAML::Node config_map, bool use_GCC_filter) : config_file(config_map), compute_grad_depth(use_GCC_filter)
{

	dataset_path = config_file["dataset_dir"].as<std::string>();
	sequence_name = config_file["sequence_name"].as<std::string>();
    dataset_type = config_file["dataset_type"].as<std::string>();

    if (dataset_type == "EuRoC") {
        try {
            YAML::Node left_cam = config_file["left_camera"];
            YAML::Node right_cam = config_file["right_camera"];
            YAML::Node stereo = config_file["stereo"];

            // Parsing left_cam parameters
            left_res = left_cam["resolution"].as<std::vector<int>>();
            left_rate = left_cam["rate_hz"].as<int>();
            left_model = left_cam["camera_model"].as<std::string>();
            left_intr = left_cam["intrinsics"].as<std::vector<double>>();
            left_dist_model = left_cam["distortion_model"].as<std::string>();
            left_dist_coeffs = left_cam["distortion_coefficients"].as<std::vector<double>>();

            // Parsing right_cam parameters
            right_res = right_cam["resolution"].as<std::vector<int>>();
            right_rate= right_cam["rate_hz"].as<int>();
            right_model = right_cam["camera_model"].as<std::string>();
            right_intr = right_cam["intrinsics"].as<std::vector<double>>();
            right_dist_model = right_cam["distortion_model"].as<std::string>();
            right_dist_coeffs = right_cam["distortion_coefficients"].as<std::vector<double>>();

            // Parsing stereo extrinsic parameters
            for (const auto& row : stereo["rotation_matrix"]) {
                rotation_matrix.push_back(row.as<std::vector<double>>());
            }

            translation_vector = stereo["translation_vector"].as<std::vector<double>>();\

            for (const auto& row : stereo["fundamental_matrix"]) {
                fundamental_matrix.push_back(row.as<std::vector<double>>());
            }

        } catch (const YAML::Exception &e) {
            std::cerr << "Error parsing YAML file: " << e.what() << std::endl;
        }
    }
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

    // Total_Num_Of_Imgs = 0;
    // Current_Frame_Index = 0;
    // has_Depth = false;

    if (compute_grad_depth) {
        Gx_2d = cv::Mat::ones(GAUSSIAN_KERNEL_WINDOW_LENGTH, GAUSSIAN_KERNEL_WINDOW_LENGTH, CV_64F);
        Gy_2d = cv::Mat::ones(GAUSSIAN_KERNEL_WINDOW_LENGTH, GAUSSIAN_KERNEL_WINDOW_LENGTH, CV_64F);
        utility_tool->get_dG_2D(Gx_2d, Gy_2d, 4*DEPTH_GRAD_GAUSSIAN_SIGMA, DEPTH_GRAD_GAUSSIAN_SIGMA); 
        Small_Patch_Radius_Map = cv::Mat::ones(2*GCC_PATCH_HALF_SIZE+1, 2*GCC_PATCH_HALF_SIZE+1, CV_64F);
    }
}


// void Dataset::PrintDatasetInfo() {
//     std::cout << "left_cam Resolution: " << left_res[0] << "x" << left_res[1] << std::endl;
//     std::cout << "\nleft_cam Intrinsics: ";
//     for (const auto& value : left_intr) std::cout << value << " ";
//     std::cout << std::endl;

//     std::cout << "right_cam Intrinsics: ";
//     for (const auto& value : right_intr) std::cout << value << " ";
//     std::cout << std::endl;

//     std::cout << "\nStereo Rotation Matrix: " << std::endl;
//     for (const auto& row : rotation_matrix) {
//         for (const auto& value : row) {
//             std::cout << value << " ";
//         }
//         std::cout << std::endl;
//     }

//     std::cout << "\nTranslation Vector: ";
//     for (const auto& value : translation_vector) std::cout << value << " ";
//     std::cout << std::endl;

//     std::cout << "\nFundamental Matrix: " << std::endl;
//     for (const auto& row : fundamental_matrix) {
//         for (const auto& value : row) {
//             std::cout << value << " ";
//         }
//         std::cout << std::endl;
//     }
// }


void Dataset::ExtractEdges(int num_images) {
    std::string left_path = dataset_path + "/" + sequence_name + "/mav0/cam0/data/";
    std::string right_path = dataset_path + "/" + sequence_name + "/mav0/cam1/data/";
    std::string csv_path = dataset_path + "/" + sequence_name + "/mav0/cam0/data.csv";

    //Check if able to open CSV file
    std::ifstream csv_file(csv_path);
    if (!csv_file.is_open()) {
        std::cerr << "Error opening CSV file at directory: " << csv_path << std::endl;
        return;
    }

    std::vector<std::string> image_filenames;
    std::string line;
    bool first_line = true;

    // Read the CSV file
    while (std::getline(csv_file, line) && image_filenames.size() < num_images) {
        if (first_line){
            first_line = false;
            continue;
        }
        std::istringstream line_stream(line);
        std::string timestamp;
        std::getline(line_stream, timestamp, ',');
        image_filenames.push_back(timestamp + ".png");
    }
    csv_file.close();

    // Loop through selected images
    for (const auto& filename : image_filenames) {
        std::string left_img_path = left_path + filename;
        std::string right_img_path = right_path + filename;

        // Load images
        cv::Mat left_img = cv::imread(left_img_path, cv::IMREAD_GRAYSCALE);
        cv::Mat right_img = cv::imread(right_img_path, cv::IMREAD_GRAYSCALE);

        if (left_img.empty() || right_img.empty()) {
            std::cerr << "Error loading image: " << filename << std::endl;
            continue;
        }

        // Use Canny edge detection
        cv::Mat left_edges, right_edges;
        cv::Canny(left_img, left_edges, 50, 150);
        cv::Canny(right_img, right_edges, 50, 150);

        // Undistort edges
        cv::Mat left_undist_edges, right_undist_edges;
        std::vector<cv::Point2f> left_edge_coords, right_edge_coords;

        UndistortEdges(left_edges, left_undist_edges, left_edge_coords, left_intr, left_dist_coeffs);
        UndistortEdges(right_edges, right_undist_edges, right_edge_coords, right_intr, right_dist_coeffs);


        //Visualize random edges
        cv::Mat left_visualization;
        cv::cvtColor(left_undist_edges, left_visualization, cv::COLOR_GRAY2BGR);

        cv::Mat right_visualization;
        cv::cvtColor(right_undist_edges, right_visualization, cv::COLOR_GRAY2BGR);

        std::vector<cv::Point2f> random_edges = SelectRandomEdges(left_edge_coords, 10);

        for (const auto& pt : random_edges) {
            cv::circle(left_visualization, pt, 3, cv::Scalar(255, 200, 100), 1);
        }
        
        //Calculate epipolar line
        Eigen::Matrix3d fund_mat;
        for (int i = 0; i < 3; i++){
            for (int j = 0; j < 3; j++){
                fund_mat(i, j) = fundamental_matrix[i][j];
            }
        }

        std::vector<Eigen::Vector3d> computed_lines = computeEpipolarLine(fund_mat, random_edges);




        //Concatenate left and right camera undistorted
        cv::Mat both_concat;
        cv::hconcat(left_visualization, right_visualization, both_concat);
        cv::imshow("Left Camera Undistorted vs Right Camera Undistorted", both_concat);
        cv::waitKey(0);


        // // Concatenate left camera distorted w/ undistorted
        // cv::Mat left_concat;
        // cv::hconcat(left_edges, left_undist_edges, left_concat);

        // // Concatenate right camera distorted w/ undistorted
        // cv::Mat right_concat;
        // cv::hconcat(right_edges, right_undist_edges, right_concat);

        // // Display results
        // // cv::imshow("Left Camera (Distorted vs Undistorted)", left_concat);
        // // cv::imshow("Right Camera (Distorted vs Undistorted)", right_concat);
    }
}


void Dataset::UndistortEdges(const cv::Mat& dist_edges, cv::Mat& undist_edges, 
                             std::vector<cv::Point2f>& edge_locations,
                             const std::vector<double>& intr, 
                             const std::vector<double>& dist_coeffs) {
    // Create calibration matrix
    cv::Mat calibration_matrix = (cv::Mat_<double>(3, 3) << 
                                  intr[0], 0, intr[2], 
                                  0, intr[1], intr[3], 
                                  0, 0, 1);

    // Convert distortion coefficients to cv::Mat
    cv::Mat dist_coeffs_matrix(dist_coeffs);

    // Extract edge points
    std::vector<cv::Point2f> edge_points;
    cv::findNonZero(dist_edges, edge_points);

    // Undistort edge points
    std::vector<cv::Point2f> undist_edge_points;
    cv::undistortPoints(edge_points, undist_edge_points, calibration_matrix, dist_coeffs_matrix);

    // Create an empty image to store undistorted edges
    undist_edges = cv::Mat::zeros(dist_edges.size(), CV_8UC1);

    // Map undistorted points back to image
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


std::vector<cv::Point2f> Dataset::SelectRandomEdges(const std::vector<cv::Point2f>& edge_points, size_t num_points) {
    std::vector<cv::Point2f> selected_points;

    // Ensure available points are not exceeded
    if (edge_points.empty()) return selected_points;
    num_points = std::min(num_points, edge_points.size());

    // Random number generator setup
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0, edge_points.size() - 1);

    std::unordered_set<int> used_indices;

    while (selected_points.size() < num_points) {
        int idx = dis(gen);
        if (used_indices.find(idx) == used_indices.end()) {
            selected_points.push_back(edge_points[idx]);
            used_indices.insert(idx);
        }
    }

    return selected_points;
}


std::vector<Eigen::Vector3d> Dataset::computeEpipolarLine(const Eigen::Matrix3d& fund_mat, const std::vector<cv::Point2f>& edge_points) {
    std::vector<Eigen::Vector3d> epipolar_lines;

    for (const auto& point : edge_points) {
        // Convert to homogeneous coordinates
        Eigen::Vector3d homo_point(point.x, point.y, 1.0);

        // Compute epipolar line
        Eigen::Vector3d epipolar_line = fund_mat * homo_point;

        // Store line
        epipolar_lines.push_back(epipolar_line);

        // Print epipolar equation
        std::cout << "Epipolar Line Equation for Point (" << point.x << ", " << point.y << "): "
                    << epipolar_line(0) << "x + " 
                    << epipolar_line(1) << "y + " 
                    << epipolar_line(2) << " = 0" << std::endl;
    }

    return epipolar_lines;
}

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