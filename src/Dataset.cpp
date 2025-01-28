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

	Dataset_Path = config_file["dataset_dir"].as<std::string>();
	Sequence_Name = config_file["sequence_name"].as<std::string>();
    Dataset_Type = config_file["dataset_type"].as<std::string>();

    if (Dataset_Type == "EuRoC") {
        try {
            YAML::Node cam0 = config_file["cam0"];
            YAML::Node cam1 = config_file["cam1"];
            YAML::Node stereo = config_file["stereo"];

            // Parsing cam0 parameters
            cam0_resolution = cam0["resolution"].as<std::vector<int>>();
            cam0_rate_hz = cam0["rate_hz"].as<int>();
            cam0_model = cam0["camera_model"].as<std::string>();
            cam0_intrinsics = cam0["intrinsics"].as<std::vector<double>>();
            cam0_dist_model = cam0["distortion_model"].as<std::string>();
            cam0_dist_coeffs = cam0["distortion_coefficients"].as<std::vector<double>>();

            // Parsing cam1 parameters
            cam1_resolution = cam1["resolution"].as<std::vector<int>>();
            cam1_rate_hz = cam1["rate_hz"].as<int>();
            cam1_model = cam1["camera_model"].as<std::string>();
            cam1_intrinsics = cam1["intrinsics"].as<std::vector<double>>();
            cam1_dist_model = cam1["distortion_model"].as<std::string>();
            cam1_dist_coeffs = cam1["distortion_coefficients"].as<std::vector<double>>();

            // Parsing stereo extrinsic parameters
            for (const auto& row : stereo["rotation_matrix"]) {
                rotation_matrix.push_back(row.as<std::vector<double>>());
            }

            translation_vector = stereo["translation_vector"].as<std::vector<double>>();

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
void Dataset::PrintDatasetInfo() {
        // Show parsed data
    std::cout << "Cam0 Resolution: " << cam0_resolution[0] << "x" << cam0_resolution[1] << std::endl;
    std::cout << "Cam0 Intrinsics: ";
    for (const auto& value : cam0_intrinsics) std::cout << value << " ";
    std::cout << std::endl;

    std::cout << "Cam1 Intrinsics: ";
    for (const auto& value : cam1_intrinsics) std::cout << value << " ";
    std::cout << std::endl;

    std::cout << "Stereo Rotation Matrix: " << std::endl;
    for (const auto& row : rotation_matrix) {
        for (const auto& value : row) {
            std::cout << value << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Translation Vector: ";
    for (const auto& value : translation_vector) std::cout << value << " ";
    std::cout << std::endl;
}

void Dataset::DetectEdges(int num_images) {
    std::string cam0_path = Dataset_Path + "/" + Sequence_Name + "/mav0/cam0/data/";
    std::string cam1_path = Dataset_Path + "/" + Sequence_Name + "/mav0/cam1/data/";
    std::string csv_file_path = Dataset_Path + "/" + Sequence_Name + "/mav0/cam0/data.csv";

    //Check if able to open CSV file
    std::ifstream csv_file(csv_file_path);
    if (!csv_file.is_open()) {
        std::cerr << "Error: Unable to open CSV file at the following directory: " << csv_file_path << std::endl;
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
        std::string cam0_image_path = cam0_path + filename;
        std::string cam1_image_path = cam1_path + filename;

        // Load images
        cv::Mat cam0_img = cv::imread(cam0_image_path, cv::IMREAD_GRAYSCALE);
        cv::Mat cam1_img = cv::imread(cam1_image_path, cv::IMREAD_GRAYSCALE);

        if (cam0_img.empty() || cam1_img.empty()) {
            std::cerr << "Error: Failed to load image " << filename << " from cam0 or cam1." << std::endl;
            continue;
        }

        // Use Canny edge detection
        cv::Mat cam0_edges, cam1_edges;
        cv::Canny(cam0_img, cam0_edges, 50, 150);
        cv::Canny(cam1_img, cam1_edges, 50, 150);

        // Undistort edges
        cv::Mat cam0_undistorted_edges, cam1_undistorted_edges;
        UndistortEdges(cam0_edges, cam0_undistorted_edges, cam0_intrinsics, cam0_dist_coeffs);
        UndistortEdges(cam1_edges, cam1_undistorted_edges, cam1_intrinsics, cam1_dist_coeffs);

        // Show edges
        cv::imshow("Left Camera Edges - " + filename, cam0_edges);
        cv::imshow("Right Camera Edges - " + filename, cam1_edges);

        // Show undistorted edges
        cv::imshow("Left Undistorted Camera Edges - " + filename, cam0_undistorted_edges);
        cv::imshow("Right Undistorted Camera Edges - " + filename, cam1_undistorted_edges);

        // Optionally save undistorted edges
        // cv::imwrite("left_cam_undistorted_edges_" + filename, cam0_undistorted_edges);
        // cv::imwrite("right_cam_undistorted_edges_" + filename, cam1_undistorted_edges);

        cv::waitKey(0);
    }
}

void Dataset::UndistortEdges(const cv::Mat& distorted_edges, cv::Mat& undistorted_edges, 
                             const std::vector<double>& intrinsics, 
                             const std::vector<double>& distortion_coeffs) {

    cv::Mat calibration_matrix = (cv::Mat_<double>(3, 3) << 
                                  intrinsics[0], 0, intrinsics[2], 
                                  0, intrinsics[1], intrinsics[3], 
                                  0, 0, 1);

    cv::Mat distortion_coeffs_matrix = cv::Mat(distortion_coeffs);
    std::vector<cv::Point2f> edge_points;

    //Extract edge points
    for (int y = 0; y < distorted_edges.rows; y++) {
        for (int x = 0; x < distorted_edges.cols; x++) {
            if (distorted_edges.at<uchar>(y, x) > 0) {
                edge_points.emplace_back(x, y);
            }
        }
    }

    // Undistort the edge points
    std::vector<cv::Point2f> undistorted_edge_points;
    cv::undistortPoints(edge_points, undistorted_edge_points, calibration_matrix, distortion_coeffs_matrix);

    undistorted_edges = cv::Mat::zeros(distorted_edges.size(), CV_8UC1);

    // Map undistorted points back to the new image
    for (const auto& point : undistorted_edge_points) {
        int x = static_cast<int>(point.x * intrinsics[0] + intrinsics[2]); 
        int y = static_cast<int>(point.y * intrinsics[1] + intrinsics[3]);

        if (x >= 0 && x < undistorted_edges.cols && y >= 0 && y < undistorted_edges.rows) {
            undistorted_edges.at<uchar>(y, x) = 255;
        }
    }
}

// bool Dataset::Init_Fetch_Data() {

//     if (Dataset_Type == "tum") {
//         const std::string Associate_Path = Dataset_Path + Sequence_Name + ASSOCIATION_FILE_NAME;
//         stream_Associate_File.open(Associate_Path, std::ios_base::in);
//         if (!stream_Associate_File) { 
//             std::cerr << "ERROR: Dataset association file does not exist!" << std::endl; 
//             return false; 
//         }

//         std::string img_time_stamp, img_file_name, depth_time_stamp, depth_file_name;
//         std::string image_path, depth_path;
//         while (stream_Associate_File >> img_time_stamp >> img_file_name >> depth_time_stamp >> depth_file_name) {
            
//             image_path = Dataset_Path + Sequence_Name + img_file_name;
//             depth_path = Dataset_Path + Sequence_Name + depth_file_name;
//             Img_Path_List.push_back(image_path);
//             Depth_Path_List.push_back(depth_path);
//             Img_Time_Stamps.push_back(img_time_stamp);

//             Total_Num_Of_Imgs++;
//         }
//         has_Depth = true;
//     }
//     else if (Dataset_Type == "kitti") {
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