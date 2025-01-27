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
        // Showing parsed data
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

// void Dataset::CheckImageLoading() {
//     // Build the complete path to first image
//     std::string cam0_image_path = Dataset_Path + "/" + Sequence_Name + "/mav0/cam0/data/1403715273262142976.png"; 
//     std::string cam1_image_path = Dataset_Path + "/" + Sequence_Name + "/mav0/cam1/data/1403715273262142976.png";

//     // Load images using OpenCV
//     cv::Mat cam0_img = cv::imread(cam0_image_path, cv::IMREAD_GRAYSCALE);
//     cv::Mat cam1_img = cv::imread(cam1_image_path, cv::IMREAD_GRAYSCALE);

//     if (cam0_img.empty()) {
//         std::cerr << "Error: Failed to load the first image from cam0 at " << cam0_image_path << std::endl;
//     } else {
//         std::cout << "Successfully loaded the first image from cam0: " << cam0_image_path << std::endl;
//         cv::imshow("Cam0 Image", cam0_img);
//     }

//     if (cam1_img.empty()) {
//         std::cerr << "Error: Failed to load the first image from cam1 at " << cam1_image_path << std::endl;
//     } else {
//         std::cout << "Successfully loaded the first image from cam1: " << cam1_image_path << std::endl;
//         cv::imshow("Cam1 Image", cam1_img);
//     }

//     cv::waitKey(0); 
// }

void Dataset::DetectEdges() {
    // Create paths for first image from cam0 and cam1
    std::string cam0_image_path = Dataset_Path + "/" + Sequence_Name + "/mav0/cam0/data/1403715273262142976.png"; 
    std::string cam1_image_path = Dataset_Path + "/" + Sequence_Name + "/mav0/cam1/data/1403715273262142976.png";

    // Load images
    cv::Mat cam0_img = cv::imread(cam0_image_path, cv::IMREAD_GRAYSCALE);
    cv::Mat cam1_img = cv::imread(cam1_image_path, cv::IMREAD_GRAYSCALE);

    if (cam0_img.empty() || cam1_img.empty()) {
        std::cerr << "Error: Failed to load images from cam0 or cam1." << std::endl;
        return;
    }

    // Use Canny edge detection
    cv::Mat cam0_edges, cam1_edges;
    cv::Canny(cam0_img, cam0_edges, 50, 150); 
    cv::Canny(cam1_img, cam1_edges, 50, 150);

    // Show edges
    cv::imshow("Cam0 Edges", cam0_edges);
    cv::imshow("Cam1 Edges", cam1_edges);
    
    // Save results
    cv::imwrite("cam0_edges.png", cam0_edges);
    cv::imwrite("cam1_edges.png", cam1_edges);

    cv::waitKey(0);
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