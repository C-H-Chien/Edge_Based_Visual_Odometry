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

#include "Dataset.h"
#include "definitions.h"

// =======================================================================================================
// class Dataset: Fetch data from dataset specified in the configuration file
//
// ChangeLogs
//    Chien  23-01-17    Initially created.
//
//> (c) LEMS, Brown University
//> Chiang-Heng Chien (chiang-heng_chien@brown.edu)
// =======================================================================================================

Dataset::Dataset(YAML::Node config_map)
    : config_file(config_map) 
{
    Calib = Eigen::Matrix3d::Identity();
    Inverse_Calib = Eigen::Matrix3d::Identity();

    //> Parse data from the yaml file
	Dataset_Path = config_file["dataset_dir"].as<std::string>();
	Sequence_Name = config_file["sequence_name"].as<std::string>();
    Dataset_Type = config_file["dataset_type"].as<std::string>();
    
	Calib(0,0) = config_file["camera.fx"].as<double>();
	Calib(1,1) = config_file["camera.fy"].as<double>();
	Calib(0,2) = config_file["camera.cx"].as<double>();
	Calib(1,2) = config_file["camera.cy"].as<double>();

    Inverse_Calib(0,0) = 1.0 / Calib(0,0);
    Inverse_Calib(1,1) = 1.0 / Calib(1,1);
    Inverse_Calib(0,2) = -Calib(0,2) / Calib(0,0);
    Inverse_Calib(1,2) = -Calib(1,2) / Calib(1,1);

    Total_Num_Of_Imgs = 0;
    Current_Frame_Index = 0;
    has_Depth = false;

    //> Used in GCC filter
    // (1) Gaussian filter in x and y direction
    Gx_2d = cv::Mat::ones(GAUSSIAN_KERNEL_WINDOW_LENGTH, GAUSSIAN_KERNEL_WINDOW_LENGTH, CV_64F);
    Gy_2d = cv::Mat::ones(GAUSSIAN_KERNEL_WINDOW_LENGTH, GAUSSIAN_KERNEL_WINDOW_LENGTH, CV_64F);
    utility_tool->get_dG_2D(Gx_2d, Gy_2d, 4*DEPTH_GRAD_GAUSSIAN_SIGMA, DEPTH_GRAD_GAUSSIAN_SIGMA); 
    //> (2) Small patch associated to point \gamma_0
    Small_Patch_Radius_Map = cv::Mat::ones(2*GCC_PATCH_HALF_SIZE+1, 2*GCC_PATCH_HALF_SIZE+1, CV_64F);
}

bool Dataset::Init_Fetch_Data() {

    //> Assume that the tum format contains both images and depths
    if (Dataset_Type == "tum") {
        const std::string Associate_Path = Dataset_Path + Sequence_Name + ASSOCIATION_FILE_NAME;
        stream_Associate_File.open(Associate_Path, std::ios_base::in);
        if (!stream_Associate_File) { 
            std::cerr << "ERROR: Dataset association file does not exist!" << std::endl; 
            return false; 
        }

        std::string img_time_stamp, img_file_name, depth_time_stamp, depth_file_name;
        std::string image_path, depth_path;
        while (stream_Associate_File >> img_time_stamp >> img_file_name >> depth_time_stamp >> depth_file_name) {
            
            image_path = Dataset_Path + Sequence_Name + img_file_name;
            depth_path = Dataset_Path + Sequence_Name + depth_file_name;
            Img_Path_List.push_back(image_path);
            Depth_Path_List.push_back(depth_path);
            Img_Time_Stamps.push_back(img_time_stamp);

            Total_Num_Of_Imgs++;
        }
        has_Depth = true;
    }
    else if (Dataset_Type == "kitti") {
        //> TODO
    }

    return true;
}

Frame::Ptr Dataset::get_Next_Frame() {
    
    //> Read the image
    std::string Current_Image_Path = Img_Path_List[Current_Frame_Index];
    cv::Mat gray_Image = cv::imread(Current_Image_Path, cv::IMREAD_GRAYSCALE);
    if (gray_Image.data == nullptr) {
        std::cerr << "ERROR: Cannot find image at index " << Current_Frame_Index << std::endl;
        std::cerr << "Path: " << Current_Image_Path << std::endl;
        return nullptr;
    }

    //> Read the depth map
    cv::Mat depth_Map;
    if (has_Depth) {
        std::string Current_Depth_Path = Depth_Path_List[Current_Frame_Index];
        depth_Map = cv::imread(Current_Depth_Path, cv::IMREAD_ANYDEPTH);
        if (depth_Map.data == nullptr) {
            std::cerr << "ERROR: Cannot find depth map at index " << Current_Frame_Index << std::endl;
            std::cerr << "Path: " << Current_Depth_Path << std::endl;
            return nullptr;
        }
        //> Scale down by the factor of 5000 (according to the TUM-RGBD dataset)
        depth_Map.convertTo(depth_Map, CV_64F);
        depth_Map /= 5000.0;
    }

    //> Create a new frame from the Frame class
    auto new_frame = Frame::Create_Frame();
    new_frame->Image = gray_Image;
    if (has_Depth) new_frame->Depth = depth_Map;
    new_frame->K = Calib;
    new_frame->inv_K = Inverse_Calib;
    new_frame->ID = Current_Frame_Index;

    //> Get depth gradients for the GDC filter
    grad_Depth_eta_ = cv::Mat::ones(depth_Map.rows, depth_Map.cols, CV_64F);
    grad_Depth_xi_  = cv::Mat::ones(depth_Map.rows, depth_Map.cols, CV_64F);
    cv::filter2D( depth_Map, grad_Depth_xi_,  depth_Map.depth(), Gx_2d );
    cv::filter2D( depth_Map, grad_Depth_eta_, depth_Map.depth(), Gy_2d );

    //> Not sure why the results are opposite in sign. Multiply by -1 to rectify.
    grad_Depth_xi_  *= (-1);
    grad_Depth_eta_ *= (-1);

    new_frame->grad_Depth_eta = grad_Depth_eta_;
    new_frame->grad_Depth_xi  = grad_Depth_xi_;

    Current_Frame_Index++;

    return new_frame;
}

#endif