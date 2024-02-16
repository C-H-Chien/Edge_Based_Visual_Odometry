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

// =======================================================================================================
// class Dataset: Fetch data from dataset specified in the configuration file
//
// ChangeLogs
//    Chien  23-01-17    Initially created.
//
//> (c) LEMS, Brown University
//> Chiang-Heng Chien (chiang-heng_chien@brown.edu)
// =======================================================================================================

class Dataset {
    
public:
    //> Allow this class to be accessed as a pointer
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Dataset> Ptr;
    Dataset(YAML::Node);

    //> Read data from dataset, e.g. image list, depth list, etc.
    bool Init_Fetch_Data();

    //> Get the next frame
    Frame::Ptr get_Next_Frame();

    unsigned Total_Num_Of_Imgs;
    int Current_Frame_Index;
    double fx, fy, cx, cy;

private:
    YAML::Node config_file;
    std::string Dataset_Type;
    std::string Dataset_Path;
    std::string Sequence_Name;
    Eigen::Matrix3d Calib;
    Eigen::Matrix3d Inverse_Calib;
    bool has_Depth;

    std::fstream stream_Associate_File;

    std::vector<std::string> Img_Path_List;
    std::vector<std::string> Depth_Path_List;
    std::vector<std::string> Img_Time_Stamps;

    cv::Mat grad_Depth_xi_;
    cv::Mat grad_Depth_eta_;
    cv::Mat Gx_2d, Gy_2d;

    Utility::Ptr utility_tool = nullptr;
};

#endif