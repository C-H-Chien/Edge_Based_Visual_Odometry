
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>

#include "../include/utility.h"
#include "../include/Frame.h"
#include "../include/definitions.h"

template< typename T >
void write_cvMat_to_File( std::string file_Path, cv::Mat Input_Mat ) {
    std::ofstream Test_File_Write_Stream;
    Test_File_Write_Stream.open(file_Path);
    if ( !Test_File_Write_Stream.is_open() ) LOG_FILE_ERROR(file_Path);
    for (int i = 0; i < Input_Mat.rows; i++) {
        for (int j = 0; j < Input_Mat.cols; j++) {
            Test_File_Write_Stream << Input_Mat.at<T>(i,j) << "\t";
        }
        Test_File_Write_Stream << "\n";
    }
    Test_File_Write_Stream.close();
}

int main(int argc, char **argv) {

    Utility utility_tool;

    //> [TEST] Gaussian derivative kernel
    cv::Mat Gx_2d, Gy_2d;
    Gx_2d = cv::Mat::ones(GAUSSIAN_KERNEL_WINDOW_LENGTH, GAUSSIAN_KERNEL_WINDOW_LENGTH, CV_64F);
    Gy_2d = cv::Mat::ones(GAUSSIAN_KERNEL_WINDOW_LENGTH, GAUSSIAN_KERNEL_WINDOW_LENGTH, CV_64F);
    utility_tool.get_dG_2D(Gx_2d, Gy_2d, 4*DEPTH_GRAD_GAUSSIAN_SIGMA, DEPTH_GRAD_GAUSSIAN_SIGMA); 
    std::cout << "Size of Gx_2d and Gy_2d: (" << Gx_2d.rows << ", " << Gx_2d.cols << ")" << std::endl;
    std::string write_Gx_2d_File_Name = REPO_PATH + OUTPUT_WRITE_PATH + std::string("Test_Gx_2d.txt");
    write_cvMat_to_File<double>(write_Gx_2d_File_Name, Gx_2d);

    //> [TEST] Depth map
    cv::Mat depth_Map;
    std::string Current_Depth_Path = std::string("/home/chchien/datasets/TUM-RGBD/rgbd_dataset_freiburg1_desk/depth/1305031453.374112.png");
    depth_Map = cv::imread(Current_Depth_Path, cv::IMREAD_ANYDEPTH);
    if (depth_Map.data == nullptr) std::cerr << "ERROR: Cannot find depth map " << Current_Depth_Path << std::endl;
    //> Scale down by the factor of 5000 (according to the TUM-RGBD dataset)
    depth_Map.convertTo(depth_Map, CV_64F);
    depth_Map /= 5000.0;
    std::string write_Depth_Map_File_Name = REPO_PATH + OUTPUT_WRITE_PATH + std::string("Test_Depth_Map.txt");
    write_cvMat_to_File<double>(write_Depth_Map_File_Name, depth_Map);

    //> [TEST] Gradient depths
    cv::Mat grad_Depth_xi_ = cv::Mat::ones(depth_Map.rows, depth_Map.cols, CV_64F);
    cv::Mat grad_Depth_eta_ = cv::Mat::ones(depth_Map.rows, depth_Map.cols, CV_64F);
    cv::filter2D( depth_Map, grad_Depth_xi_,  depth_Map.depth(), Gx_2d );
    cv::filter2D( depth_Map, grad_Depth_eta_, depth_Map.depth(), Gy_2d );
    grad_Depth_xi_  *= (-1);
    grad_Depth_eta_ *= (-1);
    std::string write_Gradient_Depth_Map_xi_File_Name = REPO_PATH + OUTPUT_WRITE_PATH + std::string("Test_Gradient_Depth_xi.txt");
    write_cvMat_to_File<double>(write_Gradient_Depth_Map_xi_File_Name, grad_Depth_xi_);
    std::string write_Gradient_Depth_Map_eta_File_Name = REPO_PATH + OUTPUT_WRITE_PATH + std::string("Test_Gradient_Depth_eta.txt");
    write_cvMat_to_File<double>(write_Gradient_Depth_Map_eta_File_Name, grad_Depth_eta_);


    //> [TEST] uniform random number generation
    // LOG_TEST("Uniform random number generator");
    // int rnd_sample = utility_tool.Uniform_Random_Number_Generator(0, 300);
    // std::cout << "random sample = " << rnd_sample << std::endl;
    
    return 0;
}