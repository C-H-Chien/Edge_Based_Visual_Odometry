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
#include <utility> 
#include <cmath>
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
double ComputeAverage(const std::vector<int>& values) {
    if (values.empty()) return 0.0;

    double sum = 0.0;
    for (int val : values) {
        sum += static_cast<double>(val);
    }

    return sum / values.size();
}

Dataset::Dataset(YAML::Node config_map, bool use_GCC_filter) : config_file(config_map), compute_grad_depth(use_GCC_filter) {
   dataset_path = config_file["dataset_dir"].as<std::string>();
   output_path = config_file["output_dir"].as<std::string>();
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

void Dataset::write_ncc_vals_to_files( int img_index ) {
    std::string file_path = OUTPUT_WRITE_PATH + "ncc_vs_err/img_" + std::to_string(img_index) + ".txt";
    std::ofstream ncc_vs_err_file_out(file_path);
    for (unsigned i = 0; i < ncc_one_vs_err.size(); i++) {
        ncc_vs_err_file_out << ncc_one_vs_err[i].first << "\t" << ncc_one_vs_err[i].second << "\t" \
                            << ncc_two_vs_err[i].first << "\t" << ncc_two_vs_err[i].second << "\n";
    }
    ncc_vs_err_file_out.close();
}

void Dataset::PerformEdgeBasedVO() {
    int num_pairs = 1;
    std::vector<std::pair<cv::Mat, cv::Mat>> image_pairs;
    std::vector<cv::Mat> disparity_maps;
    std::vector<double> max_disparity_values;

    std::vector<double> per_image_avg_before_epi;
    std::vector<double> per_image_avg_after_epi;

    std::vector<double> per_image_avg_before_disp;
    std::vector<double> per_image_avg_after_disp;

    std::vector<double> per_image_avg_before_shift;
    std::vector<double> per_image_avg_after_shift;

    std::vector<double> per_image_avg_before_clust;
    std::vector<double> per_image_avg_after_clust;

    std::vector<double> per_image_avg_before_patch;
    std::vector<double> per_image_avg_after_patch;

    std::vector<double> per_image_avg_before_ncc;
    std::vector<double> per_image_avg_after_ncc;

    std::vector<double> per_image_avg_before_lowe;
    std::vector<double> per_image_avg_after_lowe;

    std::vector<RecallMetrics> all_recall_metrics;

    if (dataset_type == "EuRoC"){
        std::string left_path = dataset_path + "/" + sequence_name + "/mav0/cam0/data/";
        std::string right_path = dataset_path + "/" + sequence_name + "/mav0/cam1/data/";
        std::string image_csv_path = dataset_path + "/" + sequence_name + "/mav0/cam0/data.csv";
        std::string ground_truth_path = dataset_path + "/" + sequence_name + "/mav0/state_groundtruth_estimate0/data.csv";

        image_pairs = LoadEuRoCImages(image_csv_path, left_path, right_path, num_pairs);

        Load_GT_Poses(ground_truth_path);
        Align_Images_and_GT_Poses();
    }
    else if (dataset_type == "ETH3D"){
        std::string stereo_pairs_path = dataset_path + "/" + sequence_name + "/stereo_pairs";
        image_pairs = LoadETH3DImages(stereo_pairs_path, num_pairs);
        disparity_maps = LoadETH3DMaps(stereo_pairs_path, num_pairs);
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    LOG_INFO("Start looping over all image pairs");

    for (size_t i = 0; i < image_pairs.size(); i++) {
    // for (size_t i = 0; i < 3; i++) {

        ncc_one_vs_err.clear();
        ncc_two_vs_err.clear();

        const cv::Mat& left_img = image_pairs[i].first;
        const cv::Mat& right_img = image_pairs[i].second;
        const cv::Mat& disparity_map = disparity_maps[i]; 
        {

        std::cout << "Image #" << i << "\n";
        cv::Mat left_calib = (cv::Mat_<double>(3, 3) << left_intr[0], 0, left_intr[2], 0, left_intr[1], left_intr[3], 0, 0, 1);
        cv::Mat right_calib = (cv::Mat_<double>(3, 3) << right_intr[0], 0, right_intr[2], 0, right_intr[1], right_intr[3], 0, 0, 1);
        cv::Mat left_dist_coeff_mat(left_dist_coeffs);
        cv::Mat right_dist_coeff_mat(right_dist_coeffs);

        cv::Mat left_undistorted, right_undistorted;
        cv::undistort(left_img, left_undistorted, left_calib, left_dist_coeff_mat);
        cv::undistort(right_img, right_undistorted, right_calib, right_dist_coeff_mat);

        if (Total_Num_Of_Imgs == 0) {
            img_height = left_undistorted.rows;
            img_width  = left_undistorted.cols;
            TOED = std::shared_ptr<ThirdOrderEdgeDetectionCPU>(new ThirdOrderEdgeDetectionCPU( img_height, img_width ));
        }

        std::string edge_dir = output_path + "/edges";
        std::filesystem::create_directories(edge_dir);

        std::string left_edge_path = edge_dir + "/left_edges_" + std::to_string(i);
        std::string right_edge_path = edge_dir + "/right_edges_" + std::to_string(i);

        ProcessEdges(left_undistorted, left_edge_path, TOED, left_third_order_edges_locations, left_third_order_edges_orientation);
        std::cout << "Number of edges on the left image: " << left_third_order_edges_locations.size() << std::endl;

        ProcessEdges(right_undistorted, right_edge_path, TOED, right_third_order_edges_locations, right_third_order_edges_orientation);
        std::cout << "Number of edges on the right image: " << right_third_order_edges_locations.size() << std::endl;

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
        RecallMetrics metrics = DisplayMatches(left_undistorted, right_undistorted, left_edge_map, right_edge_map, right_third_order_edges_locations, right_third_order_edges_orientation);
        all_recall_metrics.push_back(metrics);

        double avg_before_epi = ComputeAverage(metrics.epi_input_counts);
        double avg_after_epi = ComputeAverage(metrics.epi_output_counts);

        double avg_before_disp = ComputeAverage(metrics.disp_input_counts);
        double avg_after_disp = ComputeAverage(metrics.disp_output_counts);

        double avg_before_shift = ComputeAverage(metrics.shift_input_counts);
        double avg_after_shift = ComputeAverage(metrics.shift_output_counts);

        double avg_before_clust = ComputeAverage(metrics.clust_input_counts);
        double avg_after_clust = ComputeAverage(metrics.clust_output_counts);

        double avg_before_patch = ComputeAverage(metrics.patch_input_counts);
        double avg_after_patch = ComputeAverage(metrics.patch_output_counts);

        double avg_before_ncc = ComputeAverage(metrics.ncc_input_counts);
        double avg_after_ncc = ComputeAverage(metrics.ncc_output_counts);

        double avg_before_lowe = ComputeAverage(metrics.lowe_input_counts);
        double avg_after_lowe = ComputeAverage(metrics.lowe_output_counts);

        per_image_avg_before_epi.push_back(avg_before_epi);
        per_image_avg_after_epi.push_back(avg_after_epi);

        per_image_avg_before_disp.push_back(avg_before_disp);
        per_image_avg_after_disp.push_back(avg_after_disp);

        per_image_avg_before_shift.push_back(avg_before_shift);
        per_image_avg_after_shift.push_back(avg_after_shift);

        per_image_avg_before_clust.push_back(avg_before_clust);
        per_image_avg_after_clust.push_back(avg_after_clust);

        per_image_avg_before_patch.push_back(avg_before_patch);
        per_image_avg_after_patch.push_back(avg_after_patch);

        per_image_avg_before_ncc.push_back(avg_before_ncc);
        per_image_avg_after_ncc.push_back(avg_after_ncc);

        per_image_avg_before_lowe.push_back(avg_before_lowe);
        per_image_avg_after_lowe.push_back(avg_after_lowe);
        }                                
    }

    double total_epi_recall = 0.0;
    double total_disp_recall = 0.0;
    double total_shift_recall = 0.0;
    double total_cluster_recall = 0.0;
    double total_patch_recall = 0.0;
    double total_ncc_recall = 0.0;
    double total_lowe_recall = 0.0;

    for (const RecallMetrics& m : all_recall_metrics) {
        total_epi_recall += m.epi_distance_recall;
        total_disp_recall += m.max_disparity_recall;
        total_shift_recall += m.epi_shift_recall;
        total_cluster_recall += m.epi_cluster_recall;
        total_patch_recall += m.patch_recall;
        total_ncc_recall += m.ncc_recall;
        total_lowe_recall += m.lowe_recall;
    }

    int total_images = static_cast<int>(all_recall_metrics.size());

    double avg_epi_recall   = (total_images > 0) ? total_epi_recall / total_images : 0.0;
    double avg_disp_recall  = (total_images > 0) ? total_disp_recall / total_images : 0.0;
    double avg_shift_recall = (total_images > 0) ? total_shift_recall / total_images : 0.0;
    double avg_cluster_recall = (total_images > 0) ? total_cluster_recall / total_images : 0.0;
    double avg_patch_recall = (total_images > 0) ? total_patch_recall / total_images : 0.0;
    double avg_ncc_recall = (total_images > 0) ? total_ncc_recall / total_images : 0.0;
    double avg_lowe_recall = (total_images > 0) ? total_lowe_recall / total_images: 0.0;

    std::string edge_stat_dir = output_path + "/edge stats";
    std::filesystem::create_directories(edge_stat_dir);
    std::ofstream recall_csv(edge_stat_dir + "/recall_metrics.csv");
    recall_csv << "ImageIndex,EpiDistanceRecall,MaxDisparityRecall,EpiShiftRecall,EpiClusterRecall,PatchRecall,NCCRecall,LoweRecall\n";

    for (size_t i = 0; i < all_recall_metrics.size(); i++) {
        const auto& m = all_recall_metrics[i];
        recall_csv << i << ","
                << std::fixed << std::setprecision(4) << m.epi_distance_recall * 100 << ","
                << std::fixed << std::setprecision(4) << m.max_disparity_recall * 100 << ","
                << std::fixed << std::setprecision(4) << m.epi_shift_recall * 100 << ","
                << std::fixed << std::setprecision(4) << m.epi_cluster_recall * 100 << ","
                << std::fixed << std::setprecision(4) << m.patch_recall * 100 << ","
                << std::fixed << std::setprecision(4) << m.ncc_recall * 100 << ","
                << std::fixed << std::setprecision(4) << m.lowe_recall * 100 << "\n";
    }

    recall_csv << "Average,"
            << std::fixed << std::setprecision(4) << avg_epi_recall * 100 << ","
            << std::fixed << std::setprecision(4) << avg_disp_recall * 100 << ","
            << std::fixed << std::setprecision(4) << avg_shift_recall * 100 << ","
            << std::fixed << std::setprecision(4) << avg_cluster_recall * 100 << ","
            << std::fixed << std::setprecision(4) << avg_patch_recall * 100 << ","
            << std::fixed << std::setprecision(4) << avg_ncc_recall * 100 << ","
            << std::fixed << std::setprecision(4) << avg_lowe_recall * 100 << "\n";
    
    std::ofstream count_csv(edge_stat_dir + "/count_metrics.csv");
    count_csv 
        << "before_epi_distance,after_epi_distance,average_before_epi_distance,average_after_epi_distance,"
        << "before_max_disp,after_max_disp,average_before_max_disp,average_after_max_disp,"
        << "before_epi_shift,after_epi_shift,average_before_epi_shift,average_after_epi_shift,"
        << "before_epi_cluster,after_epi_cluster,average_before_epi_cluster,average_after_epi_cluster,"
        << "before_patch, after_patch, average_before_patch, average_after_patch,"
        << "before_ncc,after_ncc,average_before_ncc,average_after_ncc,"
        << "before_lowe,after_lowe,average_before_lowe,after_after_lowe\n";

    double total_avg_before_epi = 0.0;
    double total_avg_after_epi = 0.0;

    double total_avg_before_disp = 0.0;
    double total_avg_after_disp = 0.0;

    double total_avg_before_shift = 0.0;
    double total_avg_after_shift = 0.0;

    double total_avg_before_clust = 0.0;
    double total_avg_after_clust = 0.0;

    double total_avg_before_patch = 0.0;
    double total_avg_after_patch = 0.0;

    double total_avg_before_ncc = 0.0;
    double total_avg_after_ncc = 0.0;

    double total_avg_before_lowe = 0.0;
    double total_avg_after_lowe = 0.0;

    size_t num_rows = per_image_avg_before_epi.size();

    for (size_t i = 0; i < num_rows; ++i) {
        total_avg_before_epi += per_image_avg_before_epi[i];
        total_avg_after_epi += per_image_avg_after_epi[i];

        total_avg_before_disp += per_image_avg_before_disp[i];
        total_avg_after_disp += per_image_avg_after_disp[i];

        total_avg_before_shift += per_image_avg_before_shift[i];
        total_avg_after_shift += per_image_avg_after_shift[i];

        total_avg_before_clust += per_image_avg_before_clust[i];
        total_avg_after_clust += per_image_avg_after_clust[i];

        total_avg_before_patch += per_image_avg_before_patch[i];
        total_avg_after_patch += per_image_avg_after_patch[i];

        total_avg_before_ncc += per_image_avg_before_ncc[i];
        total_avg_after_ncc += per_image_avg_after_ncc[i];

        total_avg_before_lowe += per_image_avg_before_lowe[i];
        total_avg_after_lowe += per_image_avg_after_lowe[i];

        count_csv
            << static_cast<int>(std::ceil(per_image_avg_before_epi[i])) << ","
            << static_cast<int>(std::ceil(per_image_avg_after_epi[i])) << ","
            << ","
            << ","
            << static_cast<int>(std::ceil(per_image_avg_before_disp[i])) << ","
            << static_cast<int>(std::ceil(per_image_avg_after_disp[i])) << ","
            << ","
            << ","
            << static_cast<int>(std::ceil(per_image_avg_before_shift[i])) << ","
            << static_cast<int>(std::ceil(per_image_avg_after_shift[i])) << ","
            << ","
            << ","
            << static_cast<int>(std::ceil(per_image_avg_before_clust[i])) << ","
            << static_cast<int>(std::ceil(per_image_avg_after_clust[i])) << ","
            << ","
            << ","
            << static_cast<int>(std::ceil(per_image_avg_before_patch[i])) << ","
            << static_cast<int>(std::ceil(per_image_avg_after_patch[i])) << ","
            << ","
            << ","
            << static_cast<int>(std::ceil(per_image_avg_before_ncc[i])) << ","
            << static_cast<int>(std::ceil(per_image_avg_after_ncc[i])) << ","
            << ","
            << ","
            << static_cast<int>(std::ceil(per_image_avg_before_lowe[i])) << ","
            << static_cast<int>(std::ceil(per_image_avg_after_lowe[i])) << ","
            <<"\n";
    }

    int avg_of_avgs_before_epi = 0;
    int avg_of_avgs_after_epi = 0;

    int avg_of_avgs_before_disp = 0;
    int avg_of_avgs_after_disp = 0;

    int avg_of_avgs_before_shift = 0;
    int avg_of_avgs_after_shift = 0;

    int avg_of_avgs_before_clust = 0;
    int avg_of_avgs_after_clust = 0;

    int avg_of_avgs_before_patch = 0;
    int avg_of_avgs_after_patch = 0;
    
    int avg_of_avgs_before_ncc = 0;
    int avg_of_avgs_after_ncc = 0;

    int avg_of_avgs_before_lowe = 0;
    int avg_of_avgs_after_lowe = 0;

    if (num_rows > 0) {
        avg_of_avgs_before_epi = std::ceil(total_avg_before_epi / num_rows);
        avg_of_avgs_after_epi = std::ceil(total_avg_after_epi / num_rows);

        avg_of_avgs_before_disp = std::ceil(total_avg_before_disp / num_rows);
        avg_of_avgs_after_disp = std::ceil(total_avg_after_disp / num_rows);

        avg_of_avgs_before_shift = std::ceil(total_avg_before_shift / num_rows);
        avg_of_avgs_after_shift = std::ceil(total_avg_after_shift / num_rows);

        avg_of_avgs_before_clust = std::ceil(total_avg_before_clust / num_rows);
        avg_of_avgs_after_clust = std::ceil(total_avg_after_clust / num_rows);

        avg_of_avgs_before_patch = std::ceil(total_avg_before_patch / num_rows);
        avg_of_avgs_after_patch = std::ceil(total_avg_after_patch / num_rows);

        avg_of_avgs_before_ncc = std::ceil(total_avg_before_ncc / num_rows);
        avg_of_avgs_after_ncc = std::ceil(total_avg_after_ncc / num_rows);

        avg_of_avgs_before_lowe = std::ceil(total_avg_before_lowe / num_rows);
        avg_of_avgs_after_lowe = std::ceil(total_avg_after_lowe / num_rows);
    }

    count_csv 
        << ","                                   
        << ","                                                                     
        << avg_of_avgs_before_epi << ","       
        << avg_of_avgs_after_epi << ","   
        << "," 
        << ","     
        << avg_of_avgs_before_disp << ","   
        << avg_of_avgs_after_disp << ","
        << ","      
        << ","                              
        << avg_of_avgs_before_shift << ","              
        << avg_of_avgs_after_shift << ","    
        << ","   
        << ","                                            
        << avg_of_avgs_before_clust << ","            
        << avg_of_avgs_after_clust << ","
        << ","
        << ","
        << avg_of_avgs_before_patch << ","            
        << avg_of_avgs_after_patch << ","
        << ","
        << ","
        << avg_of_avgs_before_ncc << ","            
        << avg_of_avgs_after_ncc << ","
        << ","
        << ","
        << avg_of_avgs_before_lowe << ","            
        << avg_of_avgs_after_lowe << "\n";          
}

RecallMetrics Dataset::DisplayMatches(const cv::Mat& left_image, const cv::Mat& right_image, const cv::Mat& left_binary_map, const cv::Mat& right_binary_map, std::vector<cv::Point2d> right_edge_coords, std::vector<double> right_edge_orientations) {
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

    selected_left_edges = left_edge_coords;
    selected_left_orientations = left_edge_orientations;
    selected_ground_truth_right_edges = ground_truth_right_edges;

    auto [left_orthogonal_one, left_orthogonal_two] = CalculateOrthogonalShifts(selected_left_edges, selected_left_orientations, ORTHOGONAL_SHIFT_MAG);

    std::vector<cv::Point2d> filtered_left_edges;
    std::vector<double> filtered_left_orientations;
    std::vector<cv::Point2d> filtered_ground_truth_right_edges;

    std::vector<cv::Mat> left_patch_set_one;
    std::vector<cv::Mat> left_patch_set_two;

    ExtractPatches(
        PATCH_SIZE,
        left_image,
        selected_left_edges,
        selected_left_orientations,
        &selected_ground_truth_right_edges,
        left_orthogonal_one,
        left_orthogonal_two,
        filtered_left_edges,
        filtered_left_orientations,
        &filtered_ground_truth_right_edges,
        left_patch_set_one,
        left_patch_set_two
    );

    Eigen::Matrix3d fundamental_matrix_21 = ConvertToEigenMatrix(fund_mat_21);
    Eigen::Matrix3d fundamental_matrix_12 = ConvertToEigenMatrix(fund_mat_12);
    std::vector<Eigen::Vector3d> epipolar_lines_right = CalculateEpipolarLine(fundamental_matrix_21, filtered_left_edges);

    RecallMetrics recall_metrics = CalculateMatches(filtered_left_edges, filtered_ground_truth_right_edges, filtered_left_orientations, left_edge_coords, left_edge_orientations, right_edge_coords, 
    right_edge_orientations, left_patch_set_one, left_patch_set_two, epipolar_lines_right, left_image, right_image, fundamental_matrix_12, right_visualization);

    return recall_metrics;
}

RecallMetrics Dataset::CalculateMatches(const std::vector<cv::Point2d>& selected_left_edges, const std::vector<cv::Point2d>& selected_ground_truth_right_edges,
   const std::vector<double>& selected_left_orientations, const std::vector<cv::Point2d>& left_edge_coords, const std::vector<double>& left_edge_orientations,
   const std::vector<cv::Point2d>& right_edge_coords, const std::vector<double>& right_edge_orientations, const std::vector<cv::Mat>& left_patch_set_one, const std::vector<cv::Mat>& left_patch_set_two,
   const std::vector<Eigen::Vector3d>& epipolar_lines_right, const cv::Mat& left_image, const cv::Mat& right_image, const Eigen::Matrix3d& fundamental_matrix_12,
   cv::Mat& right_visualization) {

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

    int epi_true_positive = 0;
    int epi_false_negative = 0;
    int epi_true_negative = 0;

    int disp_true_positive = 0;
    int disp_false_negative = 0;

    int shift_true_positive = 0;
    int shift_false_negative = 0;

    int cluster_true_positive = 0;
    int cluster_false_negative = 0;
    int cluster_true_negative = 0;

    int patch_true_positive = 0;
    int patch_false_negative = 0;

    int ncc_true_positive = 0;
    int ncc_false_negative = 0;

    int lowe_true_positive = 0;
    int lowe_false_negative = 0;

    double per_edge_epi_precision = 0.0;
    double per_edge_disp_precision = 0.0;
    double per_edge_shift_precision = 0.0;
    double per_edge_clust_precision = 0.0;
    double per_edge_patch_precision = 0.0;
    double per_edge_ncc_precision = 0.0;
    double per_edge_lowe_precision = 0.0;

    double selected_max_disparity = 23.0063;

    int epi_edges_evaluated = 0;
    int patch_edges_evaluated = 0;
    int ncc_edges_evaluated = 0;
    int lowe_edges_evaluated = 0;

    int skip = 100;
    for (size_t i = 0; i < selected_left_edges.size(); i += skip) {
        const auto& left_edge = selected_left_edges[i];
        const auto& left_orientation = selected_left_orientations[i];
        const auto& ground_truth_right_edge = selected_ground_truth_right_edges[i];
        const auto& epipolar_line = epipolar_lines_right[i];
        const auto& left_patch_one = left_patch_set_one[i];
        const auto& left_patch_two = left_patch_set_two[i];

        double a = epipolar_line(0);
        double b = epipolar_line(1);
        double c = epipolar_line(2);

        if (std::abs(b) < 1e-6) continue;

        double a1_line = -a / b;
        double b1_line = -1;

        double m_epipolar = -a1_line / b1_line; 
        double angle_diff_rad = abs(left_orientation - atan(m_epipolar));
        double angle_diff_deg = angle_diff_rad * (180.0 / M_PI);
        if (angle_diff_deg > 180) {
            angle_diff_deg -= 180;
        }

        bool left_passes_tangency = (abs(angle_diff_deg - 0) > EPIP_TENGENCY_ORIENT_THRESH && abs(angle_diff_deg - 180) > EPIP_TENGENCY_ORIENT_THRESH) ? (true) : (false);
        if (!left_passes_tangency) {
            continue;
        }

        ///////////////////////////////EPIPOLAR DISTANCE THRESHOLD///////////////////////////////
        std::pair<std::vector<cv::Point2d>, std::vector<double>> right_candidates_data = ExtractEpipolarEdges(epipolar_line, right_edge_coords, right_edge_orientations, 0.5);
        std::vector<cv::Point2d> right_candidate_edges = right_candidates_data.first;
        std::vector<double> right_candidate_orientations = right_candidates_data.second;

        std::pair<std::vector<cv::Point2d>, std::vector<double>> test_right_candidates_data = ExtractEpipolarEdges(epipolar_line, right_edge_coords, right_edge_orientations, 3);
        std::vector<cv::Point2d> test_right_candidate_edges = test_right_candidates_data.first;

        epi_input_counts.push_back(right_edge_coords.size());
        ///////////////////////////////EPIPOLAR DISTANCE THRESHOLD RECALL//////////////////////////
        int epi_precision_numerator = 0;
        bool match_found = false;

        for (const auto& candidate : right_candidate_edges) {
            if (cv::norm(candidate - ground_truth_right_edge) <= 0.5) {
                epi_precision_numerator++;
                match_found = true;
            }
        }

        if (match_found) {
            epi_true_positive++;
        } 
        else {
            bool gt_match_found = false;
            for (const auto& test_candidate : test_right_candidate_edges) {
                if (cv::norm(test_candidate - ground_truth_right_edge) <= 0.5) {
                    gt_match_found = true;
                    break;
                }
            }

            if (!gt_match_found) {
                epi_true_negative++;
                continue;
            } else {
                epi_false_negative++;
            }
        }
        if (!right_candidate_edges.empty()) {
            per_edge_epi_precision += static_cast<double>(epi_precision_numerator) / right_candidate_edges.size();
            epi_edges_evaluated++; 
        }
        ///////////////////////////////MAXIMUM DISPARITY THRESHOLD//////////////////////////
        epi_output_counts.push_back(right_candidate_edges.size());

        std::vector<cv::Point2d> filtered_right_edge_coords;
        std::vector<double> filtered_right_edge_orientations;

        for (size_t j = 0; j < right_candidate_edges.size(); j++) {
            const cv::Point2d& right_edge = right_candidate_edges[j];

            double disparity = left_edge.x - right_edge.x;

            bool within_horizontal = (disparity >= 0) && (disparity <= selected_max_disparity);
            bool within_vertical = std::abs(right_edge.y - left_edge.y) <= selected_max_disparity;

            if (within_horizontal && within_vertical) {
                filtered_right_edge_coords.push_back(right_edge);
                filtered_right_edge_orientations.push_back(right_candidate_orientations[j]);
            }
        }

        disp_input_counts.push_back(right_candidate_edges.size());
        ///////////////////////////////MAXIMUM DISPARITY THRESHOLD RECALL//////////////////////////
        int disp_precision_numerator = 0;
        bool disp_match_found = false;

        for(const auto& filtered_candidate : filtered_right_edge_coords){
            if (cv::norm(filtered_candidate - ground_truth_right_edge) <= 0.5){
                disp_precision_numerator++;
                disp_match_found = true;
            }
        }

        if (disp_match_found){
            disp_true_positive++;
        } 
        else {
            disp_false_negative++;
        }
        if (!filtered_right_edge_coords.empty()) {
            per_edge_disp_precision += static_cast<double>(disp_precision_numerator) / filtered_right_edge_coords.size();
        }
        ///////////////////////////////EPIPOLAR SHIFT THRESHOLD//////////////////////////
        disp_output_counts.push_back(filtered_right_edge_coords.size());

        std::vector<cv::Point2d> shifted_right_edge_coords;
        std::vector<double> shifted_right_edge_orientations;
        std::vector<double> epipolar_coefficients = {a, b, c};

        for (size_t j = 0; j < filtered_right_edge_coords.size(); j++) {
            bool right_passes_tangency = false;

            cv::Point2d shifted_edge = PerformEpipolarShift(filtered_right_edge_coords[j], filtered_right_edge_orientations[j], epipolar_coefficients, right_passes_tangency);
            if (right_passes_tangency){
                shifted_right_edge_coords.push_back(shifted_edge);
                shifted_right_edge_orientations.push_back(filtered_right_edge_orientations[j]);
            }
        }

        shift_input_counts.push_back(filtered_right_edge_coords.size());
        ///////////////////////////////EPIPOLAR SHIFT THRESHOLD RECALL//////////////////////////
        int shift_precision_numerator = 0;
        bool shift_match_found = false;

        for(const auto& shifted_candidate : shifted_right_edge_coords){
            if (cv::norm(shifted_candidate - ground_truth_right_edge) <= 3.0){
                shift_precision_numerator++;
                shift_match_found = true;
            }
        }

        if (shift_match_found){
            shift_true_positive++;
        } 
        else {
            shift_false_negative++;
        }
        if (!shifted_right_edge_coords.empty()) {
            per_edge_shift_precision += static_cast<double>(shift_precision_numerator) / shifted_right_edge_coords.size();
        }
        ///////////////////////////////EPIPOLAR CLUSTER THRESHOLD//////////////////////////
        shift_output_counts.push_back(shifted_right_edge_coords.size());

        std::vector<std::pair<std::vector<cv::Point2d>, std::vector<double>>> clusters = ClusterEpipolarShiftedEdges(shifted_right_edge_coords, shifted_right_edge_orientations);
        std::vector<cv::Point2d> cluster_center_edge_coords;
        std::vector<double> cluster_center_edge_orientations;

        for (size_t j = 0; j < clusters.size(); j++) {
            const auto& cluster_edges = clusters[j].first;
            const auto& cluster_orientations = clusters[j].second;

            if (cluster_edges.empty()) continue;

            cv::Point2d sum_point(0.0, 0.0);
            double sum_orientation = 0.0;

            for (size_t j = 0; j < cluster_edges.size(); ++j) {
                sum_point += cluster_edges[j];
            }

            for (size_t j = 0; j < cluster_orientations.size(); ++j) {
                sum_orientation += cluster_orientations[j];
            }

            cv::Point2d avg_point = sum_point * (1.0 / cluster_edges.size());
            double avg_orientation = sum_orientation * (1.0 / cluster_orientations.size());

            cluster_center_edge_coords.push_back(avg_point);
            cluster_center_edge_orientations.push_back(avg_orientation);
        }

        clust_input_counts.push_back(shifted_right_edge_coords.size());
        ///////////////////////////////EPIPOLAR CLUSTER THRESHOLD RECALL//////////////////////////
        int clust_precision_numerator = 0;
        bool cluster_match_found = false;

        for (const auto& cluster_candidate : cluster_center_edge_coords) {
            if (cv::norm(cluster_candidate - ground_truth_right_edge) <= 3.0) {
                clust_precision_numerator++;
                cluster_match_found = true;
            }
        }

        if (cluster_match_found) {
            cluster_true_positive++;
        }
        else {
            cluster_false_negative++;
        }
        if (!cluster_center_edge_coords.empty()) {
            per_edge_clust_precision += static_cast<double>(clust_precision_numerator) / cluster_center_edge_coords.size();
        }
        ///////////////////////////////EXTRACT PATCHES THRESHOLD////////////////////////////////////////////
        clust_output_counts.push_back(cluster_center_edge_coords.size());

        auto [right_orthogonal_one, right_orthogonal_two] = CalculateOrthogonalShifts(
            cluster_center_edge_coords,
            cluster_center_edge_orientations,
            ORTHOGONAL_SHIFT_MAG
        );

        std::vector<cv::Point2d> patch_right_edge_coords;
        std::vector<double> patch_right_edge_orientations;
        std::vector<cv::Mat> right_patch_set_one;
        std::vector<cv::Mat> right_patch_set_two;

        ExtractPatches(
            PATCH_SIZE,
            right_image,
            cluster_center_edge_coords,
            cluster_center_edge_orientations,
            nullptr,
            right_orthogonal_one,
            right_orthogonal_two,
            patch_right_edge_coords,
            patch_right_edge_orientations,
            nullptr,
            right_patch_set_one,
            right_patch_set_two
        );

        patch_input_counts.push_back(cluster_center_edge_coords.size());
        ///////////////////////////////EXTRACT PATCHES RECALL////////////////////////////////////////////
        int patch_precision_numerator = 0;
        bool patch_match_found = false;

        for (const auto& patch_candidate : patch_right_edge_coords) {
            if (cv::norm(patch_candidate - ground_truth_right_edge) <= 3.0) {
                patch_precision_numerator++;
                patch_match_found = true;
            }
        }

        if (patch_match_found) {
            patch_true_positive++;
        }
        else {
            patch_false_negative++;
        }
        if (!patch_right_edge_coords.empty()) {
            per_edge_patch_precision += static_cast<double>(patch_precision_numerator) / patch_right_edge_coords.size();
            patch_edges_evaluated++;
        }
        ///////////////////////////////NCC THRESHOLD/////////////////////////////////////////////////////
        patch_output_counts.push_back(patch_right_edge_coords.size());
        int ncc_precision_numerator = 0;

        double ncc_threshold = -0.1;
        bool ncc_match_found = false;
        std::vector<PatchMatch> passed_ncc_matches;

        if (!left_patch_one.empty() && !left_patch_two.empty() &&
            !right_patch_set_one.empty() && !right_patch_set_two.empty()){

            for (size_t i = 0; i < patch_right_edge_coords.size(); ++i) {
                double ncc_A = ComputeNCC(left_patch_one, right_patch_set_one[i]);
                double ncc_B = ComputeNCC(left_patch_two, right_patch_set_two[i]);
                double ncc_C = ComputeNCC(left_patch_one, right_patch_set_two[i]);
                double ncc_D = ComputeNCC(left_patch_two, right_patch_set_one[i]);

                double score_one = std::min(ncc_A, ncc_B);
                double score_two = std::min(ncc_C, ncc_D);

                double final_score = std::max(score_one, score_two);

                if(final_score >= ncc_threshold){
                    PatchMatch info;
                    info.coord = patch_right_edge_coords[i];
                    info.orientation = patch_right_edge_orientations[i];
                    info.final_score = final_score;
                    passed_ncc_matches.push_back(info);
                    if (cv::norm(patch_right_edge_coords[i] - ground_truth_right_edge) <= 3.0) {
                        ncc_precision_numerator++;
                        ncc_match_found = true;
                    }
                }
            }
            if (ncc_match_found) {
                ncc_true_positive++;
            } else {
                ncc_false_negative++;
            }
        }

        if (!patch_right_edge_coords.empty()) {
            per_edge_ncc_precision += static_cast<double>(ncc_precision_numerator) / patch_right_edge_coords.size();
            ncc_edges_evaluated++;
        }
        ncc_input_counts.push_back(patch_right_edge_coords.size());
        ncc_output_counts.push_back(passed_ncc_matches.size());
        ///////////////////////////////LOWES RATIO TEST//////////////////////////////////////////////
        lowe_input_counts.push_back(passed_ncc_matches.size());
        int lowe_precision_numerator = 0;

        PatchMatch best_match;
        double best_score = -1;

        if (passed_ncc_matches.size() >= 2){
            PatchMatch second_best_match;
            double second_best_score = -1;

            for(const auto& match : passed_ncc_matches){
                if(match.final_score > best_score){
                    second_best_score = best_score;
                    second_best_match = best_match;

                    best_score = match.final_score;
                    best_match = match;
                }
                else if (match.final_score > second_best_score){
                    second_best_score = match.final_score;
                    second_best_match = match;
                }
            }
            double lowe_ratio = second_best_score / best_score;

            if(lowe_ratio < 0.1){
                if (cv::norm(best_match.coord - ground_truth_right_edge) <= 3.0) {
                    lowe_precision_numerator++;
                    lowe_true_positive++;
                } else {
                    lowe_false_negative++;
                }
            }
        }
        else if (passed_ncc_matches.size() == 1){
            best_match = passed_ncc_matches[0];
        }
        if (!passed_ncc_matches.empty()) {
            per_edge_lowe_precision += static_cast<double>(lowe_precision_numerator) / passed_ncc_matches.size();
            lowe_edges_evaluated++;
        }
        lowe_output_counts.push_back(1);
    }   

    double epi_distance_recall = 0.0;
    if ((epi_true_positive + epi_false_negative) > 0) {
        epi_distance_recall = static_cast<double>(epi_true_positive) / (epi_true_positive + epi_false_negative);
    }

    double max_disparity_recall = 0.0;
    if ((disp_true_positive + disp_false_negative) > 0) {
        max_disparity_recall = static_cast<double>(disp_true_positive) / (disp_true_positive + disp_false_negative);
    }

    double epi_shift_recall = 0.0;
    if ((shift_true_positive + shift_false_negative) > 0) {
        epi_shift_recall = static_cast<double>(shift_true_positive) / (shift_true_positive + shift_false_negative);
    }

    double epi_cluster_recall = 0.0;
    if ((cluster_true_positive + cluster_false_negative) > 0) {
        epi_cluster_recall = static_cast<double>(cluster_true_positive) / (cluster_true_positive + cluster_false_negative);
    }

    double patch_recall = 0.0;
    if ((patch_true_positive + patch_false_negative) > 0) {
        patch_recall = static_cast<double>(patch_true_positive) / (patch_true_positive + patch_false_negative);
    }

    double ncc_recall = 0.0;
    if ((ncc_true_positive + ncc_false_negative) > 0) {
        ncc_recall = static_cast<double>(ncc_true_positive) / (ncc_true_positive + ncc_false_negative);
    }

    double lowe_recall = 0.0;
    if ((lowe_true_positive + lowe_false_negative) > 0) {
        lowe_recall = static_cast<double>(lowe_true_positive) / (lowe_true_positive + lowe_false_negative);
    }

    // std::cout << "Epipolar Distance Recall: " 
    //         << std::fixed << std::setprecision(2) 
    //         << epi_distance_recall * 100 << "%" << std::endl;

    // std::cout << "Max Disparity Threshold Recall: "
    //       << std::fixed << std::setprecision(2)
    //       << max_disparity_recall * 100 << "%" << std::endl;

    // std::cout << "Epipolar Shift Threshold Recall: "
    //       << std::fixed << std::setprecision(2)
    //       << epi_shift_recall * 100 << "%" << std::endl;

    // std::cout << "Epipolar Cluster Threshold Recall: "
    //       << std::fixed << std::setprecision(2)
    //       << epi_cluster_recall * 100 << "%" << std::endl;

    // std::cout << "Patch Threshold Recall: "
    //       << std::fixed << std::setprecision(2)
    //       << patch_recall * 100 << "%" << std::endl;

    // std::cout << "NCC Threshold Recall: "
    //       << std::fixed << std::setprecision(2)
    //       << ncc_recall * 100 << "%" << std::endl;

    // std::cout << "LRT Threshold Recall: "
    //       << std::fixed << std::setprecision(2)
    //       << lowe_recall * 100 << "%" << std::endl;

    double per_image_epi_precision = (epi_edges_evaluated > 0)
    ? (per_edge_epi_precision / epi_edges_evaluated)
    : 0.0;
    double per_image_disp_precision = (epi_edges_evaluated > 0)
    ? (per_edge_disp_precision / epi_edges_evaluated)
    : 0.0;
    double per_image_shift_precision = (epi_edges_evaluated > 0)
    ? (per_edge_shift_precision / epi_edges_evaluated)
    : 0.0;
    double per_image_clust_precision = (epi_edges_evaluated > 0)
    ? (per_edge_clust_precision / epi_edges_evaluated)
    : 0.0;
    double per_image_patch_precision = (patch_edges_evaluated > 0)
    ? (per_edge_patch_precision / patch_edges_evaluated)
    : 0.0;
    double per_image_ncc_precision = (ncc_edges_evaluated > 0)
    ? (per_edge_ncc_precision / ncc_edges_evaluated)
    : 0.0;
    double per_image_lowe_precision = (lowe_edges_evaluated > 0)
    ? (per_edge_lowe_precision / lowe_edges_evaluated)
    : 0.0;

    std::cout << "Epipolar Distance Precision: " 
        << std::fixed << std::setprecision(2) 
        << per_image_epi_precision * 100 << "%" << std::endl;
    std::cout << "Maximum Disparity Precision: " 
        << std::fixed << std::setprecision(2) 
        << per_image_disp_precision * 100 << "%" << std::endl;
    std::cout << "Epipolar Shift Precision: " 
        << std::fixed << std::setprecision(2) 
        << per_image_shift_precision * 100 << "%" << std::endl;
    std::cout << "Epipolar Cluster Precision: " 
        << std::fixed << std::setprecision(2) 
        << per_image_clust_precision * 100 << "%" << std::endl;
    std::cout << "Patch Precision: " 
        << std::fixed << std::setprecision(2) 
        << per_image_patch_precision * 100 << "%" << std::endl;
    std::cout << "NCC Precision: " 
        << std::fixed << std::setprecision(2) 
        << per_image_ncc_precision * 100 << "%" << std::endl;
    std::cout << "LRT Precision: " 
        << std::fixed << std::setprecision(2) 
        << per_image_lowe_precision * 100 << "%" << std::endl;

   return RecallMetrics {
       epi_distance_recall,
       max_disparity_recall,
       epi_shift_recall,
       epi_cluster_recall,
       patch_recall,
       ncc_recall,
       lowe_recall,
       epi_input_counts,
       epi_output_counts,
       disp_input_counts,
       disp_output_counts,
       shift_input_counts,
       shift_output_counts,
       clust_input_counts,
       clust_output_counts,
       patch_input_counts,
       patch_output_counts,
       ncc_input_counts,
       ncc_output_counts,
       lowe_input_counts,
       lowe_output_counts
   };
}  

double Dataset::ComputeNCC(const cv::Mat& patch_one, const cv::Mat& patch_two){
    double mean_one = (cv::mean(patch_one))[0];
    double mean_two = (cv::mean(patch_two))[0];
    double sum_of_squared_one  = (cv::sum((patch_one - mean_one).mul(patch_one - mean_one))).val[0];
    double sum_of_squared_two  = (cv::sum((patch_two - mean_two).mul(patch_two - mean_two))).val[0];

    cv::Mat norm_one = (patch_one - mean_one) / sqrt(sum_of_squared_one);
    cv::Mat norm_two = (patch_two - mean_two) / sqrt(sum_of_squared_two);
    return norm_one.dot(norm_two);
}

std::pair<std::vector<cv::Point2d>, std::vector<cv::Point2d>> Dataset::CalculateOrthogonalShifts(const std::vector<cv::Point2d>& edge_points, const std::vector<double>& orientations, double shift_magnitude){
   std::vector<cv::Point2d> shifted_points_one;
   std::vector<cv::Point2d> shifted_points_two;

   if (edge_points.size() != orientations.size()) {
       std::cerr << "ERROR: Mismatch between number of edge points and orientations." << std::endl;
       return {shifted_points_one, shifted_points_two};
   }

   for (size_t i = 0; i < edge_points.size(); i++) {
       const auto& edge_point = edge_points[i];
       double theta = orientations[i];

       double orthogonal_x1 = std::sin(theta);
       double orthogonal_y1 = -std::cos(theta);
       double orthogonal_x2 = -std::sin(theta);
       double orthogonal_y2 = std::cos(theta);

       double shifted_x1 = edge_point.x + shift_magnitude * orthogonal_x1;
       double shifted_y1 = edge_point.y + shift_magnitude * orthogonal_y1;
       double shifted_x2 = edge_point.x + shift_magnitude * orthogonal_x2;
       double shifted_y2 = edge_point.y + shift_magnitude * orthogonal_y2;

       shifted_points_one.emplace_back(shifted_x1, shifted_y1);
       shifted_points_two.emplace_back(shifted_x2, shifted_y2);
   }

   return {shifted_points_one, shifted_points_two};
}

void Dataset::ExtractPatches(
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
    std::vector<cv::Mat>& patch_set_two_out
)
{
    int half_patch = std::ceil(patch_size / 2);

    for (int i = 0; i < shifted_one.size(); i++) {
        double x1 = shifted_one[i].x;
        double y1 = shifted_one[i].y;
        double x2 = shifted_two[i].x;
        double y2 = shifted_two[i].y;

        bool in_bounds_one = (x1 - half_patch >= 0 && x1 + half_patch < image.cols &&
                            y1 - half_patch >= 0 && y1 + half_patch < image.rows);
        bool in_bounds_two = (x2 - half_patch >= 0 && x2 + half_patch < image.cols &&
                            y2 - half_patch >= 0 && y2 + half_patch < image.rows);

        if (in_bounds_one && in_bounds_two) {
            cv::Point2f center1(static_cast<float>(x1), static_cast<float>(y1));
            cv::Point2f center2(static_cast<float>(x2), static_cast<float>(y2));
            cv::Size size(patch_size, patch_size);

            cv::Mat patch1, patch2;
            cv::getRectSubPix(image, size, center1, patch1);
            cv::getRectSubPix(image, size, center2, patch2);

            if (patch1.type() != CV_32F) {
                patch1.convertTo(patch1, CV_32F);
            }
            if (patch2.type() != CV_32F) {
                patch2.convertTo(patch2, CV_32F);
            }

            filtered_edges_out.push_back(edges[i]);
            filtered_orientations_out.push_back(orientations[i]);
            patch_set_one_out.push_back(patch1);
            patch_set_two_out.push_back(patch2);

            if (right_edges && filtered_right_edges_out) {
                filtered_right_edges_out->push_back((*right_edges)[i]);
            }
        }
    }
}

std::vector<std::pair<std::vector<cv::Point2d>, std::vector<double>>> Dataset::ClusterEpipolarShiftedEdges(std::vector<cv::Point2d>& valid_shifted_edges, std::vector<double>& valid_shifted_orientations) {
    std::vector<std::pair<std::vector<cv::Point2d>, std::vector<double>>> clusters;
    
    if (valid_shifted_edges.empty() || valid_shifted_orientations.empty()) {
        return clusters;
    }

    std::vector<std::pair<cv::Point2d, double>> edge_orientation_pairs;
    for (size_t i = 0; i < valid_shifted_edges.size(); ++i) {
        edge_orientation_pairs.emplace_back(valid_shifted_edges[i], valid_shifted_orientations[i]);
    }

    std::sort(edge_orientation_pairs.begin(), edge_orientation_pairs.end(),
              [](const std::pair<cv::Point2d, double>& a, const std::pair<cv::Point2d, double>& b) {
                  return a.first.x < b.first.x;
              });

    valid_shifted_edges.clear();
    valid_shifted_orientations.clear();
    for (const auto& pair : edge_orientation_pairs) {
        valid_shifted_edges.push_back(pair.first);
        valid_shifted_orientations.push_back(pair.second);
    }

    std::vector<cv::Point2d> current_cluster_edges;
    std::vector<double> current_cluster_orientations;
    current_cluster_edges.push_back(valid_shifted_edges[0]);
    current_cluster_orientations.push_back(valid_shifted_orientations[0]);

    for (size_t i = 1; i < valid_shifted_edges.size(); ++i) {
        double distance = cv::norm(valid_shifted_edges[i] - valid_shifted_edges[i - 1]); 
        double orientation_difference = std::abs(valid_shifted_orientations[i] - valid_shifted_orientations[i - 1]);

        if (distance <= EDGE_CLUSTER_THRESH && orientation_difference < 5.0) {
            current_cluster_edges.push_back(valid_shifted_edges[i]);
            current_cluster_orientations.push_back(valid_shifted_orientations[i]);
        } else {
            clusters.emplace_back(current_cluster_edges, current_cluster_orientations);
            current_cluster_edges.clear();
            current_cluster_orientations.clear();
            current_cluster_edges.push_back(valid_shifted_edges[i]);
            current_cluster_orientations.push_back(valid_shifted_orientations[i]);
        }
    }

    if (!current_cluster_edges.empty()) {
        clusters.emplace_back(current_cluster_edges, current_cluster_orientations);
    }

    return clusters;
}

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

std::vector<Eigen::Vector3d> Dataset::CalculateEpipolarLine(const Eigen::Matrix3d& fund_mat, const std::vector<cv::Point2d>& edges) {
   std::vector<Eigen::Vector3d> epipolar_lines;

   for (const auto& point : edges) {
       Eigen::Vector3d homo_point(point.x, point.y, 1.0); 

       Eigen::Vector3d epipolar_line = fund_mat * homo_point;

       epipolar_lines.push_back(epipolar_line);
   }

   return epipolar_lines;
}

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

// Note: You could try to break this into a function that just reads the files, and a function that creates the files once and then never again!
// Note: Create the valid disparities CSV inside the outputs folder! Keep everything in one place!
void Dataset::CalculateGTRightEdge(const std::vector<cv::Point2d> &left_third_order_edges_locations, const std::vector<double> &left_third_order_edges_orientation, const cv::Mat &disparity_map, const cv::Mat &left_image, const cv::Mat &right_image) {
    gt_edge_data.clear();

    static size_t total_rows_written = 0;
    static int file_index = 1;
    static std::ofstream csv_file;
    static const size_t max_rows_per_file = 1'000'000;

    if (!csv_file.is_open()) {
        std::string filename = "valid_disparities_part_" + std::to_string(file_index) + ".csv";
        csv_file.open(filename, std::ios::out);
    }

    for (size_t i = 0; i < left_third_order_edges_locations.size(); i++) {
        const cv::Point2d &left_edge = left_third_order_edges_locations[i];
        double orientation = left_third_order_edges_orientation[i];

        double disparity = Bilinear_Interpolation(disparity_map, left_edge);

        if (std::isnan(disparity) || std::isinf(disparity) || disparity < 0) {
            continue;
        }

        cv::Point2d right_edge(left_edge.x - disparity, left_edge.y);
        gt_edge_data.emplace_back(left_edge, right_edge, orientation);

        if (total_rows_written >= max_rows_per_file) {
            csv_file.close();
            ++file_index;
            total_rows_written = 0;
            std::string next_filename = "valid_disparities_part_" + std::to_string(file_index) + ".csv";
            csv_file.open(next_filename, std::ios::out);
        }

        csv_file << disparity << "\n";
        ++total_rows_written;
    }

    csv_file.flush();
}

cv::Point2d Dataset::PerformEpipolarShift( 
    cv::Point2d original_edge_location, double edge_orientation,
    std::vector<double> epipolar_line_coeffs, bool& b_pass_epipolar_tengency_check )
{
    cv::Point2d corrected_edge;
    assert(epipolar_line_coeffs.size() == 3);
    double EL_coeff_A = epipolar_line_coeffs[0];
    double EL_coeff_B = epipolar_line_coeffs[1];
    double EL_coeff_C = epipolar_line_coeffs[2];
    double a1_line  = -epipolar_line_coeffs[0] / epipolar_line_coeffs[1];
    double b1_line  = -1;
    double c1_line  = -epipolar_line_coeffs[2] / epipolar_line_coeffs[1];
    
    //> Parameters of the line passing through the original edge along its direction (tangent) vector
    double a_edgeH2 = tan(edge_orientation);
    double b_edgeH2 = -1;
    double c_edgeH2 = -(a_edgeH2*original_edge_location.x - original_edge_location.y); //−(a⋅x2−y2)

    //> Find the intersected point of the two lines
    corrected_edge.x = (b1_line * c_edgeH2 - b_edgeH2 * c1_line) / (a1_line * b_edgeH2 - a_edgeH2 * b1_line);
    corrected_edge.y = (c1_line * a_edgeH2 - c_edgeH2 * a1_line) / (a1_line * b_edgeH2 - a_edgeH2 * b1_line);
    
    //> Find (i) the displacement between the original edge and the corrected edge, and
    //       (ii) the intersection angle between the epipolar line and the line passing through the original edge along its direction vector
    double epipolar_shift_displacement = cv::norm(corrected_edge - original_edge_location);
    double m_epipolar = -a1_line / b1_line; //> Slope of epipolar line
    double angle_diff_rad = abs(edge_orientation - atan(m_epipolar));
    double angle_diff_deg = angle_diff_rad * (180.0 / M_PI);
    if (angle_diff_deg > 180){
        angle_diff_deg -= 180;
    }

    //> check if the corrected edge passes the epoplar tengency test (intersection angle < 4 degrees and displacement < 6 pixels)
    b_pass_epipolar_tengency_check = (epipolar_shift_displacement < EPIP_TENGENCY_PROXIM_THRESH && abs(angle_diff_deg - 0) > EPIP_TENGENCY_ORIENT_THRESH && abs(angle_diff_deg - 180) > EPIP_TENGENCY_ORIENT_THRESH) ? (true) : (false);
    
    return corrected_edge;
}

std::vector<double> Dataset::LoadMaximumDisparityValues(const std::string& stereo_pairs_path, int num_pairs) {
    std::vector<double> max_disparities;
    std::string csv_filename = stereo_pairs_path + "/maximum_disparity_values.csv";
    std::ifstream file(csv_filename);
    std::string line;

    if (!file.is_open()) {
        std::cerr << "ERROR: Could not open file " << csv_filename << "!"<<std::endl;
        return max_disparities; 
    }

    int count = 0;
    while (std::getline(file, line) && count < num_pairs) {
        max_disparities.push_back(std::stod(line));
        count++;
    }

    file.close();
    return max_disparities;
}

std::vector<std::pair<cv::Mat, cv::Mat>> Dataset::LoadEuRoCImages(const std::string& csv_path, const std::string& left_path, const std::string& right_path,
   int num_pairs) {
   std::ifstream csv_file(csv_path);
   if (!csv_file.is_open()) {
       std::cerr << "ERROR: Could not open the CSV file located at " << csv_path << "!" << std::endl;
       return {};
   }

   std::vector<std::pair<cv::Mat, cv::Mat>> image_pairs;
   std::string line;
   bool first_line = true;

   while (std::getline(csv_file, line) && image_pairs.size() < num_pairs) {
       if (first_line) {
           first_line = false;
           continue;
       }

       std::istringstream line_stream(line);
       std::string timestamp;
       std::getline(line_stream, timestamp, ',');

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

std::vector<std::pair<cv::Mat, cv::Mat>> Dataset::LoadETH3DImages(const std::string &stereo_pairs_path, int num_pairs) {
    std::vector<std::pair<cv::Mat, cv::Mat>> image_pairs;

    std::vector<std::string> stereo_folders;
    for (const auto &entry : std::filesystem::directory_iterator(stereo_pairs_path)) {
        if (entry.is_directory()) {
            stereo_folders.push_back(entry.path().string());
        }
    }

    std::sort(stereo_folders.begin(), stereo_folders.end());

    for (int i = 0; i < std::min(num_pairs, static_cast<int>(stereo_folders.size())); ++i) {
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

std::vector<cv::Mat> Dataset::LoadETH3DMaps(const std::string &stereo_pairs_path, int num_maps) {
    std::vector<cv::Mat> disparity_maps;
    std::vector<std::string> stereo_folders;

    for (const auto &entry : std::filesystem::directory_iterator(stereo_pairs_path)) {
        if (entry.is_directory()) {
            stereo_folders.push_back(entry.path().string());
        }
    }

    std::sort(stereo_folders.begin(), stereo_folders.end());

    for (int i = 0; i < std::min(num_maps, static_cast<int>(stereo_folders.size())); i++) {
        std::string folder_path = stereo_folders[i];
        std::string disparity_csv_path = folder_path + "/disparity_map.csv";
        std::string disparity_bin_path = folder_path + "/disparity_map.bin";

        cv::Mat disparity_map;

        if (std::filesystem::exists(disparity_bin_path)) {
            // std::cout << "Loading disparity data from: " << disparity_bin_path << std::endl;
            disparity_map = ReadDisparityFromBinary(disparity_bin_path);
        } else {
            // std::cout << "Parsing and storing disparity data from: " << disparity_csv_path << std::endl;
            disparity_map = LoadDisparityFromCSV(disparity_csv_path);
            if (!disparity_map.empty()) {
                WriteDisparityToBinary(disparity_bin_path, disparity_map);
                // std::cout << "Saved disparity data to: " << disparity_bin_path << std::endl;
            }
        }

        if (!disparity_map.empty()) {
            disparity_maps.push_back(disparity_map);
        }
    }

    return disparity_maps;
}

void Dataset::WriteDisparityToBinary(const std::string& filepath, const cv::Mat& disparity_map) {
    std::ofstream ofs(filepath, std::ios::binary);
    if (!ofs.is_open()) {
        std::cerr << "ERROR: Could not write disparity to: " << filepath << std::endl;
        return;
    }

    int rows = disparity_map.rows;
    int cols = disparity_map.cols;
    ofs.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    ofs.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
    ofs.write(reinterpret_cast<const char*>(disparity_map.ptr<float>(0)), sizeof(float) * rows * cols);
}

cv::Mat Dataset::ReadDisparityFromBinary(const std::string& filepath) {
    std::ifstream ifs(filepath, std::ios::binary);
    if (!ifs.is_open()) {
        std::cerr << "ERROR: Could not read disparity from: " << filepath << std::endl;
        return {};
    }

    int rows, cols;
    ifs.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    ifs.read(reinterpret_cast<char*>(&cols), sizeof(cols));

    cv::Mat disparity_map(rows, cols, CV_32F);
    ifs.read(reinterpret_cast<char*>(disparity_map.ptr<float>(0)), sizeof(float) * rows * cols);

    return disparity_map;
}

cv::Mat Dataset::LoadDisparityFromCSV(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "ERROR: Could not open disparity CSV: " << path << std::endl;
        return {};
    }

    std::vector<std::vector<float>> data;
    std::string line;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        std::vector<float> row;

        while (std::getline(ss, value, ',')) {
            try {
                float d = std::stof(value);

                if (value == "nan" || value == "NaN") {
                    d = std::numeric_limits<float>::quiet_NaN();
                } else if (value == "inf" || value == "Inf") {
                    d = std::numeric_limits<float>::infinity();
                } else if (value == "-inf" || value == "-Inf") {
                    d = -std::numeric_limits<float>::infinity();
                }

                row.push_back(d);
            } catch (const std::exception &e) {
                    std::cerr << "WARNING: Invalid value in file: " << path << " -> " << value << std::endl;
                    row.push_back(std::numeric_limits<float>::quiet_NaN()); 
            }
        }

        if (!row.empty()) data.push_back(row);
    }

    int rows = data.size();
    int cols = data[0].size();
    cv::Mat disparity(rows, cols, CV_32F);

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            disparity.at<float>(r, c) = data[r][c];
        }
    }

    return disparity;
}

void Dataset::WriteEdgesToBinary(const std::string& filepath,
                                  const std::vector<cv::Point2d>& locations,
                                  const std::vector<double>& orientations) {
    std::ofstream ofs(filepath, std::ios::binary);
    if (!ofs.is_open()) {
        std::cerr << "ERROR: Could not open binary file for writing: " << filepath << std::endl;
        return;
    }

    size_t size = locations.size();
    ofs.write(reinterpret_cast<const char*>(&size), sizeof(size));
    ofs.write(reinterpret_cast<const char*>(locations.data()), sizeof(cv::Point2d) * size);
    ofs.write(reinterpret_cast<const char*>(orientations.data()), sizeof(double) * size);
}

void Dataset::ReadEdgesFromBinary(const std::string& filepath,
                                   std::vector<cv::Point2d>& locations,
                                   std::vector<double>& orientations) {
    std::ifstream ifs(filepath, std::ios::binary);
    if (!ifs.is_open()) {
        std::cerr << "ERROR: Could not open binary file for reading: " << filepath << std::endl;
        return;
    }

    size_t size = 0;
    ifs.read(reinterpret_cast<char*>(&size), sizeof(size));

    locations.resize(size);
    orientations.resize(size);

    ifs.read(reinterpret_cast<char*>(locations.data()), sizeof(cv::Point2d) * size);
    ifs.read(reinterpret_cast<char*>(orientations.data()), sizeof(double) * size);
}

void Dataset::ProcessEdges(const cv::Mat& image,
                           const std::string& filepath,
                           std::shared_ptr<ThirdOrderEdgeDetectionCPU>& toed,
                           std::vector<cv::Point2d>& locations,
                           std::vector<double>& orientations) {
    std::string path = filepath + ".bin";

    if (std::filesystem::exists(path)) {
        // std::cout << "Loading edge data from: " << path << std::endl;
        ReadEdgesFromBinary(path, locations, orientations);
    } else {
        // std::cout << "Running third-order edge detector..." << std::endl;
        toed->get_Third_Order_Edges(image);
        locations = toed->toed_locations;
        orientations = toed->toed_orientations;

        WriteEdgesToBinary(path, locations, orientations);
        // std::cout << "Saved edge data to: " << path << std::endl;
    }
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

#endif