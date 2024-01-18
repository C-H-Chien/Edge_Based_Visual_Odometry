#ifndef PIPELINE_CPP
#define PIPELINE_CPP

#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include "Pipeline.h"
#include "definitions.h"
#include "utility.h"

// =====================================================================================================================
// class Pipeline: visual odometry pipeline 
//
// ChangeLogs
//    Chien  23-01-17    Initially created.
//    Chien  23-01-18    Add (SIFT) feature detection, matching, and creating a rank-order list of correspondences
//
//> (c) LEMS, Brown University
//> Chiang-Heng Chien (chiang-heng_chien@brown.edu)
// ======================================================================================================================

Pipeline::Pipeline() {
    //> Set zeros
    Num_Of_SIFT_Features = 0;
    Num_Of_Good_Feature_Matches = 0;
}

bool Pipeline::Add_Frame(Frame::Ptr frame) {
    //> Set the frame pointer
    Current_Frame = frame;

    switch (status_) {
        case PipelineStatus::STATE_INITIALIZATION:
            //> Initialization: extract SIFT features on the first frame
            LOG_STATES("STATE_INITIALIZATION");
            Num_Of_SIFT_Features = get_Features();
            break;
        case PipelineStatus::STATE_GET_AND_MATCH_SIFT:
            //> Extract and match SIFT features of the current frame with the previous frame
            LOG_STATES("STATE_GET_AND_MATCH_SIFT");
            Num_Of_SIFT_Features = get_Features();
            Num_Of_Good_Feature_Matches = get_Feature_Correspondences();
            break;
        //case PipelineStatus::STATE_
    }

    //> Swap the frame
    Previous_Frame = Current_Frame;

    return true;
}


int Pipeline::get_Features() {

    //> SIFT feature detector. Parameters same as the VLFeat default values, defined in Macros.
    //cv::Ptr<cv::xfeatures2d::SIFT> sift;
    cv::Ptr<cv::SIFT> sift;
    sift = cv::SIFT::create(SIFT_NFEATURES, SIFT_NOCTAVE_LAYERS, SIFT_CONTRAST_THRESHOLD, \
                            SIFT_EDGE_THRESHOLD, SIFT_GAUSSIAN_SIGMA);

    //> Detect SIFT keypoints and extract SIFT descriptor from the image
    std::vector<cv::KeyPoint> SIFT_Keypoints;
    cv::Mat SIFT_KeyPoint_Descriptors;
    sift->detect(Current_Frame->Image, SIFT_Keypoints);
    sift->compute(Current_Frame->Image, SIFT_Keypoints, SIFT_KeyPoint_Descriptors);

    //> Copy to the class Frame
    Current_Frame->SIFT_Locations = SIFT_Keypoints;
    Current_Frame->SIFT_Descriptors = SIFT_KeyPoint_Descriptors;

    //> Change to the next state
    status_ = PipelineStatus::STATE_GET_AND_MATCH_SIFT;
    
    return SIFT_Keypoints.size();
}

int Pipeline::get_Feature_Correspondences() {

    //> Match SIFT features via OpenCV built-in KNN approach
    //  Matching direction: from previous frame -> current frame
    cv::BFMatcher matcher;
    std::vector< std::vector< cv::DMatch > > feature_matches;
    matcher.knnMatch( Previous_Frame->SIFT_Descriptors, Current_Frame->SIFT_Descriptors, feature_matches, K_IN_KNN_MATCHING );
    
    //> Apply Lowe's ratio test
    std::vector< cv::DMatch > Good_Matches;
    for (int i = 0; i < feature_matches.size(); ++i) {
        if (feature_matches[i][0].distance < LOWES_RATIO * feature_matches[i][1].distance) {
            Good_Matches.push_back(feature_matches[i][0]);
        }
    }

    //> Sort the matches based on their matching score (Euclidean distances of feature descriptors)
    std::sort( Good_Matches.begin(), Good_Matches.end(), less_than_Eucl_Dist() );

    //> Push back the match feature locations of the previous and current frames
    for (int fi = 0; fi < Good_Matches.size(); fi++) {
        cv::DMatch f = Good_Matches[fi];

        //> Push back the match feature locations of the previous and current frames
        cv::Point2d previous_pt = Previous_Frame->SIFT_Locations[f.queryIdx].pt;
        cv::Point2d current_pt = Current_Frame->SIFT_Locations[f.trainIdx].pt;

        Eigen::Vector3d Previous_Match_Location = {previous_pt.x, previous_pt.y, 1.0};
        Eigen::Vector3d Current_Match_Location = {current_pt.x, current_pt.y, 1.0};

        Previous_Frame->SIFT_Match_Locations.push_back(Previous_Match_Location);
        Current_Frame->SIFT_Match_Locations.push_back(Current_Match_Location);
    }

#if OPENCV_DISPLAY_CORRESPONDENCES
    IO_TOOLS::Display_Feature_Correspondences(Previous_Frame->Image, Current_Frame->Image, \
                                    Previous_Frame->SIFT_Locations, Current_Frame->SIFT_Locations, \
                                    Good_Matches ) ;
#endif

    //> Get 3D matches



    //> Change to the next state
    //status_ = PipelineStatus::STATE_GET_AND_MATCH_SIFT;

    return Good_Matches.size();
}

#endif