#ifndef MOTION_TRACKER_H
#define MOTION_TRACKER_H

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "Frame.h"
#include "utility.h"

// =============================================================================================================================
// class MotionTracker: track camera motion, i.e., estimate camera poses, similar to "tracking" used in ORB-SLAM or OpenVSLAM,
//                      but the name aims to differentiate "camera motion tracks" from "feature tracks".
//
// ChangeLogs
//    Chien  24-05-18    Initially created. Add relative pose under a RANSAC loop with depth prior.
//
//> (c) LEMS, Brown University
//> Chiang-Heng Chien (chiang-heng_chien@brown.edu)
// ======================================================================================================================

class MotionTracker {

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<MotionTracker> Ptr;

    //> Constructor (nothing special)
    MotionTracker();

    //> 

    /**
     * Estimate relative poses
     * @return true if success
     */
    bool get_Relative_Pose( Frame::Ptr Curr_Frame, Frame::Ptr Prev_Frame, int Num_Of_Good_Feature_Matches, bool use_GCC_filter = false );

    /**
     * Geometric Correspondence Consistency (GCC) filter acting in the observation space
     * @return d as distance from point to curve in pixels
     */
    double get_GCC_dist( Frame::Ptr Curr_Frame, Frame::Ptr Prev_Frame, int anchor_index, int picked_index );

private:
    

    //> Pointers to the classes
    Utility::Ptr utility_tool = nullptr;
};


#endif
