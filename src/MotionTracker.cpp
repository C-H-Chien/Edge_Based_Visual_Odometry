#ifndef MOTION_TRACKER_CPP
#define MOTION_TRACKER_CPP

#include <Eigen/Core>
#include <Eigen/Dense>
#include "MotionTracker.h"
#include "definitions.h"

// ============================================================================================================================
// class MotionTracker: track camera motion, i.e., estimate camera poses, similar to "tracking" used in ORB-SLAM or OpenVSLAM,
//                      but the name aims to differentiate "camera motion tracks" from "feature tracks".
//
// ChangeLogs
//    Chien  24-01-17    Initially created.
//
//> (c) LEMS, Brown University
//> Chiang-Heng Chien (chiang-heng_chien@brown.edu)
// ============================================================================================================================

MotionTracker::MotionTracker() {

}

bool MotionTracker::get_Relative_Pose( Frame::Ptr Curr_Frame, Frame::Ptr Prev_Frame, int Num_Of_Good_Feature_Matches, bool use_GCC_filter ) {

    //> RANSAC loop
    for (unsigned iter = 0; iter < RANSAC_NUM_OF_ITERATIONS; iter++) {
        //> 1) Hypothesis formulation
        //> Randomly pick 3 points from Num_Of_Good_Feature_Matches passed by Pipeline.cpp
        int Sample_Indices[3] = {0, 0, 0};
        for (int i = 0; i < 3; i++) 
            Sample_Indices[i] = utility_tool->Uniform_Random_Number_Generator<int>(0, Num_Of_Good_Feature_Matches-1);

        if (use_GCC_filter) {
            double gcc_dist_forward, gcc_dist_backward;
            //> Make Sample_Indices[0] as the anchor index and the other two as picked index
            gcc_dist_forward  = get_GCC_dist( Curr_Frame, Prev_Frame, Sample_Indices[0], Sample_Indices[1] );
            gcc_dist_backward = get_GCC_dist( Prev_Frame, Curr_Frame, Sample_Indices[0], Sample_Indices[1] );

        }
        
        
        //> 2) Hypothesis support measurement

    }
    

    
    return 0;
}

double MotionTracker::get_GCC_dist( Frame::Ptr Curr_Frame, Frame::Ptr Prev_Frame, int anchor_index, int picked_index ) {

    //> View 1 is previous frame, view 2 is current frame
    double phi_view1 = (Prev_Frame->Gamma[ anchor_index ] - Prev_Frame->Gamma[ picked_index ]).norm();
    double phi_view2 = (Curr_Frame->Gamma[ anchor_index ] - Curr_Frame->Gamma[ picked_index ]).norm();
    Eigen::Vector3d gamma_view2   = Prev_Frame->inv_K * Prev_Frame->SIFT_Match_Locations_Pixels[ picked_index ];
    Eigen::Vector3d gamma_0_view2 = Prev_Frame->inv_K * Prev_Frame->SIFT_Match_Locations_Pixels[ anchor_index ];

    double rho_0 = (Curr_Frame->Gamma[ anchor_index ])(2);
    double rho_p = (Curr_Frame->Gamma[ picked_index ])(2);

    double gradient_phi_xi  = 2*(rho_p*(gamma_view2.norm()*gamma_view2.norm()) + rho_0*(gamma_view2.dot(gamma_0_view2))) * Curr_Frame->gradient_Depth_at_Features[ picked_index ].first \
                            + 2*rho_p*( (1.0/Curr_Frame->K(0,0))*(rho_p*gamma_view2(0) - rho_0*gamma_0_view2(0)) );
    double gradient_phi_eta = 2*(rho_p*(gamma_view2.norm()*gamma_view2.norm()) + rho_0*(gamma_view2.dot(gamma_0_view2))) * Curr_Frame->gradient_Depth_at_Features[ picked_index ].second \
                            + 2*rho_p*( (1.0/Curr_Frame->K(1,1))*(rho_p*gamma_view2(1) - rho_0*gamma_0_view2(1)) );

    double gradient_phi = sqrt(gradient_phi_xi*gradient_phi_xi + gradient_phi_eta*gradient_phi_eta);
    return fabs(phi_view1 - phi_view2) / gradient_phi;
}

#endif