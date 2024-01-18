//> Macro definitions

#define REPO_PATH                       std::string("/home/chchien/BrownU/research/LEMS_VO_SLAM/Chien_Version/")
#define OUTPUT_WRITE_PATH               std::string("Output_Results/")
#define OUTPUT_DATA_TYPE                std::string("TUM")  //> Either TUM or KITTI

//> Use for the TUM type dataset
#define ASSOCIATION_FILE_NAME           std::string("associate.txt")

//> DEBUGGING PURPOSE
#define SHOW_YAML_FILE_DATA             (false)
#define WRITE_FEATURES_TO_FILE          (false)
#define WRITE_CORRESPONDENCES_TO_FILE   (false)
#define OPENCV_DISPLAY_FEATURES         (false)
#define OPENCV_DISPLAY_CORRESPONDENCES  (true)

//> SIFT parameters
#define SIFT_NFEATURES                  (0)
#define SIFT_NOCTAVE_LAYERS             (4)
#define SIFT_CONTRAST_THRESHOLD         (0.04)
#define SIFT_EDGE_THRESHOLD             (10)
#define SIFT_GAUSSIAN_SIGMA             (1.6)

#define LOWES_RATIO                     (0.8)        //> Suggested in Lowe's paper
#define K_IN_KNN_MATCHING               (2)

//> Print outs
#define LOG_STATES(state_)              printf("\033[1;35mSTATE: %s\033[0m\n", state_);
