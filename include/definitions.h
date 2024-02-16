//> Macro definitions

#define REPO_PATH                       std::string("/home/chchien/BrownU/research/LEMS_VO_SLAM/Chien_Version/")
#define OUTPUT_WRITE_PATH               std::string("Output_Results/")
#define OUTPUT_DATA_TYPE                std::string("TUM")  //> Either TUM or KITTI

//> Use for the TUM type dataset
#define ASSOCIATION_FILE_NAME           std::string("associate.txt")

//> Geometric Depth Consistency
#define TOP_N_RANK_ORDERED_LIST         (100)

#define DEPTH_GRAD_GAUSSIAN_SIGMA       (3)
#define GAUSSIAN_KERNEL_WINDOW_LENGTH   (2*4*DEPTH_GRAD_GAUSSIAN_SIGMA+1)

//> DEBUGGING PURPOSE
#define SHOW_YAML_FILE_DATA             (false)
#define WRITE_FEATURES_TO_FILE          (false)
#define WRITE_CORRESPONDENCES_TO_FILE   (false)
#define OPENCV_DISPLAY_FEATURES         (false)
#define OPENCV_DISPLAY_CORRESPONDENCES  (false)

//> SIFT parameters
#define SIFT_NFEATURES                  (0)
#define SIFT_NOCTAVE_LAYERS             (4)
#define SIFT_CONTRAST_THRESHOLD         (0.04)
#define SIFT_EDGE_THRESHOLD             (10)
#define SIFT_GAUSSIAN_SIGMA             (1.6)

#define LOWES_RATIO                     (0.8)        //> Suggested in Lowe's paper
#define K_IN_KNN_MATCHING               (2)

//> Print outs
#define LOG_STATUS(status_)             printf("\033[1;35mSTATUS: %s\033[0m\n", status_);
#define LOG_ERROR(err_msg)              printf("\033[1;31mERROR: %s\033[0m\n", err_msg );
