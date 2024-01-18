#include <stdio.h>
#include <stdlib.h>
#include <stdexcept>
#include <iostream>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <yaml-cpp/yaml.h>

#include "../include/definitions.h"
#include "../include/Dataset.h"
#include "../include/Pipeline.h"

// =======================================================================================================
// main_VO: main function for LEMS VO pipeline
//
// ChangeLogs
//    Chien  23-01-16    Initially built on top of Hongyi's LEMS Visual Odometry framework.
//
//> (c) LEMS, Brown University
//> Chiang-Heng Chien (chiang-heng_chien@brown.edu)
// =======================================================================================================

//> usage: sudo ./main_VO --config_file=../config/tum.yaml

//> Define default values for the input argument
DEFINE_string(config_file, "../config/tum.yaml", "config file path");

int main(int argc, char **argv) {

	//> Get input arguments
	google::ParseCommandLineFlags(&argc, &argv, true);
	YAML::Node config_map;
	
	try {
		config_map = YAML::LoadFile(FLAGS_config_file);
#if SHOW_YAML_FILE_DATA
		std::cout << config_map << std::endl;
#endif
	}
	catch (const std::exception& e) {
		std::cerr << "Exception: " << e.what() << std::endl;
		std::cerr << "File does not exist!" << std::endl;
	}

	//> Setup the dataset class pointer
	Dataset::Ptr dataset_ = Dataset::Ptr(new Dataset(config_map));
    CHECK_EQ(dataset_->Init_Fetch_Data(), true);

	Frame::Ptr new_frame = dataset_->get_Next_Frame();
	if (new_frame == nullptr) std::cerr << "ERROR: failed to fetch a frame." << std::endl;

	Pipeline::Ptr vo_sys = Pipeline::Ptr(new Pipeline);
	bool success = vo_sys->Add_Frame(new_frame);

	//std::cout << vo_sys->status_ << std::endl;
	std::cout << "Number of SIFT Features: " << vo_sys->Num_Of_SIFT_Features << std::endl;

	new_frame = dataset_->get_Next_Frame();
	success = vo_sys->Add_Frame(new_frame);
	std::cout << vo_sys->Num_Of_Good_Feature_Matches << std::endl;
	

	return 0;
}
