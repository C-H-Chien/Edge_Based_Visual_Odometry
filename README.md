# Edge-Based Visual Odometry 
## Research @ LEMS, Brown University
This repository is an internal research project at LEMS lab, Brown University. It is still under development. 

## Dependencies
* Eigen 3.X or beyond
* OpenCV 4.X with opencv_contrib (feature detection and matching depends on opencv_contrib)
* [yaml-cpp](https://github.com/jbeder/yaml-cpp) (used to parse data from the dataset config file)
* glog (Optional; if not used, comment the [macro definition](https://github.com/C-H-Chien/Edge_Based_Visual_Odometry/blob/55bb7cba10cef12bf177b05b091fd6ddb47415e2/include/definitions.h#L2) in ``include/definitions.h`` or set it as false, and the ``find_package`` in the ``CMakeLists.txt`` file.)

## Build and Compile
Follow the standard build and compilation, namely, 
```bash
mkdir build && cd build
cmake ..
make -j
```
You shall find executive files under the ``bin`` folder. The code to be executed is ``./main_VO`` while ``./test_functions`` executes ``test_functions.cpp`` used to test certain functionalities.

## Usage
Under the ``bin`` folder, if glog is used
```bash
./main_VO --config_file=../config/tum.yaml
```
If not, do
```bash
./main_VO --config_file ../config/tum.yaml
```
where the configuration file can be customized depending on the dataset in use. See ``.yaml`` files under the ``config`` folder for more information. If no ``--config_file`` input argument is given, the default value will be used which is defined in ``cmd/main_VO.cpp`` if glog is used. Otherwise, a help message will show up. <br />
There is also a ``./test_functions`` executive file which runs the ``test/test_functions.cpp`` and is used to test and verify some functionalities of the C++ implementation.

## Notes
Make sure that the path of your YAML-CPP header files matches the path in the ``CMakeLists.txt``. See this line.

## Authors
Saul Lopez Lucas (saul_lopez_lucas@brown.edu) <br />
Chiang-Heng Chien (chiang-heng_chien@brown.edu)