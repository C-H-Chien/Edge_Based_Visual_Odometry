# Visual Odometry / SLAM Pipeline Research @Brown University
This repository is still under development. 
(c) Chiang-Heng Chien (chiang-heng_chien@brown.edu)

## Dependencies
* Eigen 3.X
* OpenCV 4.X with opencv_contrib (feature detection and matching depends on opencv_contrib)
* [yaml-cpp](https://github.com/jbeder/yaml-cpp) (used to parse data from the dataset config file)
* glog
* gflags

## Build and Compile
Follow the standard compilation, namely, 
```bash
mkdir build && cd build
cmake ..
make -j
```
You shall find executive files under the ``bin`` folder.

## Usage
Under the ``bin`` folder, do
```bash
./main_VO --config_file=../config/tum.yaml
```
where the configuration file can be customized depending on the dataset in use. See ``.yaml`` files under the ``config`` folder for more information. If no ``--config_file`` input argument is given, the default value will be used which is defined in ``cmd/main_VO.cpp``.

## Notes
Make sure that the path of your YAML-CPP header files matches the path in the ``CMakeLists.txt``. See this line.

## Functionalities
Relative pose estimation from a RGB-D sequence, with geometric correspondence consistency (GCC) filter to speedup and stablize the RANSAC process.