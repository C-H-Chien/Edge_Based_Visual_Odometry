# Visual Odometry / SLAM Pipeline Research @Brown University
This repository is still under development. 
(c) Chiang-Heng Chien (chiang-heng_chien@brown.edu)

## Dependencies
* Eigen 3.X
* OpenCV 4.X
* yaml-cpp (see its [official page](https://github.com/jbeder/yaml-cpp))
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
where the configuration file can be customized. See ``.yaml`` files under the ``config`` folder for more information.

## Functionalities
Relative pose estimation from a RGB-D sequence, with geometric correspondence consistency (GCC) filter to speedup and stablize the RANSAC process.