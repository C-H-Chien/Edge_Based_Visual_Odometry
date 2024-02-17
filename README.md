# Visual Odometry / SLAM Pipeline Research @Brown University
(c) Chiang-Heng Chien (chiang-heng_chien@brown.edu)

## Dependencies
* Eigen 3.X
* yaml-cpp
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