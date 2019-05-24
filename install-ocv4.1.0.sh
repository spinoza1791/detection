#!/bin/sh -x

sudo apt-get install -y build-essential git cmake unzip pkg-config && sudo apt-get install -y libjpeg-dev libpng-dev libtiff-dev && sudo apt-get install -y libavcodec-dev libavformat-dev libswscale-dev libv4l-dev && sudo apt-get install -y libxvidcore-dev libx264-dev && sudo apt-get install -y libgtk-3-dev && sudo apt-get install -y libatlas-base-dev gfortran && sudo apt-get install -y python-dev python3-dev python3-pip python-pip python3-setuptools python3-wheel python3-numpy python3-scipy python3-matplotlib && sudo apt-get install -y libeigen2-dev libeigen3-dev libopenexr-dev libgstreamer1.0-dev libgstreamermm-1.0-dev libgoogle-glog-dev libgflags-dev libprotobuf-c-dev libprotobuf-dev protobuf-c-compiler protobuf-compiler

//cd ~ && wget -O opencv.zip https://github.com/opencv/opencv/archive/4.1.0.zip && wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.1.0.zip && unzip opencv.zip && unzip opencv_contrib.zip && mv opencv-4.1.0 opencv && mv opencv_contrib-4.1.0 opencv_contrib

cd ~/opencv && mkdir build && cd build

cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
    -D ENABLE_NEON=ON \
    -D OPENMP=ON \
    -D INSTALL_C_EXAMPLES=OFF \
    -D OPENCV_ENABLE_NONFREE=ON \
    -D BUILD_TESTS=OFF \
    -D BUILD_DOCS=OFF \
    -D BUILD_PERF_TESTS=OFF \
    -D INSTALL_PYTHON_EXAMPLES=OFF \
    -D PYTHON3_EXECUTABLE=$(which python3) \
    -D BUILD_opencv_python3=YES \
    -D BUILD_opencv_python2=YES \
    -D PYTHON_DEFAULT_EXECUTABLE=$(which python3) \
    -D BUILD_EXAMPLES=OFF ..

cd ~/opencv/build && make -j4 && sudo make install & sudo ldconfig
