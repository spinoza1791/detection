#!/bin/sh -x

sudo apt-get -y remove x264 libx264-dev
 
## Install dependencies
sudo apt-get -y install build-essential checkinstall cmake pkg-config yasm
sudo apt-get -y install git gfortran
sudo apt-get -y install libjpeg8-dev libjasper-dev libpng12-dev
sudo apt-get -y install libtiff5-dev
sudo apt-get -y install libtiff-dev
sudo apt-get -y install libavcodec-dev libavformat-dev libswscale-dev libdc1394-22-dev
sudo apt-get -y install libxine2-dev libv4l-dev
cd /usr/include/linux
sudo ln -s -f ../libv4l1-videodev.h videodev.h
cd $cwd
sudo apt-get -y install libgstreamer0.10-dev libgstreamer-plugins-base0.10-dev
sudo apt-get -y install libgtk2.0-dev libtbb-dev qt5-default
sudo apt-get -y install libatlas-base-dev
sudo apt-get -y install libmp3lame-dev libtheora-dev
sudo apt-get -y install libvorbis-dev libxvidcore-dev libx264-dev
sudo apt-get -y install libopencore-amrnb-dev libopencore-amrwb-dev
sudo apt-get -y install libavresample-dev
sudo apt-get -y install x264 v4l-utils
 
# Optional dependencies
sudo apt-get -y install libprotobuf-dev protobuf-compiler
sudo apt-get -y install libgoogle-glog-dev libgflags-dev
sudo apt-get -y install libgphoto2-dev libeigen3-dev libhdf5-dev doxygen

sudo apt-get install -y \
     gettext \
     ccache \
     pkg-config \
     libpng-dev \
     libpng++-dev \
     libjpeg-dev \
     libtiff5-dev \
     libjasper-dev \
     libavcodec-dev \
     libavformat-dev \
     libavresample-dev \
     libswresample-dev \
     libavutil-dev \
     libswscale-dev \
     libv4l-dev \
     libxvidcore-dev \
     v4l-utils \
     libx264-dev \
     libgtk-3-dev \
     libgdk-pixbuf2.0-dev \
     libpango1.0-dev \
     libcairo2-dev \
     libfontconfig1-dev \
     libatlas-base-dev \
     liblapack-dev \
     liblapacke-dev \
     libblas-dev \
     libopenblas-dev \
     gfortran \
     python-pip \
     python3-pip \
     python-numpy \
     python-dev \
     python3-dev \
     libeigen2-dev \
     libeigen3-dev \
     libopenexr-dev \
     libgstreamer1.0-dev \
     libgstreamermm-1.0-dev \
     libgoogle-glog-dev \
     libgflags-dev \
     libprotobuf-c-dev \
     libprotobuf-dev \
     protobuf-c-compiler \
     protobuf-compiler \
     libgphoto2-dev \
     qt5-default \
     libvtk6-dev \
     libvtk6-qt-dev \
     libhdf5-dev \
     freeglut3-dev \
     libgtkglext1-dev \
     libgtkglextmm-x11-1.2-dev \
     libwebp-dev \
     libtbb-dev \
     libdc1394-22-dev \
     libunicap2-dev \
     ffmpeg 
     
//sudo apt-get install -y build-essential git cmake unzip pkg-config && sudo apt-get install -y libjpeg-dev libpng-dev libtiff-dev && sudo apt-get install -y libavcodec-dev libavformat-dev libswscale-dev libv4l-dev && sudo apt-get install -y libxvidcore-dev libx264-dev && sudo apt-get install -y libgtk-3-dev && sudo apt-get install -y libatlas-base-dev gfortran && 

sudo apt-get install -y \
    python3-setuptools \
    python3-wheel \
    python3-numpy \
    python3-scipy \
    python3-matplotlib

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
