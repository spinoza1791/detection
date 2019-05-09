#!/bin/sh -x

V=4.1.0

#sudo umount /tmp
#sudo mount -t tmpfs -o size=10485760,mode=1777 overflow /tmp

sudo apt-get install -y \
     cmake \
     gettext \
     ccache \
     pkg-config \
     libpng-dev \
     libpng++-dev \
     libjpeg-dev \
     libtiff5-dev \
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
     python3-numpy \
     python-dev \
     python3-dev \
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

sudo -H pip3 install wheel numpy 
sudo apt-get install -y python3-scipy python3-matplotlib
cd /tmp && sudo rm -rf * && cd ~
#sudo -H pip3 install scikit-image scikit-learn ipython dlib
#cd /tmp && sudo rm -rf * && cd ~

git clone --depth=1 -b ${V} --single-branch https://github.com/opencv/opencv.git
git clone --depth=1 -b ${V} --single-branch https://github.com/opencv/opencv_contrib.git
cd opencv
mkdir -p build
cd build

#export CFLAGS="-mcpu=cortex-a53 -mfpu=neon-vfpv4 -ftree-vectorize -mfloat-abi=hard -fPIC -O3"
#export CXXFLAGS="-mcpu=cortex-a53 -mfpu=neon-vfpv4 -ftree-vectorize -mfloat-abi=hard -fPIC -O3"
#      -D CMAKE_CXX_FLAGS_RELEASE=-mcpu=cortex-a53 -mfpu=neon-vfpv4 -ftree-vectorize -mfloat-abi=hard \
#      -D CMAKE_C_FLAGS_RELEASE=-mcpu=cortex-a53 -mfpu=neon-vfpv4 -ftree-vectorize -mfloat-abi=hard \
cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
    -D ENABLE_NEON=ON \
    -D OPENMP=ON \
    -D BUILD_TESTS=OFF \
    -D INSTALL_PYTHON_EXAMPLES=OFF \
    -D PYTHON3_EXECUTABLE=$(which python3) \
    -D PYTHON_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
    -D PYTHON_INCLUDE_DIR2=$(python3 -c "from os.path import dirname; from distutils.sysconfig import get_config_h_filename; print(dirname(get_config_h_filename()))") \
    -D PYTHON_LIBRARY=$(python3 -c "from distutils.sysconfig import get_config_var;from os.path import dirname,join ; print(join(dirname(get_config_var('LIBPC')),get_config_var('LDLIBRARY')))") \
    -D PYTHON3_NUMPY_INCLUDE_DIRS=$(python3 -c "import numpy; print(numpy.get_include())") \
    -D PYTHON3_PACKAGES_PATH=$(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
    -D BUILD_opencv_python3=YES \
    -D BUILD_opencv_python2=YES \
    -D PYTHON_DEFAULT_EXECUTABLE=$(which python3) \
    -D BUILD_EXAMPLES=OFF ..

cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D OPENCV_GENERATE_PKGCONFIG=YES \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
      -D BUILD_SHARED_LIBS=ON \
      -D OPENCV_ENABLE_NONFREE=ON \
      -D BUILD_DOCS=OFF \
      -D BUILD_ZLIB=OFF \
      -D BUILD_PERF_TESTS=OFF \
      -D BUILD_WEBP=OFF \
      -D BUILD_TESTS=OFF \
      -D BUILD_EXAMPLES=OFF \
      -D BUILD_JAVA=OFF \
      -D WITH_GSTREAMER=ON \
      -D WITH_GTK=OFF \
      -D WITH_JPEG=ON \
      -D WITH_OPENEXR=ON \
      -D WITH_PNG=ON \
      -D WITH_TIFF=ON \
      -D WITH_V4L=ON \
      -D WITH_LIBV4L=ON \
      -D WITH_VTK=ON \
      -D WITH_LAPACK=ON \
      -D WITH_LAPACKE=ON \
      -D WITH_PROTOBUF=ON \
      -D WITH_1394=ON \
      -D WITH_FFMPEG=ON \
      -D WITH_GPHOTO2=ON \
      -D WITH_OPENGL=ON \
      -D WITH_QT=ON \
      -D WITH_TBB=OFF \
      -D WITH_WEBP=ON \
      -D WITH_UNICAP=ON \
      -D WITH_OPENNI=OFF \
      -D WITH_GDAL=OFF \
      -D WITH_CUBLAS=OFF \
      -D WITH_NVCUVID=OFF \
      -D WITH_CUDA=OFF \
      -D WITH_CUFFT=OFF \
      -D WITH_IPP=OFF \
      -D WITH_IPP_A=OFF \
      -D WITH_OPENMP=ON \
      -D WITH_PTHREADS_PF=OFF \
      -D WITH_PVAPI=OFF \
      -D WITH_MATLAB=OFF \
      -D WITH_XIMEA=OFF \
      -D WITH_XINE=OFF \
      -D WITH_OPENCL=ON \
      -D WITH_OPENCLAMDBLAS=OFF \
      -D WITH_OPENCLAMDFFT=OFF \
      -D WITH_OPENCL_SVM=OFF \
      -D INSTALL_PYTHON_EXAMPLES=OFF \
      -D ENABLE_CXX11=ON \
      -D ENABLE_CCACHE=ON \
      -D ENABLE_FAST_MATH=ON \
      -D ENABLE_NEON=ON \
      -D BUILD_opencv_apps=ON \
      -D BUILD_opencv_aruco=ON \
      -D BUILD_opencv_bgsegm=ON \
      -D BUILD_opencv_calib3d=ON \
      -D BUILD_opencv_bioinspired=ON \
      -D BUILD_opencv_dnn=ON \
      -D BUILD_opencv_dpm=ON \
      -D BUILD_opencv_core=ON \
      -D BUILD_opencv_face=ON \
      -D BUILD_opencv_features2d=ON \
      -D BUILD_opencv_flann=ON \
      -D BUILD_opencv_freetype=ON \
      -D BUILD_opencv_fuzzy=ON \
      -D BUILD_opencv_hfs=ON \
      -D BUILD_opencv_highgui=ON \
      -D BUILD_opencv_imgcodecs=ON \
      -D BUILD_opencv_imgproc=ON \
      -D BUILD_opencv_ml=ON \
      -D BUILD_opencv_objdetect=ON \
      -D BUILD_opencv_optflow=ON \
      -D BUILD_opencv_phase_unwrapping=ON \
      -D BUILD_opencv_photo=ON \
      -D BUILD_opencv_plot=ON \
      -D BUILD_opencv_python3=ON \
      -D BUILD_opencv_python2=OFF \
      -D BUILD_opencv_reg=ON \
      -D BUILD_opencv_rgbd=ON \
      -D BUILD_opencv_saliency=ON \
      -D BUILD_opencv_shape=ON \
      -D BUILD_opencv_stereo=ON \
      -D BUILD_opencv_stitching=ON \
      -D BUILD_opencv_superres=ON \
      -D BUILD_opencv_surface_matching=ON \
      -D BUILD_opencv_text=ON \
      -D BUILD_opencv_tracking=ON \
      -D BUILD_opencv_ts=ON \
      -D BUILD_opencv_video=ON \
      -D BUILD_opencv_videoio=ON \
      -D BUILD_opencv_videostab=ON \
      -D BUILD_opencv_viz=OFF \
      -D BUILD_opencv_world=OFF \
      -D BUILD_opencv_xfeature2d=ON \
      -D BUILD_opencv_ximgproc=ON \
      -D BUILD_opencv_xobjdetect=ON \
      -D BUILD_opencv_xphoto=ON .. 
      
      make -j4 && sudo make install & sudo ldconfig
