FROM python:3.10
WORKDIR /app

ARG OPENCV_VERSION=4.7.0 

# Install basic utility, C++ tools and OpenCV dependencies
RUN apt-get update && apt-get install -y \
    wget pkg-config vim \
	build-essential cmake --no-install-recommends \
	libgtk2.0-dev libncurses5-dev libcanberra-gtk-module 
	

# Numpy needs to be installed before opencv so Python bindings can be generated
RUN pip install numpy

# Get, build and install OpenCV main modules (may take some time)
RUN mkdir -p /opencv && cd /opencv && \
    wget -O opencv.zip https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip && \
    unzip opencv.zip && \
    mkdir -p build && cd build && \ 
    cmake \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D BUILD_TESTS=OFF\
    -D BUILD_PERF_TESTS=OFF \
    -D BUILD_EXAMPLES=OFF \
    -D BUILD_opencv_apps=OFF \
    ../opencv-${OPENCV_VERSION} | tee cmake.log && \
    make -j"$(nproc)" && \
    make install && ldconfig

# Install additional Python packages
COPY ./requirements.txt /app
RUN pip install -r ./requirements.txt 