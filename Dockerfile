FROM nvidia/cuda:12.0.1-devel-ubuntu22.04
WORKDIR /app


# no prompts during ubuntu setup
ENV DEBIAN_FRONTEND=noninteractive 

# latest stable release
ARG OPENCV_VERSION=4.x 

RUN apt-get update && apt-get install -y \
    build-essential cmake --no-install-recommends gdb \
	wget unzip pkg-config vim \
	python3.10 python3-pip python3-dev python3-tk \
	libgtk2.0-dev libncurses5-dev 


# Install Python3.10 as default version
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 

# numpy needs to be installed before for opencv bindings to be generated
RUN pip install numpy

# Get, Build and Install OpenCV with CUDA support
RUN mkdir -p /opt/opencv && cd /opt/opencv && \
    wget -O opencv.zip https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip && unzip opencv.zip && \
	wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.zip && unzip opencv_contrib.zip && \
    rm opencv.zip opencv_contrib.zip && \
    mkdir -p build && cd build && \ 
    cmake \
	-D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
	-D PYTHON_EXECUTABLE=$(which python3) \
    -D PYTHON3_PACKAGES_PATH=/usr/lib/python3/dist-packages \
    -D OPENCV_EXTRA_MODULES_PATH=../opencv_contrib-${OPENCV_VERSION}/modules \
    -D WITH_CUDA=ON \
    -D BUILD_TESTS=OFF\
    -D BUILD_PERF_TESTS=OFF \
    -D BUILD_EXAMPLES=OFF \
    -D BUILD_opencv_apps=OFF \
    ../opencv-${OPENCV_VERSION} | tee cmake.log && \
    make -j"$(nproc)" && \
    make install && ldconfig

COPY ./requirements.txt /app
RUN pip install -r ./requirements.txt 