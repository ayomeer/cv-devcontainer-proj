FROM nvidia/cuda:12.1.0-devel-ubuntu22.04
WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive 
ENV NVIDIA_VISIBLE_DEVICES=all # equivalent to '--gpus all' in Docker run command 
ARG OPENCV_VERSION=4.x

RUN apt-get update && apt-get install -y \
    build-essential cmake --no-install-recommends \
    libgtk2.0-dev pkg-config \
	wget unzip vim \
	python3.10 \
	python3-pip \
	python3-dev \
	libncurses5-dev && \
	apt-get purge --auto-remove && apt-get clean 
    

# Install Python3.10 as default version
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 

# numpy needs to be installed before for opencv bindings to be generated correctly
RUN pip install numpy

# Get, Build and Install OpenCV (w/o contrib)
RUN mkdir -p /opt/opencv && cd /opt/opencv && \
	wget -O opencv.zip https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip && \
	unzip opencv.zip && rm opencv.zip && \
	mkdir -p build && cd build && \ 
	cmake \
	-D CMAKE_BUILD_TYPE=Release \
	-D CMAKE_INSTALL_PREFIX=/usr/local \
#	-D PYTHON3_PACKAGES_PATH=/usr/lib/python3/dist-packages \
	-D BUILD_LIST=core, highgui \
	-D BUILD_TESTS=OFF\
	-D BUILD_PERF_TESTS=OFF \
	-D BUILD_EXAMPLES=OFF \
	-D BUILD_opencv_apps=OFF \
	../opencv-${OPENCV_VERSION} && \
   make -j"$(nproc)" && \
   make install && ldconfig

COPY ./requirements.txt /app
RUN pip install -r ./requirements.txt 
	
