FROM python:3.10
WORKDIR /app

ENV DEBIAN_FRONTEND noninteractive

ENV MVIMPACT_ACQUIRE_DIR /opt/mvIMPACT_Acquire
ENV MVIMPACT_ACQUIRE_DATA_DIR /opt/mvIMPACT_Acquire/data

RUN apt-get update && apt-get install -y \
    build-essential cmake --no-install-recommends \
    libgtk2.0-dev pkg-config \
		wget vim \
		libncurses5-dev \
        libcanberra-gtk-module

# numpy needs to be installed before for opencv bindings to be generated correctly
RUN pip install numpy

RUN mkdir -p /opencv && cd /opencv && \
    wget -O opencv.zip https://github.com/opencv/opencv/archive/4.x.zip && \
    unzip opencv.zip && \
    mkdir -p build && cd build && \ 
    cmake \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D BUILD_TESTS=OFF\
    -D BUILD_PERF_TESTS=OFF \
    -D BUILD_EXAMPLES=OFF \
    -D BUILD_opencv_apps=OFF \
    ../opencv-4.x | tee cmake.log && \
    make -j"$(nproc)" && \
    make install && ldconfig

COPY ./requirements.txt /app
RUN pip install -r ./requirements.txt 

# Setup mvBlueFox3 camera
COPY ./cam/mvImpactAquire /var/lib/mvIMPACT_Acquire

RUN cd /var/lib/mvIMPACT_Acquire # && \
    ./install_mvGenTL_Acquire.sh -m && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
