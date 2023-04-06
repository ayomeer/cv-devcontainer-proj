FROM ubuntu:20.04
WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential cmake --no-install-recommends \
    libgtk2.0-dev pkg-config \
		wget vim \
		libncurses5-dev

# numpy needs to be installed before for opencv bindings to be generated correctly
RUN pip install numpy

# RUN mkdir -p /opencv && cd /opencv && \
#     wget -O opencv.zip https://github.com/opencv/opencv/archive/4.x.zip && \
#     unzip opencv.zip && \
#     mkdir -p build && cd build && \ 
#     cmake \
#     -D CMAKE_BUILD_TYPE=Release \
#     -D CMAKE_INSTALL_PREFIX=/usr/local \
#     ../opencv-4.x && \
#    make -j"$(nproc)" && \
#    make install && ldconfig

# Install CUDA and dependencies
RUN apt-get install software-properties-common && \
	wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda-repo-debian11-12-1-local_12.1.0-530.30.02-1_amd64.deb && \
	sudo dpkg -i cuda-repo-debian11-12-1-local_12.1.0-530.30.02-1_amd64.deb && \
	sudo cp /var/cuda-repo-debian11-12-1-local/cuda-*-keyring.gpg /usr/share/keyrings/ && \
	sudo add-apt-repository contrib && \
	sudo apt-get update && \
	sudo apt-get -y install cuda 

COPY ./requirements.txt /app
RUN pip install -r ./requirements.txt 
	
