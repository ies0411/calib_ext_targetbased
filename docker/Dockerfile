# docker build -t mmdetection3d -f docker/Dockerfile .

ARG ROS_VERSION="melodic"
FROM ros:${ROS_VERSION}
#TODO : pcl
#package install
RUN apt-get update && apt-get install -y libc++-10-dev libc++abi-10-dev curl sudo wget build-essential ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgoogle-glog-dev libatlas-base-dev libeigen3-dev libsuitesparse-dev\
  && apt-get update;\
  apt-get clean;\
  rm -rf /var/lib/apt/lists/*;
#ROS install
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list';\
  curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -;\
  apt-get update;\
  apt-get install -y ros-melodic-desktop-full libpcl-dev ros-melodic-pcl-* ros-melodic-cv-bridge-*;echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc;\
  /bin/bash -c 'source ~/.bashrc';apt-get install -y python-rosdep python-rosinstall python-rosinstall-generator python-wstool build-essential;\
  sudo apt install python3-pip -y\
  apt-get install -y python-rosdep;rosdep init;rosdep update;

#CMAKE update
RUN cd /root;wget https://github.com/Kitware/CMake/releases/download/v3.23.3/cmake-3.23.3.tar.gz;\
  tar -xvf cmake-3.23.3.tar.gz;cd cmake-3.23.3;./bootstrap;make -j$(nproc);make install;

#CERES
RUN /bin/bash -c 'source ~/.bashrc';cd /root;git clone https://ceres-solver.googlesource.com/ceres-solver;mkdir ceres-bin;cd ceres-bin;cmake ../ceres-solver;make -j$(nproc);make test;sudo make install;

#OPEN 3D
RUN cd /root;git clone --recursive https://github.com/intel-isl/Open3D;cd Open3D;\
  /bin/bash -c 'source util/scripts/install_deps_ubuntu.sh';\
  mkdir build && cd build;\
  apt-get install -y xorg-dev libglu1-mesa-dev;\
  cmake -DBUILD_EIGEN3=ON -DBUILD_GLEW=ON -DBUILD_GLFW=ON -DBUILD_JSONCPP=ON -DBUILD_PNG=ON -DGLIBCXX_USE_CXX11_ABI=ON -DPYTHON_EXECUTABLE=/usr/bin/python -DBUILD_UNIT_TESTS=ON ..;\
  make -j$(nproc);\
  make install;

#fmt
RUN cd /root;git clone https://github.com/fmtlib/fmt.git;cd fmt;mkdir build;cd build;cmake ..;make -j$(nproc);make install;

# ext calib
COPY . /workspace/src/calib_targetbased
WORKDIR /workspace/src
RUN git clone https://github.com/ntnu-arl/open3d_ros.git


RUN /bin/bash -c 'source /opt/ros/melodic/setup.bash &&\
  cd /workspace &&\
  catkin_make && source ~/.bashrc'

