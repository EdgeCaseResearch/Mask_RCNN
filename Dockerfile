FROM tensorflow/tensorflow:latest-gpu

# Install ROS Kinetic
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu xenial main" > /etc/apt/sources.list.d/ros-latest.list' \
    && apt-key adv --keyserver hkp://ha.pool.sks-keyservers.net:80 --recv-key 421C365BD9FF1F717815A3895523BAEEB01FA116 \
    && apt-get update && apt-get install -y \
    ros-kinetic-ros-base \
    ros-kinetic-image-transport \
    ros-kinetic-cv-bridge

RUN rosdep init && rosdep update

RUN echo "source /opt/ros/kinetic/setup.sh" >> ~/.bashrc

RUN apt-get update && apt-get install -y \
    less \
    vim \
    wget \
	python3-pip \
	python3-tk

# Run them like this so docker can cache the results
RUN pip3 install numpy jupyter scikit-image scipy Pillow cython h5py tensorflow tensorflow-gpu keras opencv-python tqdm rospkg catkin_pkg

# Requires cython is installed first:
RUN pip3 install pycocotools

WORKDIR /workspace

# Copy over the Mask_RCNN code
COPY . .

# # Build the ROS messages
WORKDIR /workspace/catkin_ws
RUN rm src/CMakeLists.txt
RUN rm -rf build
RUN rm -rf devel

WORKDIR /workspace/catkin_ws/src
RUN /bin/bash -c "source /opt/ros/kinetic/setup.bash && catkin_init_workspace"

WORKDIR /workspace/catkin_ws
RUN /bin/bash -c "source /opt/ros/kinetic/setup.bash && catkin_make"
RUN echo "source $(pwd)/devel/setup.sh" >> ~/.bashrc

# Source the CUDA requirements
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-9.0/lib64
ENV PATH=${PATH}:/usr/local/cuda-9.0/bin

WORKDIR /workspace

# Run the ros interface by default
CMD /bin/bash -c "source /opt/ros/kinetic/setup.bash && source catkin_ws/devel/setup.bash && python3 run_ros_interface.py"

