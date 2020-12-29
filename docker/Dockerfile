FROM ros:melodic

ARG user_uid
ARG user_name
ARG user_gid
ARG user_group


# Install dependences
RUN \
  apt update && \
  apt install -y \
    bash-completion \
    htop \
    nano \
    python-pandas \
    python3-pandas \
    python-yaml \
    python3-yaml \
    python-termcolor \
    python3-termcolor \
    python-psutil \
    python3-psutil \
    python-pip \
    python3-pip \
    python-skimage \
    python3-skimage \
    python-sklearn \
    python3-sklearn \
    python-catkin-tools \
    ros-melodic-catkin \
    ros-melodic-map-server \
    ros-melodic-move-base \
    ros-melodic-global-planner \
    ros-melodic-dwa-local-planner \
    ros-melodic-filters \
    ros-melodic-sbpl \
    ros-melodic-sbpl-dbgsym \
    ros-melodic-sbpl-recovery \
    ros-melodic-sbpl-recovery-dbgsym \
    ros-melodic-ompl \
    ros-melodic-ompl-dbgsym \
    ros-melodic-turtlebot3 \
    ros-melodic-tf-conversions \
    ros-melodic-tf2-sensor-msgs \
    && apt upgrade -y \
    && rm -rf /var/lib/apt/lists/*


# copy enrtypoint
COPY ros_entrypoint.sh /ros_entrypoint.sh


# switch to user
RUN \
  groupadd --gid $user_gid $user_group && \
  useradd --uid $user_uid --gid $user_group --shell /bin/bash --create-home $user_name
RUN adduser $user_name sudo
RUN echo "$user_name ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers
USER $user_name:$user_group


# clone repos
RUN mkdir -p ~/w/catkin_ws/src
RUN \
  cd ~/w/catkin_ws/src && \
  git clone --branch=master https://github.com/AIRLab-POLIMI/performance_modelling.git && \
  git clone --branch=melodic-devel https://github.com/fcelitok/global_planning_performance_modelling.git && \
  git clone --branch=master https://github.com/fcelitok/move_base_ompl.git && \
  git clone --branch=melodic-devel https://github.com/fcelitok/navigation_experimental.git


# build workspace and install python user packages
RUN \
  . /opt/ros/melodic/setup.sh && \
  cd ~/w/catkin_ws && \
  catkin build -j 6

RUN \
  pip install pyquaternion && \
  pip3 install pyquaternion && \
  pip install networkx && \
  pip3 install networkx && \
  pip install scikit-image && \
  pip3 install scikit-image

RUN \
  cd ~/w/catkin_ws/src/performance_modelling/ && \
  ./install_dev.sh

RUN mkdir -p ~/ds/performance_modelling
RUN mkdir -p ~/.ros/log

# copy bashrc files
COPY .bashrc /home/$user_name/.bashrc
RUN sudo chown $user_name:$user_name /home/$user_name/.bashrc
COPY .bash_aliases /home/$user_name/.bash_aliases
RUN sudo chown $user_name:$user_name /home/$user_name/.bash_aliases
