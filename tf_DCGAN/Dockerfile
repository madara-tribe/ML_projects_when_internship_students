# CPU
FROM ubuntu:16.04

LABEL maintainer "example@example.jp"

RUN apt-get update && apt-get upgrade
RUN apt-get -y install python3-pip curl
# SELECT CPU or GPU
# RUN pip3 install keras tensorflow-gpu jupyter opencv-python
RUN pip3 install keras tensorflow jupyter

RUN pip3 install --upgrade pip
RUN apt-get -y install python3.5 python3.5-dev
RUN apt-get -y install python3-numpy python3-scipy python3-matplotlib
RUN apt-get -y install libcupti-dev # NVIDIA CUDA Profiling
RUN pip3 install opencv-python
RUN mkdir /home/cnn_dir /home/dcgan_dir
ADD manshion_train.tfrecords /home
ADD manshion_test.tfrecords /home
ADD gakan_train128.pickle /home
ENTRYPOINT ["bash"]
