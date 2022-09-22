# Target based calibration

## Overall

- Camera Extrinsic Parameter

## Index

- [Dependancy](#Dependancy)
- [Install](#Install)
- [Parameter](#Parameter)
- [Demo](#Demo)
- [Results](#Results)

---

## Target-based

### Dependancy

- 1.1 ROS and Ubuntu
  Our code has been tested on Ubuntu 16.04 with ROS Kinetic and Ubuntu 18.04 with ROS Melodic.

- 1.2 Ceres Solver
  Our code has been tested on ceres solver 1.14.x.

- 1.3 OpenCV
  Our code has been tested on OpenCV 3.4.14.

- 1.4 Eigen
  Our code has been tested on Eigen 3.3.7.

- 1.5 PCL
  Our code has been tested on PCL 1.8.

- 1.6 Open3D
  Our code has been tested on Open3D 0.7.

## Install

### 1. Env

```
(in workspace folder)
$ git clone <tagetbased:branch>
$ git clone <open3d-msgs:branch>

$ docker calib_ext_targetbased/docker -t <name>

$ docker run --gpus all -it --ipc=host  --expose 22 --net=host --privileged \
  -e DISPLAY=unix$DISPLAY  -v /tmp/.X11-unix:/tmp/.X11-unix:rw -e \
  NVIDIA_DRIVER_CAPABILITIES=all --name <container name> <img name> bash
```

### 2. launch Lidar-Lidar

- Step 1: base LiDAR pose optimization (the initial pose is stored in scene-x/original_pose)

```
roslaunch mlcc pose_refine.launch
```

- Step 2: LiDAR extrinsic optimization (the initial extrinsic is stored in config/init_extrinsic)

```
roslaunch mlcc extrinsic_refine.launch
```

- Step 3: pose and extrinsic joint optimization

```
roslaunch mlcc global_refine.launch
```

### 3 LiADR-Camera Extrinsic Calibration

```
roslaunch mlcc calib_camera.launch
```

## Parameter

파라미터 세팅

\*\* todo
![launchfile](https://user-images.githubusercontent.com/44966311/168544455-4b78a416-0f66-44b5-bb10-503b9b8b13f9.png))

## Demo

1. Dataset 이용
   [Dataset](https://drive.google.com/a/tamu.edu/file/d/19Ke-oOhqkPKJBACmrfba4R5-w71_wrvT/view?usp=sharing)

도커 이용시 /home 디렉토리에 파일 존재함

```
$ rosrun mono_cam_calib mono_cam_calib
$ rosbag play [rosbag 파일명] // rosbag실행
```

[Demo Video](https://www.notion.so/Paper-Review-Study-a59fb867065248fbb20b4a5840a5f661)

## Results

result폴더에 저장

## Ref

---
