# Project Title

Camera Extrinsic Parameter


## Index
### Target-based
  - [Dependancy](#Dependancy) 
  - [Install](#Install)
  - [Parameter](#Parameter)
  - [Demo](#Demo)
  - [Results](#Results)

---
## Target-based

### Dependancy

- ROS melodic ver
- Opencv 3.4 이상
- c++17

## Install

- Docker 사용 : Nvidia docker Image 설치필요

```
$ sudo apt-get install x11-xserver-utils
$ xhost +

$ docker pull authorsoo/px4:9.0
$ docker run --gpus all -it --ipc=host  --expose 22 --net=host --privileged -e DISPLAY=unix$DISPLAY 
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw -e NVIDIA_DRIVER_CAPABILITIES=all --name calib authorsoo/px4:9.0 bash
```

## Parameter
rosrun 이 아닌 launch 파일로 실행시길 경우 직접 파라미터 세팅
![launchfile](https://user-images.githubusercontent.com/44966311/168544455-4b78a416-0f66-44b5-bb10-503b9b8b13f9.png))
```
$ roslaunch cam_lidar_calib cam_lidar_calib_basler_VLP
```
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

