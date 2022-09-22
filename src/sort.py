import json
import numpy as np
# import quaternion
import math
import os
import cv2
# import open3d as o3d
import natsort
import shutil


img_path = '/workspace/data/ratectn/img/'
pcd_path = '/workspace/data/ratectn/pcd/'

img_path_list = os.listdir(img_path)
print(len(img_path_list))
img_path_list = natsort.natsorted(img_path_list)

pcd_path_list = os.listdir(pcd_path)
print(len(pcd_path_list))
pcd_path_list = natsort.natsorted(pcd_path_list)


for idx,file in enumerate(img_path_list):
  src = os.path.join(img_path,file)
  dst = str(idx) + '.png'
  dst = os.path.join(img_path, dst)
  os.rename(src,dst)

for idx,file in enumerate(pcd_path_list):
  src = os.path.join(pcd_path,file)
  dst = str(idx) + '.pcd'
  dst = os.path.join(pcd_path, dst)
  os.rename(src,dst)