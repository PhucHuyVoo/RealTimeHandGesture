import shutil
import os
import glob

source_below = 'D:/TFODCourse/HGM-4/HGM-1.0/Below_CAM/A'
source_left =  'D:/TFODCourse/HGM-4/HGM-1.0/Left_CAM/A'
source_right = 'D:/TFODCourse/HGM-4/HGM-1.0/Right_CAM/A'
source_front = 'D:/TFODCourse/HGM-4/HGM-1.0/Front_CAM/A'
dest_test = 'D:/TFODCourse/Tensorflow/workspace/images/test'
dest_train = 'D:/TFODCourse/Tensorflow/workspace/images/train'

files_below = os.listdir(source_below)
for file in files_below:
    shutil.move(f"{source_below}/{file}", dest_test)

files_left = os.listdir(source_left)
for file in files_left:
    shutil.move(f"{source_left}/{file}", dest_test)

files_right = os.listdir(source_right)
for file in files_right:
    shutil.move(f"{source_right}/{file}", dest_test)

files_front = os.listdir(source_front)
for file in files_front:
    shutil.move(f"{source_front}/{file}", dest_train)