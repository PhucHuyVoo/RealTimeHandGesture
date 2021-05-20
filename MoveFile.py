import shutil
import os
import glob

source_below = 'C:/Users/Admin/Downloads/jzy8zngkbg-8/HGM-4/HGM-1.0/Below_CAM'
source_left = 'C:/Users/Admin/Downloads/jzy8zngkbg-8/HGM-4/HGM-1.0/Left_CAM'
source_right = 'C:/Users/Admin/Downloads/jzy8zngkbg-8/HGM-4/HGM-1.0/Right_CAM'
source_front = 'C:/Users/Admin/Downloads/jzy8zngkbg-8/HGM-4/HGM-1.0/Front_CAM'

dest_test = 'D:/TFODCourse/Tensorflow/workspace/images/test'
dest_train = 'D:/TFODCourse/Tensorflow/workspace/images/train'

name = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
for i in range(26):
    os.chdir(os.path.join(source_below,name[i]))
    files_below = os.listdir()
    for file in files_below:
        shutil.copy(f"{os.path.join(source_below,name[i])}/{file}", dest_train)
for i in range(26):
    os.chdir(os.path.join(source_left, name[i]))
    files_left = os.listdir()
    for file in files_left:
        shutil.copy(f"{os.path.join(source_left, name[i])}/{file}", dest_train)
for i in range(26):
    os.chdir(os.path.join(source_right, name[i]))
    files_right = os.listdir()
    for file in files_right:
        shutil.copy(f"{os.path.join(source_right, name[i])}/{file}", dest_train)
for i in range(26):
    os.chdir(os.path.join(source_front, name[i]))
    files_front = os.listdir()
    for file in files_front:
        shutil.copy(f"{os.path.join(source_front, name[i])}/{file}", dest_test)


