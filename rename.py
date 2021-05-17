# Pythono3 code to rename multiple
# files in a directory or folder

# importing os module
import os

os.chdir('D://TFODCourse//HGM-4//HGM-1.0//Right_CAM//Z')
print(os.getcwd())

for f in os.listdir():
    f_name, f_ext = os.path.splitext(f)
    f_name = "Z_Right" +f_name

    new_name = '{} {}'.format(f_name, f_ext)
    os.rename(f, new_name)