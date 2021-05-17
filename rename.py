# Pythono3 code to rename multiple
# files in a directory or folder

# importing os module
import os

os.chdir('') #Your dataset's path
print(os.getcwd())

for f in os.listdir():
    f_name, f_ext = os.path.splitext(f)
    #f_name = ... :New file name

    new_name = '{} {}'.format(f_name, f_ext)
    os.rename(f, new_name)