# Pythono3 code to rename multiple
# files in a directory or folder

# importing os module
import os


path = 'Below_CAM'
name = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

for i in range(26):
    os.chdir(os.path.join(path,name[i]))
    #print(os.getcwd())
    for f in os.listdir():
        f_name, f_ext = os.path.splitext(f)
        #f_name = ... :New file name
        f_name = '{}_Below'.format(name[i])+f_name
        new_name = '{} {}'.format(f_name, f_ext)
        os.rename(f, new_name)