import os
import pathlib
import shutil

cwd = os.getcwd()

fp = open('valid.txt', 'r')
for line in fp.readlines():
    line = line.strip()
    dest = 'img_for_valid\\' + line.replace(cwd, '')
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(dest)
    shutil.copyfile(line, dest)