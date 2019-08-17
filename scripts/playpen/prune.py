import glob
import os
import csv
import pathlib
import shutil

folders = glob.glob('v3*')
print(folders)

prune_target = 50

for folder in folders:
    skus = glob.glob(folder + '/img/*')
    for sku in skus:
        images = glob.glob(sku + '/*.jpg')
        img_count = len(images)
        delete_frequency = int(prune_target / img_count)
        print(folder, img_count, '->', prune_target)
        for (i, img) in enumerate(images):
            if i % delete_frequency == (delete_frequency - 1):
                os.remove(img)
