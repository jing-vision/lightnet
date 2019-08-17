import glob
import os
import csv
import pathlib
import shutil

folders = glob.glob('v3*')
print(folders)

# setup mapping
SKU2_to_folders = {}
for folder in folders:
    with open(folder + '/obj.names') as fp:
        names = fp.readlines()
        for name in names:
            sku = name[4:-1]
            SKU2_to_folders[sku] = folder


# copy images
with open('2019-Aug-17-12-58-50-994811.csv') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    next(csv_reader)

    for row in csv_reader:
        sku = row['SKU']
        img_name = row['image']
        if sku in SKU2_to_folders:
            sub = os.path.join(SKU2_to_folders[sku], 'img', sku)
            os.makedirs(sub, exist_ok=True)
            shutil.copyfile(img_name, os.path.join(sub, img_name.replace('socket_debug/', '')))



