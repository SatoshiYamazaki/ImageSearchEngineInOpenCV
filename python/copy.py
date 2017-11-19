# coding:utf-8
import os
import glob
import shutil

IMAGE_DIR = "caltech101"
OUT_DIR = "caltech101_10"

# for file in os.listdir(IMAGE_DIR):
for file in glob.glob(IMAGE_DIR + "/*.jpg"):
    x = file.split("/")[-1]
    id = int(x[-8:-4])
    if id <= 10:
        print file
        shutil.copy("%s/%s" % (IMAGE_DIR, x), "%s/%s" % (OUT_DIR, x))
