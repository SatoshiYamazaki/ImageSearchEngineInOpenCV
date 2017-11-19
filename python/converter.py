#coding:utf-8
import codecs
import os
import shutil

TARGET = "101_ObjectCategories"
OUTDIR = "caltech101"

for category in os.listdir(TARGET):
    for file in os.listdir("%s/%s" % (TARGET, category)):
        image_file = "%s/%s/%s" % (TARGET, category, file)    # 101_ObjectCategories/airplanes/image_0001.jpg
        rename_file = "%s/%s_%s" % (OUTDIR, category, file)   # caltech101/airplanes_image_0001.jpg
        print "%s -> %s" % (image_file, rename_file)
        shutil.copyfile(image_file, rename_file)
