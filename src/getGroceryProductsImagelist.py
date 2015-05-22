"""
This script extracts a list of pre-cropped grocery images from Larry's grocery set directory
"""
import os
import sys

def maclistdir(fpath):
    filelist=os.listdir(fpath)
    return [x for x in filelist if not (x.startswith('.'))]

if __name__ == '__main__':
    rootImgDir='/media/cuda/data/iqnect-set/Larry/Grocery_products/Testing/'
    ImgDirs=maclistdir(rootImgDir)
    cropImgDirs=[]
    for dirs in ImgDirs:
        cropImgDirs.append(rootImgDir+dirs+'/crops/')

    imgFilelist=[]
    for dirs in cropImgDirs:
        imgFilelist.append([dirs+x for x in maclistdir(dirs)])

    for dirs in imgFilelist:
        for imgs in dirs:
            print imgs


