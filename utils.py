import os
import glob
import re
import torch
import numpy as np

def load_module(dirpath):
    filelist = glob.glob(os.path.join(dirpath,"*.pth"))

    listlen = len(filelist)
    sorted_filelist = [0 for _ in range(listlen)]
    for filename in filelist:
        epoch = int(*re.findall("\d+",filename))
      #  print(epoch)
        sorted_filelist[epoch] = filename

    return listlen, sorted_filelist[-1]

def randomcrop(image, crop_size):
    size = image.shape
    random_x, random_y = np.random.randint(0,size[0]-crop_size), np.random.randint(0,size[1]-crop_size)
    cropped_image = image[random_x:random_x+crop_size,random_y:random_y+crop_size,:]

    return cropped_image




