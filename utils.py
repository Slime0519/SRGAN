import os
import glob
import re
import torch

def load_module(dirpath):
    filelist = glob.glob(os.path.join(dirpath,"*.pth"))

    listlen = len(filelist)
    sorted_filelist = [0 for _ in range(listlen)]
    for filename in filelist:
        epoch = int(*re.findall("\d+",filename))
      #  print(epoch)
        sorted_filelist[epoch] = filename

    return listlen, sorted_filelist[-1]



