import cv2
from cv2 import sort
import numpy as np
import glob
import os

# partName = "train" 
# numVideo = 5 # 1, 2, 3, 5

partName = "test" 
numVideo = 8 # 4, 6, 7, 8

dirInputName = "/home/helen/DataSets/MOT20/" + partName + "/MOT20-0" + str(numVideo) + "/"
dirOutName = os.path.join(dirInputName, "video/")
if not os.path.exists(dirOutName):
    os.mkdir(dirOutName)

fileOutName = "mot20-0" + str(numVideo) + ".avi"
vidOutPath = dirOutName + fileOutName

fileNames = glob.glob(dirInputName + 'img1/*.jpg')
fileNames.sort()

frameSize = cv2.imread(fileNames[0]).shape[1::-1]
out = cv2.VideoWriter(vidOutPath, cv2.VideoWriter_fourcc(*'DIVX'), 25, frameSize)

for filename in fileNames:
    print(filename)
    img = cv2.imread(filename)
    out.write(img)

out.release()