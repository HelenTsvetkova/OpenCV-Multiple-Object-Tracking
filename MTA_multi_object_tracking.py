# import the necessary packages
from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2
import pandas as pd
import numpy as np

def getFirstLastFrame(inputFile, objectId):
    tmpInputFile = inputFile.copy()
    tmpInputFile = tmpInputFile[(tmpInputFile["objectId"] == objectId)]
    lastFrame = max(tmpInputFile['frameNumber'])
    firstFrame = min(tmpInputFile['frameNumber'])
    return firstFrame, lastFrame

def checkBbox(videoWidth, videoHeight, box):
    (x, y, w, h) = [int(v) for v in box]
    if(x < 0):
        x = 0
    if(y < 0):
        y = 0
    if(x > videoWidth):
        x = videoWidth
    if(y > videoHeight):
        y = videoHeight
    if((x + w) > videoWidth):
        w = videoWidth - x
    if((y + h) > videoHeight):
        h = videoHeight - y
    return (x, y, w, h)


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--tracker", type=str, default="kcf",
    help="OpenCV object tracker type")
ap.add_argument("--root-dir", type=str, default="/home/helen/DataSets/MTA/MTA_ext_short/test/",
    help="Root directory with cameras reid info directories.")
ap.add_argument("-id", "--camera-id", type=int, default=0,
    help="camera id (num from 0 to 5")

args = vars(ap.parse_args())

# initialize a dictionary that maps strings to their corresponding
# OpenCV object tracker implementations
OPENCV_OBJECT_TRACKERS = {
    "boosting": cv2.TrackerBoosting_create, 
    "mil": cv2.TrackerMIL_create, 
    # "tld": cv2.TrackerTLD_create, 
    # "medianflow": cv2.TrackerMedianFlow_create, 
    "mosse": cv2.TrackerMOSSE_create,
    "csrt": cv2.TrackerCSRT_create, 
    "kcf": cv2.TrackerKCF_create,  
}

# input params
rootDir = args["root_dir"]
camIdx = args["camera_id"]
tracker_type = args["tracker"]

# main loop per cameras, tracker types and frames
for id in [4]:

    for tracker_type in ['mil']:#OPENCV_OBJECT_TRACKERS.keys():

        camIdx = id
        videoName = "cam_" + str(camIdx)

        # load ground truth file
        gtFileName = "coords_fib_" + videoName + ".txt"
        gtFilePath = rootDir + videoName + "/" + gtFileName
        gtInfo = pd.read_csv(gtFilePath, header=None)
        # rename the columns so we know what are we working with
        gtInfo['frameNumber'] = gtInfo[0]
        gtInfo['objectId'] = gtInfo[1]
        gtInfo['x'] = gtInfo[2]
        gtInfo['y'] = gtInfo[3]
        gtInfo['w'] = gtInfo[4]
        gtInfo['h'] = gtInfo[5]

        # save results file
        f = open("ResultsTMP2/" + videoName + "_" + tracker_type + ".txt", "w")

        # load input video
        videoPath = rootDir + videoName + "/" + videoName + ".mp4"
        vs = cv2.VideoCapture(videoPath)
        videoWidth  = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))   
        videoHeight = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
        frame = vs.read() 

        # number of objects currently tracked
        numTrackObjects = 0
        # threshold of how many objects to track
        maxObjects = 50
        # num frames
        frameIdx = -1
        # num max frames
        frameIdxMax = int(max(gtInfo['frameNumber']) / 10)
        # initialized (first appearance) of objects - bool
        initializedObjects = [False] * (max(gtInfo['objectId']) + 1) # max() is the number of objects that will be tracked in this video
        # labels (ids) of objects
        idOfObject = []
        labels = []
        #tracking results
        boxes = ()
        success = 0
        # initialize OpenCV's special multi-object tracker
        trackers = cv2.MultiTracker_create()
        trackers.clear()

        # loop over frames from the video stream
        while True:
            # grab the current frame, then handle if we are using a
            # VideoStream or VideoCapture object
            frameIdx += 1
            frame = vs.read()
            frame = frame[1] #if args.get("video", False) else frame
            
            # check to see if we have reached the end of the stream
            if frame is None:
                break
            
            # USE ONLY Half of data
            if frameIdx==frameIdxMax:
                print("Use only 1/10 of data. End of progressing " + videoPath)
                break
            
            # grab the updated bounding box coordinates (if any) for each
            # object that is being tracked
            boxes = ()
            (success, boxes) = trackers.update(frame)
            
            # loop over the bounding boxes and draw then on the frame
            objCnt = 0
            for box in boxes:
                (x, y, w, h) = [int(v) for v in box]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                lbl = str(idOfObject[objCnt])
                cv2.putText(frame, lbl, (x,y),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                f.write(str(frameIdx) + "," + str(idOfObject[objCnt]) + "," + str(x) + "," + str(y) + "," + str(w) + "," + str(h) + "\n")
                objCnt += 1

            # show the output frame
            print("Frame: " + str(frameIdx) + "; cam " + str(camIdx) + "; currently tracking " + str(numTrackObjects) + " objects with " + tracker_type)
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

            # check how many objects are currently tracked
            if (numTrackObjects + 1 <= maxObjects):    
                # get all objects that are seen in the current frameNumber 
                listOfObjectsInCurrentFrame = gtInfo.loc[gtInfo['frameNumber'] == frameIdx]
                # get all objects that are not already initialized in current frameNumber
                for x in listOfObjectsInCurrentFrame['objectId']: 
                    firstFrame, lastFrame = getFirstLastFrame(gtInfo, x)
                    if firstFrame + 30 > frameIdx:
                        continue
                    if initializedObjects[x] == False: 
                        row = listOfObjectsInCurrentFrame.loc[listOfObjectsInCurrentFrame['objectId'] == x]
                        box = (row['x'], row['y'], row['w'], row['h']) 
                        box = checkBbox(videoWidth, videoHeight, box)
                        try:
                            if numTrackObjects + 1 <= maxObjects:
                                # tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
                                tracker = OPENCV_OBJECT_TRACKERS[tracker_type]()
                                trackers.add(tracker, frame, box)
                                labels.append(str(x))
                                numTrackObjects += 1 # increase the number of tracked objects
                                # set the object to be initialized - ignore all future occurences
                                initializedObjects[x] = True
                                # initialize the tracker for the object
                                idOfObject.append(x)
                                obj = trackers.getObjects()
                                print("Start tracking object with id :" + str(x) + " (the " + str(numTrackObjects) + "th object)")
                        except: 
                            print("Error while initializing the tracker")
                            idOfObject.append(987654) # faulty box
        
            # check every object to be removed (it's last frame)
            objCnt = 0
            for x in listOfObjectsInCurrentFrame['objectId']:
                if initializedObjects[x] == False:
                    continue
                firstFrame, lastFrame = getFirstLastFrame(gtInfo, x)
                if lastFrame == frameIdx:
                    print("Removing object with id :" + str(x) + " (the " + str(objCnt) + "th object)")
                    trackers_arr = np.array(trackers.getObjects())
                    idx = np.where(trackers_arr.sum(axis=1)!=0)[0]
                    trackers_arr=trackers_arr[idx]
                    trackers = cv2.MultiTracker_create()
                    for idx, box in enumerate(trackers_arr):
                        box = checkBbox(videoWidth, videoHeight, box)
                        if(idx == objCnt):
                            continue

                        tracker = OPENCV_OBJECT_TRACKERS[tracker_type]()
                        try:
                            trackers.add(tracker, frame, tuple(box))
                        except:
                            print("Error while initializing the tracker")
                            label = idOfObject[idx]
                            idOfObject.remove(label)
                            initializedObjects[label] = False
                            numTrackObjects = numTrackObjects - 1
                    
                    initializedObjects[x] = False
                    idOfObject.remove(x)      
                    numTrackObjects = numTrackObjects - 1

                objCnt += 1

                        
        vs.release()
        # close all windows
        cv2.destroyAllWindows()
        f.close()