from ast import main
import pandas as pd
import math 

root_dir = '/home/helen/DataSets/MTA/MOT_MTA/'
subset = 'test/'
results_dir = 'ResultsTMP2/'
results_processed_dir = 'ResultsprocessedTMP1/'


# /////////////////////////////////////////
# /      Begin of metric functions       /
# ////////////////////////////////////////

# first function to return tracking benchmark
def center_distance(boxA, boxB):
    # determine the (x, y)-coordinates of the centers of rectangle
    centerAx = boxA[0] + boxA[2] / 2
    centerAy = boxA[1] + boxA[3] / 2
    centerBx = boxB[0] + boxB[2] / 2
    centerBy = boxB[1] + boxB[3] / 2
    xKvadrat = (centerAx - centerBx) * (centerAx - centerBx)
    yKvadrat = (centerAy - centerBy) * (centerAy - centerBy) 
    # compute the distance
    distance = math.sqrt(xKvadrat + yKvadrat)
    
 
    # return the distance between centers
    return distance

# second function to return tracking benchmark
def intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
 
    # return the intersection over union value
    return iou

# /////////////////////////////////////////
# /        End of metric functions       /
# ////////////////////////////////////////

# function that takes three arguments: video name, name of tracker and object id
# returns the CD and IoU values when comparing the tracker (trackerName) with the ground truth, on video (videoName) for a specified object id
def process(videoName, trackerName, desiredObjectid):

    # read the ground truth file for the video provided
    gtFilePath = root_dir + subset + videoName + '/gt/gt.txt'
    groundTruth = pd.read_csv(gtFilePath, ',', header=None)
    groundTruth = groundTruth.rename(columns={
        0: 'frameNumber', 
        1: 'objectID', 
        2: 'x', 
        3: 'y', 
        4: 'w', 
        5: 'h', 
        6: 'confidence', 
        7: 'typeOfObject',
        8: 'visibility'
    })

    # read the results file we got
    df = pd.read_csv(results_dir + videoName + '_' + trackerName + '.txt', ',', header=None)
    df = df.rename(columns={
        0: 'frameNumber', 
        1: 'objectID', 
        2: 'x', 
        3: 'y', 
        4: 'w', 
        5: 'h'
    })

    # which object are we analyzing?
    queryObjectID = desiredObjectid

    objectID1_res = df.loc[df['objectID'] == queryObjectID]
    objectID1_gt = groundTruth.loc[groundTruth['objectID'] == queryObjectID]

    if (len(objectID1_gt) == 0): 
        return
    # drop the first frame in gt file - the tracker is initialized with these values, so the results start from second frame
    #print(objectID1_gt)
    # objectID1_gt = objectID1_gt.drop([0], axis=0)

    # check if the lengths are the same - did the tracker detect dissapearing of the object?
    if (len(objectID1_gt) < len(objectID1_res)):
        print(trackerName + " did not detect disappearing of the object in frame " + str(len(objectID1_gt))) 

    # get the upper bound length (the minimum of two values) to compare frame by frame
    upperBound = min(len(objectID1_gt), len(objectID1_res))
    iou = 0
    cd = 0
    numOfAnalyzed = 0
    shift = 31
    for i in range(shift, upperBound): 

        frameData_gt = objectID1_gt.loc[objectID1_gt['frameNumber'] == i]
        frameData_res = objectID1_res.loc[objectID1_res['frameNumber'] == i]
        if len(frameData_gt)==0 or len(frameData_res)==0:
            continue
        
        # take the current frames bounding boxes to compute metrics IoU and CD
        boxA = (int(frameData_res['x']), int(frameData_res['y']), int(frameData_res['x']) + int(frameData_res['w']), int(frameData_res['y']) + int(frameData_res['h']))
        boxB = (int(frameData_gt['x']), int(frameData_gt['y']), int(frameData_gt['x']) + int(frameData_gt['w']), int(frameData_gt['y']) + int(frameData_gt['h']))
        
        # call the predefined metric function
        intersectionOverUnion = intersection_over_union(boxA, boxB)
        centerDistance = center_distance(boxA, boxB)
        
        if intersectionOverUnion == 0:
            continue

        # then compute the values
        cd += centerDistance 
        iou += intersectionOverUnion
        numOfAnalyzed += 1

    if (numOfAnalyzed != 0): 
        iou /= numOfAnalyzed
        cd /= numOfAnalyzed
    else: 
        print("Object with ID " + str(queryObjectID) + " was not tracked with " + trackerName + " in video " + videoName + ".")

    print("Average IoU for " + trackerName + " on video " + videoName + " for objectID " + str(queryObjectID) + " is = " + str(iou))
    print("Average CD for " + trackerName + " on video " + videoName + " for objectID " + str(queryObjectID) + " is = " + str(cd))
    return (iou, cd)

# function that takes two arguments: name of video and name of the tracker
# function returns an array of object ids tracked in that video with that tracker
def getTrackedObjectIds(videoName, trackerName):
    df = pd.read_csv(results_dir + videoName + '_' + trackerName + '.txt', ',', header=None)
    df = df.rename(columns={
        0: 'frameNumber', 
        1: 'objectID', 
        2: 'x', 
        3: 'y', 
        4: 'w', 
        5: 'h'
    })

    listOfIds = []

    df = df['objectID']
    for x in df: 
        # check if current ID is a new one
        if (x not in listOfIds) and x != 987654: 
            # double check - not to go over 50 objects
            if (len(listOfIds) + 1 <= 50):
                listOfIds.append(x)

    return listOfIds

# function that takes two arguments - list of objects tracked in a video and video name
# function outputs pair object of structure (objectID, objectType)
def getObjectTypes(listOfIds, videoName):
    returnArray = []

    groundTruth = pd.read_csv(root_dir + subset + videoName + '/gt/gt.txt', ',', header=None)
    groundTruth = groundTruth.rename(columns={
        0: 'frameNumber', 
        1: 'objectID', 
        2: 'x', 
        3: 'y', 
        4: 'w', 
        5: 'h', 
        6: 'confidence', 
        7: 'typeOfObject'
    })

    for objectID in listOfIds: 
        # find the row where objectID is located
        object_type = groundTruth.loc[groundTruth['objectID'] == objectID]['typeOfObject']
        #print(str(object_type.iloc[0]))
        returnArray.append((objectID , object_type.iloc[0]))
    
    return returnArray

if __name__ == '__main__':
    
    videoNames = ['cam_0', 'cam_2', 'cam_5', 'cam_4'] 
    trackerNames = ['csrt', 'kcf', 'boosting', 'mosse', 'mil']

    objectTypes = []
    for videoName in videoNames: 
        for trackerName in trackerNames: 
            listOfIds = getTrackedObjectIds(videoName, trackerName)
            currentObjectArray = getObjectTypes(listOfIds, videoName)
            print("Processing " + videoName + " " + trackerName)
            for x in currentObjectArray: 
                if x not in objectTypes: 
                    objectTypes.append(x)

            f = open(results_processed_dir + videoName + "_" + trackerName + ".txt", "w")
            print(videoName + " " + trackerName + " " + str(listOfIds))
            print("Objects tracked for video " + videoName + " and tracker " + trackerName + " = " + str(listOfIds))
            objectNumber = 0
            for oneObject in listOfIds: 
                objectNumber += 1
                print("Analyzing object " + str(objectNumber) + " on video " + videoName + " with tracker " + trackerName)
                (iou, cd) = process(videoName, trackerName, oneObject)
                f.write(str(oneObject) + " " + str(iou) + " " + str(cd) + "\n")
            f.close()

    f = open(results_processed_dir + "objectTypes.txt", "w")
    for x in objectTypes: 
        f.write(str(x[0]) + " " + str(x[1]) + "\n")
    f.close()