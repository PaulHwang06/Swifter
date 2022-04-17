#python hackTJ.py --File testing/IMG_0130.MOV --height 73
# pylint: disable=no-member
from __future__ import print_function
from time import time
import cv2
import time
import numpy as np
import argparse
import random as rng
import os
import sys; args = sys.argv[0]
rng.seed(12345)
 

#variables
listDistance = []  #pixels
listDistanceMeters = [] #meters
listSpeed = [] #m/s
imageName = []
xCoord = []
heights = []
averageHeight = 0 
averageSpeed = 0  #m/s
totalDist = 0


#user input variables
parserUser = argparse.ArgumentParser(description='Information')
parserUser.add_argument('--FPS', default=60, dest='FPS', type=int, help='Frames your phone records in')
parserUser.add_argument('--File', dest='File', type=str, help='File name of the video')
parserUser.add_argument('--height', default=69, dest='height', type=int, help='Height of the runner')
args = parserUser.parse_args()

frameNum = args.FPS #frames that your phone records with
filename = args.File #file name that user uploads
user_height = args.height #inches
# change to args later

#takes a screenshot of the video every 1.5 seconds and puts the image names in a list
vidcap = cv2.VideoCapture(filename)
frames=frameNum
success,image = vidcap.read()
count = 0
success = True
while success:
    success,image = vidcap.read()        
    if(count%(frames*1.5)==0):
        cv2.imwrite("image%d.jpg" % count, image)     # save frame as JPEG file
        imageName.append("image%d.jpg" % count)
    count += 1

vidcap.release()
cv2.destroyAllWindows()


#methods for speed

def getSpeed(coveredDistance, timeTaken):
    speed=coveredDistance/timeTaken
    return speed

def averageSpeed(completeList):
    average = sum(completeList) /len(completeList)
    return average

def averageHeight(heights):
    average = sum(heights)/len(heights)
    return average


#speed finder
#making a bounding box around person

def get_output_layers(net):
        
        layer_names = net.getLayerNames()
        
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

        return output_layers

def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
        label = str(classes[class_id])

        color = COLORS[class_id]

        cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

        cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        if(label=='person'):
            xCoord.append(x)
            heights.append(y_plus_h-y)

for theimg in imageName:
        
    image = cv2.imread(theimg)

    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392

    classes = None

    with open('yolov3.txt', 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

    net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

    blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

    net.setInput(blob)

    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    for i in indices:
        i = i
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))

    count=count+1
        
    cv2.imwrite((str(theimg)[:len(str(theimg))-4])+"identifier"+'.jpg', image)
    #cv2.destroyAllWindows()


#finding distance between bounding boxes

averageHeight = averageHeight(heights)
scale = ((user_height-2)*0.0254)/averageHeight #meters/pixels

for item in range(len(xCoord)-1):
    listDistance.append(abs(xCoord[item+1]-xCoord[item]))
for item in range(len(listDistance)):
    listSpeed.append(getSpeed((listDistance[item]*scale),1.5))
averageSpeed = averageSpeed(listSpeed)
for item in listDistance:
    listDistanceMeters.append(item*scale)

print("Distance in pixels: " + str(listDistance))
print("Speed in m/s: " + str(listSpeed))
print("Average speed: " + str(averageSpeed) + " m/s")
print("Distance in meters: " + str(listDistanceMeters))
print("Total distance: " + str(sum(listDistanceMeters)) + " meters")


#pose estimator 

for eachImg in imageName:
#for x in range(1):
    #image = cv2.imread(eachImg)
    
    # parser = argparse.ArgumentParser(description='Run keypoint detection')
    # parser.add_argument("--device", default="cpu", help="Device to inference on")
    # parser.add_argument("--image_file", default=eachImg, help="Input image")

    #args = parser.parse_args()

    protoFile = "openpose-master/models/pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
    weightsFile = "openpose-master/models/pose/mpi/pose_iter_160000.caffemodel"
    nPoints = 15
    POSE_PAIRS = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14], [14,8], [8,9], [9,10], [14,11], [11,12], [12,13]]

    frame = cv2.imread(eachImg)
    #frame = cv2.imread('testing/usainbolt.jpeg')
    frameCopy = np.copy(frame)
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    threshold = 0.1

    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

    net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
    #print("Using CPU device")

    #t = time.time()
    # input image dimensions for the network
    inWidth = 368
    inHeight = 368
    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                            (0, 0, 0), swapRB=False, crop=False)

    net.setInput(inpBlob)

    output = net.forward()
    #print("time taken by network : {:.3f}".format(time.time() - t))

    H = output.shape[2]
    W = output.shape[3]

    # Empty list to store the detected keypoints
    points = []

    for i in range(nPoints):
        # confidence map of corresponding body's part.
        probMap = output[0, i, :, :]

        # Find global maxima of the probMap.
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
        
        # Scale the point to fit on the original image
        x = (frameWidth * point[0]) / W
        y = (frameHeight * point[1]) / H

        if prob > threshold : 
            cv2.circle(frameCopy, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frameCopy, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)

            # Add the point to the list if the probability is greater than the threshold
            points.append((int(x), int(y)))
        else :
            points.append(None)

    # Draw Skeleton
    for pair in POSE_PAIRS:
        partA = pair[0]
        partB = pair[1]

        if points[partA] and points[partB]:
            cv2.line(frame, points[partA], points[partB], (0, 255, 255), 2)
            cv2.circle(frame, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

    # cv2.imshow('Output-Keypoints', frameCopy)
    # cv2.imshow('Output-Skeleton', frame)


    #cv2.imwrite('Output-Keypoints.jpg', frameCopy)
    #cv2.imwrite('Output-Skeleton.jpg', frame)

    #print("Total time taken : {:.3f}".format(time.time() - t))

    #cv2.waitKey(0)
    
    #count=count+1
    #cv2.imwrite(str(eachImg) + "points"+'.jpg', frameCopy)
    cv2.imwrite((str(eachImg)[:len(str(eachImg))-4]) + "skeleton"+'.jpg', frame)
    #cv2.destroyAllWindows()