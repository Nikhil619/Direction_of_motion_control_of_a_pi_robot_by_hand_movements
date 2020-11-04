import cv2
import time
import numpy as np
import os
import numpy as np
import RPi.GPIO as GPIO
from time import sleep
import time
from datetime import datetime


GPIO.setmode(GPIO.BCM)

m10 = 20
m11 = 16
m20 = 23
m21 = 24
reset = 2

GPIO.setup(m10,GPIO.OUT)
GPIO.setup(m11,GPIO.OUT)
GPIO.setup(m20,GPIO.OUT)
GPIO.setup(m21,GPIO.OUT)

GPIO.setup(reset,GPIO.IN)

p10 = GPIO.PWM(m10,50)
p11 = GPIO.PWM(m11,50)
p20 = GPIO.PWM(m20,50)
p21 = GPIO.PWM(m21,50)
p10.start(0)
p11.start(0)
p20.start(0)
p21.start(0)


def left():
    p10.ChangeDutyCycle(0)
    p11.ChangeDutyCycle(0)
    p20.ChangeDutyCycle(50)
    p21.ChangeDutyCycle(0)


def right():
    p10.ChangeDutyCycle(50)
    p11.ChangeDutyCycle(0)
    p20.ChangeDutyCycle(0)
    p21.ChangeDutyCycle(0)


def frwd():
    p10.ChangeDutyCycle(100)
    p11.ChangeDutyCycle(0)
    p20.ChangeDutyCycle(100)
    p21.ChangeDutyCycle(0)


def stop():
    p10.ChangeDutyCycle(0)
    p11.ChangeDutyCycle(0)
    p20.ChangeDutyCycle(0)
    p21.ChangeDutyCycle(0)

def back():
    p10.ChangeDutyCycle(0)
    p11.ChangeDutyCycle(100)
    p11.ChangeDutyCycle(100)
    p11.ChangeDutyCycle(0)
    
    
def nothing():
    pass


MODE = "MPI"
device = "gpu"


if MODE == "MPI" :
    protoFile = "pose\mpi\pose_deploy_linevec_faster_4_stages.prototxt"
    weightsFile = "pose\mpi\pose_iter_160000.caffemodel"
    nPoints = 15
    POSE_PAIRS = [[0,1], [1,2]]
inWidth = 368
inHeight = 368
threshold = 0.1


#input_source = args.video_file
cap = cv2.VideoCapture("input_video.mp4")

#setting the frame size
cap.set(3, 320) #setting the horizontal frame size 
cap.set(4, 240) #setting the vertical frame size

hasFrame, frame = cap.read()

net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
if device == "cpu":
    net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
    print("Using CPU device")
elif device == "gpu":
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    print("Using GPU device")

while cv2.waitKey(1) < 0:
    t = time.time()
    hasFrame, frame = cap.read()
    frameCopy = np.copy(frame)
    if not hasFrame:
        cv2.waitKey()
        break

    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                              (0, 0, 0), swapRB=False, crop=False)
    net.setInput(inpBlob)
    output = net.forward()

    H = output.shape[2]
    W = output.shape[3]
    # Empty list to store the detected keypoints
    points = []

    for i in range(nPoints):
        if i == 5 or i == 6:
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
                
        elif i == 7:
            probMap = output[0, i, :, :]
            # Find global maxima of the probMap.
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
        
            # Scale the point to fit on the original image
            x = (frameWidth * point[0]) / W
            y = (frameHeight * point[1]) / H

            if prob > threshold :
                if x > 214 and y > 160:
                    nothing()
                    cv2.putText(frame, "BACK RIGHT", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, lineType=cv2.LINE_AA)
                elif x > 214 and y < 80:
                    nothing()
                    cv2.putText(frame, "FRONT RIGHT", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, lineType=cv2.LINE_AA)
                elif x > 214:
                    right()
                    cv2.putText(frame, "GO RIGHT", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, lineType=cv2.LINE_AA)
                elif x < 107 and y > 160:
                    nothing()
                    cv2.putText(frame, "BACK LEFT", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, lineType=cv2.LINE_AA)
                elif x < 107 and y < 80:
                    nothing()
                    cv2.putText(frame, "FRONT LEFT", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, lineType=cv2.LINE_AA)
                elif x < 107:
                    left()
                    cv2.putText(frame, "GO LEFT", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, lineType=cv2.LINE_AA)
                elif y > 160:
                    back()
                    cv2.putText(frame, "COME BACK", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, lineType=cv2.LINE_AA)
                elif y < 80:
                    frwd()
                    cv2.putText(frame, "GO FRONT", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, lineType=cv2.LINE_AA)
                else:
                    stop()
                    cv2.putText(frame, "STOP", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, lineType=cv2.LINE_AA)

                # Add the point to the list if the probability is greater than the threshold
                points.append((int(x), int(y)))
            else :
                points.append(None)
        

    # Draw Skeleton
    for pair in POSE_PAIRS:
        partA = pair[0]
        partB = pair[1]

        if points[partA] and points[partB]:
            cv2.line(frame, points[partA], points[partB], (0, 255, 255), 1, lineType=cv2.LINE_AA)
            cv2.circle(frame, points[partA], 3, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.circle(frame, points[partB], 3, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

    cv2.imshow('Output-Skeleton', frame)

nothing()


