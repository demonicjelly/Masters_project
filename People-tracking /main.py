# Dominic Jolley
# 10/05/2020
# The University of Sheffield

# Parts adpated from OpenCV People Counter 
# Adrian Rosebrock
# 13/08/2018
# https://www.pyimagesearch.com/2018/08/13/opencv-people-counter/

#To run: enter directory with this file in first 
#        python main.py    to run from video stream with no output an not linking to iot platform
#        python main.py --input videos/example_01.mp4   to run with a video file instead of the pi camera
#        python main.py --output output/out_vid.avi
#        python main.py --Iot True     to run and send room count to the iot platform

#        python main.py --output output/out_vid.avi --Iot True
#
# import the necessary packages and libraries
from centroidtracker import CentroidTracker
from trackableobject import TrackableObject
from camera import VideoCamera
from urllib.request import urlopen
import numpy as np
import argparse
import imutils
import time
import cv2
import http.client
import datetime
import sys

#Construct argument parse and parse aguments
ap = argparse.ArgumentParser()
ap.add_argument("--input", type=str,
    help="path to optional input video file")
ap.add_argument("--output", type=str,
    help="path to optional output video file")
ap.add_argument("--Iot", type=bool,
    help="connect to iot platform or not")
args = vars(ap.parse_args())

# if a video path was not supplied, get the webcam stream
if not args.get("input", False):
    print("[INFO] starting video stream...")
    video_camera = VideoCamera(flip=False) # creates a camera object, flip vertically
    time.sleep(2.0)
    video = False
# otherwise, grab a reference to the video file
else:
    print("[INFO] opening video file...")
    vs = cv2.VideoCapture(args["input"])
    video = True

#initialise the background subtractor and the number of frames it takes for an object
# to dissapear 
fgbg = cv2.createBackgroundSubtractorMOG2(history = 500, varThreshold = 25, detectShadows=True)
maxDisappeared=15


# initialize our centroid tracker and frame dimensions
ct = CentroidTracker(maxDisappeared)
trackableObjects = {}
(H, W) = (None, None)

# initialize the total number of objects that have moved either up or down,
# and the total count in the room
totalDown = 0
totalUp = 0
totalCount = 0 

# initialize the video writer (instantiated later if needed)
writer = None

#Set up IoT credentials

key = 'DQGNBBNS99STVOBX' #Entrance 1 Channel 
#key = 'APQHC5LHPK5ZNW5L'  #Entrance 2 Channel
baseURL = 'https://api.thingspeak.com/update?api_key=%s' % key

framecount =0 #variable used for taking testing images

# loop over the frames from the video or webcam stream
while True:
    # Get a timestamp for the current frame
    timestamp = datetime.datetime.now()


    #SET UP
    if video == True:
        # grab the next frame and handle and resize frame
        frame = vs.read()
        frame = frame[1] if args.get("input", False) else frame
        frame = imutils.resize(frame, width=500)    
        # apply the background subtractor to the frame, and remove shadows by setting
        # them to 0 (black)
        fgmask = fgbg.apply(frame)
        gray = fgmask 
        gray[gray==127] = 0
        
    else:
        frame, fgmask, gray = video_camera.get_processed(fgbg)

    # if we are viewing a video and we did not find a frame then we
    # have reached the end of the video
    if args["input"] is not None and frame is None:
            break 

    # if the frame dimensions are empty, set them
    if W is None or H is None:
        (H, W) = frame.shape[:2]
        
    #  initialise writer if needed
    if args["output"] is not None and writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 30,
            (W, H), True)
           
    # draw a horizontal line in the center of the output frame 
    cv2.line(frame, (0, H // 2), (W, H // 2), (0, 255, 255), 2)
   
    # Dilate the background subtracted image to fill in holes
    # (with a 3x3 rectangle kernel)
    dilated = cv2.dilate(gray, cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)), iterations=2)
    
    #DETECT
    
    #find contours of the dilated image 
    cnts = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    rects = []

    # loop over the contours
    for c in cnts:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < 3500:
            continue

        # compute the bounding box for the contour and append it to
        # a list of rectangles
        (x, y, w, h) = cv2.boundingRect(c)
        box = [x, y, x + w, y + h]
        rects.append(box)
        
        #Draw bounding box rectangle on output frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


    #TRACKING

    # use the centroid tracker to associate the old object
    # centroids with the newly computed object centroids
    objects, disappeared = ct.update(rects)
    
    # loop over the tracked objects
    for (objectID, centroid) in objects.items():
        # check to see if a trackable object exists for the current
        # object ID
        to = trackableObjects.get(objectID, None)
        

        # if there is no existing trackable object, create one
        if to is None:
            to = TrackableObject(objectID, centroid)

        # otherwise, there is a trackable object so we can utilize it
        # to determine direction
        else:
            # the difference between the y-coordinate of the *current*
            # centroid and the mean of previous centroids can be used 
            # to find which direction the object is moving in(negative for
            # 'up' and positive for 'down')
            # additionally add dissapeared count to the trackable object
            y = [c[1] for c in to.centroids]
            direction = centroid[1] - np.mean(y)
            to.centroids.append(centroid)
            to.disappeared = disappeared.get(objectID)
            
            # if the centroid is recorded 'near' to the centre of the
            # screen (within 20 pixels) then record that it has passed the middle
            if H//2 - 20 < centroid[1] < H//2 + 20:
                to.passedMid = True

            # Counting objects takes place only if the object has not already been counted,
            # is has passed the centre of the screen and it is about to dissapear
            if not to.counted and to.passedMid == True and to.disappeared > maxDisappeared-1:
                # if the direction is negative (indicating the object
                # is moving up) AND the centroid is above the center
                # line, count the object
                if direction < 0 and centroid[1] < H // 2:
                    totalUp += 1
                    totalCount += 1
                    to.counted = True

                # if the direction is positive (indicating the object
                # is moving down) AND the centroid is below the
                # center line, count the object
                elif direction > 0 and centroid[1] > H // 2:
                    totalDown += 1
                    totalCount -= 1
                    to.counted = True
                
                # Optional if to reset the total count to 0, only to be applied
                # if there is one entrance to the room and a person is already inside
                # if there are multiple entrances a negative room count is fine
                #if totalCount < 0
                    #totalCount = 0

                #Upload to IoT channel if required 
                if args["Iot"] is not None:
                    try:  
                        conn = urlopen(baseURL + '&field1=%s&field2=%s&field3=%s' % (totalUp, totalDown, totalCount))
                        print(conn.read())
                        conn.close()
                        print("Uploading data")
                    except:
                        print("connection failed")


        # store the trackable object in the dictionary  
        trackableObjects[objectID] = to

        # draw both the ID of the object and the centroid of the
        # object on the output frame
        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)


    #MAKE VIDEO LOOK NICE

    # construct a tuple of information to display on the
    # frame
    info = [
        ("Up", totalUp),
        ("Down", totalDown),
        ("Count", totalCount),
    ]

    # loop over the info tuples and draw them on our frame
    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    #ts = timestamp.strftime("%A %d %B %Y %I:%M:%S%p")
    #cv2.putText(frame, ts, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
    #    0.35, (0, 0, 255), 1)
    
    # check to see if we should write the frame to disk
    if writer is not None:
        writer.write(frame)

    # show the output frames
    cv2.imshow("Frame", frame)
    #cv2.imshow("fgmask", fgmask)
    #cv2.imshow("gray", gray)
    cv2.imshow("dilated", dilated)

    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed in a cv frame, break from the loop
    if key == ord("q"):
        break

# Used to take test images of the different stages of the frame on 'p'
    if key == ord("p"):
        cv2.imwrite("images/frame%d.jpg" % framecount, frame)
        cv2.imwrite("images/fgmask%d.jpg" % framecount, fgmask)
        cv2.imwrite("images/gray%d.jpg" % framecount, gray)
        cv2.imwrite("images/dilated%d.jpg" % framecount, dilated)
        framecount += 1

# check to see if we need to release the video writer pointer
if writer is not None:
    writer.release()

# if we are not using a video file, stop the camera video stream
if not args.get("input", False):
    vs.stop()

# otherwise, release the video file pointer
else:
    vs.release()

# do a bit of cleanup
cv2.destroyAllWindows()



