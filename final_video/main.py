# import the necessary packages
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
from imutils.video import VideoStream
from camera_MOG import VideoCamera
from urllib.request import urlopen
import numpy as np
import argparse
import imutils
import time
import cv2
import http.client
import threading
import datetime
import sys

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str,
    help="path to optional input video file")
ap.add_argument("-o", "--output", type=str,
    help="path to optional output video file")
ap.add_argument("-iot", "--Iot", type=bool,
    help="connect to iot platform or not")
args = vars(ap.parse_args())

# if a video path was not supplied, grab a reference to the webcam
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


fgbg = cv2.createBackgroundSubtractorMOG2(history = 20, varThreshold = 25, detectShadows=True)
avg = None
maxDisappeared=30

# initialize our centroid tracker and frame dimensions
ct = CentroidTracker(maxDisappeared)
trackableObjects = {}
(H, W) = (None, None)

# initialize the total number of frames processed thus far, along
# with the total number of objects that have moved either up or down
totalFrames = 0
totalDown = 0
totalUp = 0

# initialize the video writer (we'll instantiate later if need be)
writer = None

#Set up IoT credentials

#key = 'DQGNBBNS99STVOBX' #Entrance 1 Channel 
key = 'APQHC5LHPK5ZNW5L'  #Entrance 2 Channel
baseURL = 'https://api.thingspeak.com/update?api_key=%s' % key

# loop over the frames from the video stream
while True:

    timestamp = datetime.datetime.now()


    #SET UP
    if video == True:
        # grab the next frame and handle if we are reading from either
        # VideoCapture or VideoStream
        frame = vs.read()
        frame = frame[1] if args.get("input", False) else frame
        frame = imutils.resize(frame, width=500)    

        fgmask = fgbg.apply(frame)
        fgmask[fgmask==127] = 0
        gray = fgmask
        # if we are viewing a video and we did not grab a frame then we
        # have reached the end of the video          
    else:
        frame, gray = video_camera.get_processed(fgbg)

    if args["input"] is not None and frame is None:
            break 


    # if the frame dimensions are empty, set them
    if W is None or H is None:
        (H, W) = frame.shape[:2]
        
    # if we are supposed to be writing a video to disk, initialize
    # the writer
    if args["output"] is not None and writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 30,
            (W, H), True)
           
    # draw a horizontal line in the center of the frame -- once an
    # object crosses this line we will determine whether they were
    # moving 'up' or 'down'
    cv2.line(frame, (0, H // 2), (W, H // 2), (0, 255, 255), 2)


    #DETECT
    
    # if the average frame is None, initialize it
    if avg is None:
        print("[INFO] starting background model...")
        avg = gray.copy().astype("float")
        #rawCapture.truncate(0)
        continue

    # accumulate the weighted average between the current frame and
    # previous frames, then compute the difference between the current
    # frame and running average
    cv2.accumulateWeighted(gray, avg, 0.5)
    frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))
    
    # threshold the delta image, dilate the thresholded image to fill
    # in holes, then find contours on thresholded image
    thresh = cv2.threshold(frameDelta, 5, 255, cv2.THRESH_BINARY)[1]   
    thresh = cv2.dilate(thresh, None, iterations=2)
    
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    rects = []

    # loop over the contours
    for c in cnts:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < 3500:
            continue

        # compute the bounding box for the contour
        (x, y, w, h) = cv2.boundingRect(c)
        box = [x, y, x + w, y + h]
        rects.append(box)
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


    #TRACKING

    # use the centroid tracker to associate the (1) old object
    # centroids with (2) the newly computed object centroids
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
            # centroid and the mean of *previous* centroids will tell
            # us in which direction the object is moving (negative for
            # 'up' and positive for 'down')
            y = [c[1] for c in to.centroids]
            direction = centroid[1] - np.mean(y)
            to.centroids.append(centroid)
            to.disappeared = disappeared.get(objectID)
            

            if H//2 - 10 < centroid[1] < H//2 + 10:
                to.passedMid = True

            # check to see if the object has been counted or not
            if not to.counted and to.passedMid == True and to.disappeared > maxDisappeared-5:
                # if the direction is negative (indicating the object
                # is moving up) AND the centroid is above the center
                # line, count the object
                if direction < 0 and centroid[1] < H // 2:
                    totalUp += 1
                    to.counted = True

                # if the direction is positive (indicating the object
                # is moving down) AND the centroid is below the
                # center line, count the object
                elif direction > 0 and centroid[1] > H // 2:
                    totalDown += 1
                    to.counted = True
                
                #Upload to IoT channel 
                if args["Iot"] is not None:
                    totalCount = totalUp - totalDown
                    if totalCount < 0
                        totalCount = 0
                    try:  
                        conn = urlopen(baseURL + '&field1=%s&field2=%s&field3=%s' % (totalUp, totalDown, totalCount))
                        print(conn.read())
                        conn.close()
                        print("Uploading data")
                    except:
                        print("connection failed")


        # store the trackable object in our nn  
        trackableObjects[objectID] = to

        # draw both the ID of the object and the centroid of the
        # object on the output frame
        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)


    #MAKE VIDEO LOOK NICE

    # construct a tuple of information we will be displaying on the
    # frame
    info = [
        ("Up", totalUp),
        ("Down", totalDown),
    ]

    # loop over the info tuples and draw them on our frame
    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    ts = timestamp.strftime("%A %d %B %Y %I:%M:%S%p")
    cv2.putText(frame, ts, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
        0.35, (0, 0, 255), 1)
    
    # check to see if we should write the frame to disk
    if writer is not None:
        writer.write(frame)

    # show the output frame
    cv2.imshow("Frame", frame)
    #cv2.imshow("gray", thresh)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

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



