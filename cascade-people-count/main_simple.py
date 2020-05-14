# import the necessary packages
from pyimagesearch.centroidtracker import CentroidTracker
from imutils.video import VideoStream
from camera import VideoCamera
import numpy as np
import argparse
import imutils
import time
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", type=str,
    help="path to optional output video file")
args = vars(ap.parse_args())

video_camera = VideoCamera(flip=False) # creates a camera object, flip vertically
object_classifier = cv2.CascadeClassifier("models/facial_recognition_model.xml") # an opencv classifier

# initialize our centroid tracker and frame dimensions
ct = CentroidTracker()
(H, W) = (None, None)

# initialize the video stream and allow the camera sensor to warmup
print("[INFO] starting video stream...")
time.sleep(2.0)

# loop over the frames from the video stream
while True:
    # read the next frame from the video stream and resize it
    #frame = video_camera.get_frame()
    #frame = imutils.resize(frame, width=400)

    # construct a blob from the frame, pass it through the network,
    # obtain our output predictions, and initialize the list of
    # bounding box rectangles
    
    frame, found_obj, rects = video_camera.get_object(object_classifier)
    
    if args["output"] is not None and writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 30,
            (W, H), True)

    #blob = cv2.dnn.blobFromImage(frame, 1.0, (W, H),
    #    (104.0, 177.0, 123.0))
    #net.setInput(blob)
    #detections = net.forward()
    

    # update our centroid tracker using the computed set of bounding
    # box rectangles
    objects = ct.update(rects)

    # loop over the tracked objects
    for (objectID, centroid) in objects.items():
        # draw both the ID of the object and the centroid of the
        # object on the output frame
        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    if writer is not None:
        writer.write(frame)
    
    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
    
# check to see if we need to release the video writer pointer
if writer is not None:
    writer.release()

# do a bit of cleanup
cv2.destroyAllWindows()
