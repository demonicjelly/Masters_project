import cv2
import sys
import copy
import time
#import threading
import numpy as np
from camera import VideoCamera
from tracker import Tracker


video_camera = VideoCamera(flip=False) # creates a camera object, flip vertically
object_classifier = cv2.CascadeClassifier("models/facial_recognition_model.xml") # an opencv classifier

def main():
    
    cap = cv2.VideoCapture('data/TrackingBugs.mp4')
    
    # Create Object Tracker
    tracker = Tracker(160, 30, 5, 100)

    # Variables initialization
    track_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                    (0, 255, 255), (255, 0, 255), (255, 127, 255),
                    (127, 0, 255), (127, 0, 127)]
    
    # initialize the total number of frames processed thus far, along
    # with the total number of objects that have moved either up or down
    totalFrames = 0
    totalDown = 0
    totalUp = 0

    # initialize the frame dimensions (we'll set them as soon as we read
    # the first frame from the video)
    W = None
    H = None
    
    while True:
        
        ret, frame = cap.read()
        found_objects = False
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        objects = classifier.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        if len(objects) > 0:
            found_objects = True

        centers = []

        # Draw a rectangle around the objects
        for (x, y, w, h) in objects:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            #centeroid = (x + w//2, y + h//2)
            b = np.array([[x + w//2], [y + h//2]])
            centers.append(np.round(b))


        orig_frame = copy.copy(frame)
        
        # if the frame dimensions are empty, set them
        #if W is None or H is None:
            #(H, W) = frame.shape[:2]

        #cv2.line(frame, (0, H // 2), (W, H // 2), (0, 255, 255), 2)

        # If centroids are detected then track them
        if (len(centers) > 0):

            # Track object using Kalman Filter
            tracker.Update(centers)

            # For identified object tracks draw tracking line
            # Use various colors to indicate different track_id
            for i in range(len(tracker.tracks)):
                if (len(tracker.tracks[i].trace) > 1):
                    for j in range(len(tracker.tracks[i].trace)-1):
                        # Draw trace line
                        x1 = tracker.tracks[i].trace[j][0][0]
                        y1 = tracker.tracks[i].trace[j][1][0]
                        x2 = tracker.tracks[i].trace[j+1][0][0]
                        y2 = tracker.tracks[i].trace[j+1][1][0]
                        clr = tracker.tracks[i].track_id % 9
                        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                                 track_colors[clr], 2)                        
                        
                        trace_dir = y2-np.mean(y1)
                        # check to see if the object has been counted or not
                        if not tracker.tracks[i].counted:
                            # if the direction is negative (indicating the object
                            # is moving up) AND the centroid is above the center
                            # line, count the object
                            if trace_dir < 0 and y2 < H // 2:
                                totalUp += 1
                                tracker.tracks[i].counted = True

                            # if the direction is positive (indicating the object
                            # is moving down) AND the centroid is below the
                            # center line, count the object
                            elif trace_dir > 0 and y2 > H // 2:
                                totalDown += 1
                                tracker.tracks[i].counted = True
                        
    


 


        # construct a tuple of information we will be displaying on the
        # frame
        info = [
            ("Up", totalUp),
            ("Down", totalDown),
            #("Status", status),
        ]

        # loop over the info tuples and draw them on our frame
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


        # Display the frame
        cv2.imshow('Track', frame)

        # Slower the FPS
        cv2.waitKey(50)

        # Check for key strokes
        k = cv2.waitKey(50) & 0xff
        if k == 27:  # 'esc' key has been pressed, exit program.
            break
                    
        totalFrames += 1
    
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()