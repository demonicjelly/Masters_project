import cv2
from imutils.video.pivideostream import PiVideoStream
import imutils
import time
import numpy as np

class VideoCamera(object):
    def __init__(self, flip = False):
        self.vs = PiVideoStream(resolution=(640, 480), framerate=16).start()
        self.flip = flip
        time.sleep(2.0)

    def __del__(self):
        self.vs.stop()

    def flip_if_needed(self, frame):
        if self.flip:
            return np.flip(frame, 0)
        return frame

    def get_frame(self):
        frame = self.flip_if_needed(self.vs.read())
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
    
    def get_processed(self, fgbg):
        frame = self.flip_if_needed(self.vs.read()).copy()
        frame = imutils.resize(frame, width=500)    

        fgmask = fgbg.apply(frame)
        fgmask[fgmask==127] = 0
        gray = fgmask
        
        return (frame, gray)

#     def get_object(self, fgbg):
#         found_objects = False
#         frame = self.flip_if_needed(self.vs.read()).copy() 
#         #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         fgmask = fgbg.apply(frame)
#         fgmask[fgmask==127] = 0
#         gray = fgmask
# 
# 
#         # if the average frame is None, initialize it
#         if avg is None:
#             print("[INFO] starting background model...")
#             avg = gray.copy().astype("float")
#             #rawCapture.truncate(0)
# 
# 
#         # accumulate the weighted average between the current frame and
#         # previous frames, then compute the difference between the current
#         # frame and running average
#         cv2.accumulateWeighted(gray, avg, 0.5)
#         frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))
# 
#         # threshold the delta image, dilate the thresholded image to fill
#         # in holes, then find contours on thresholded image
#         thresh = cv2.threshold(frameDelta, conf["delta_thresh"], 255,
#             cv2.THRESH_BINARY)[1]
#         thresh = cv2.dilate(thresh, None, iterations=2)
#         cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
#             cv2.CHAIN_APPROX_SIMPLE)
#         cnts = imutils.grab_contours(cnts)
# 
#         # loop over the contours
#         for c in cnts:
#             # if the contour is too small, ignore it
#             if cv2.contourArea(c) < conf["min_area"]:
#                 continue
# 
#             # compute the bounding box for the contour, draw it on the frame,
#             # and update the text
#             (x, y, w, h) = cv2.boundingRect(c)
#             box = [x, y, x + w, y + h]
#             rects.append(box)
#             #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
# 
# 
#         return (frame, rects)