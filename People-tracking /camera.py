# Original
# Thomas Fortier under HackerShack
# 15/10/2017
# https://github.com/HackerShackOfficial/Smart-Security-Camera

# Revised Version
# Dominic Jolley
# 10/05/2020
# The University of Sheffield

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

        # apply the background subtractor to the frame, and remove shadows by setting
        # them to 0 (black)
        fgmask = fgbg.apply(frame)
        gray = fgmask 
        gray[gray==127] = 0
        
        return (frame,fgmask, gray)
