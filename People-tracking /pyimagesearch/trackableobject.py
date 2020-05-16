# Orginal Version from OpenCV People Counter 
# Adrian Rosebrock
# 13/08/2018
# https://www.pyimagesearch.com/2018/08/13/opencv-people-counter/

# Revised Version
# Dominic Jolley
# 10/05/2020
# The University of Sheffield

class TrackableObject:
    def __init__(self, objectID, centroid):
        # store the object ID, then initialize a list of centroids
        # using the current centroid
        self.objectID = objectID
        self.centroids = [centroid]
        
        # initialize a boolean used to indicate if the object has
        # already been counted or not
        self.counted = False
        self.disappeared = 0
        self.passedMid = False