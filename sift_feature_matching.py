""" 
    Purpose: Use SIFT to extract features from a template image. This template image represents the object we are searching for. We can then
    use those features to check for the object in a larger, more uncontrolled scene.
    
    For further detail: David G. Lowe's paper on Distinctive Image Features from Scale-Invariant Keypoints
    
    Note: Query image --> template and training image --> search space
"""
#Libraries required (details added for Windows users)
import cv2 #pip install opencv-python
import numpy as np #pip install numpy
import matplotlib.pyplot as plt #pip install matplotlib
import os
import glob

#Detector basics
MIN_MATCH_COUNT=10 #Used to control how many matches are required before an object is "detected"
detector = cv2.SIFT_create() #SIFT object constructor

#Setting up Flann-based feature/description matcher to be used later (a bit faster than brute force for large datasets)
#to find k best matches for each descriptor from our query set 
FLANN_INDEX_KDTREE=0
flannParam = dict(algorithm=FLANN_INDEX_KDTREE, tree=5) #builds dictionary (https://python-reference.readthedocs.io/en/latest/docs/functions/dict.html)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(indexParams=flannParam, searchParams=search_params) #will train flann on descriptor collection (of dicts)

templateImg = cv2.imread(filename="images/template_images/stop_sign_1.png", flags=0) #read the image that is used to train the classifier (aka a template to match on other images)
#Detection section (template image aka query image)
templateKP, templateDesc = detector.detectAndCompute(image=templateImg, mask=None) #used to detect keypoints (templateKP) and get the descriptors (templateDesc) in a single step
templateWithKP = cv2.drawKeypoints(image=templateImg, keypoints=templateKP, outImage=None, color=(255, 0, 0), flags=4) #draws keypoints previously computed
plt.imshow(X=templateWithKP) #X is an array-like or PIL image
plt.show() #shows plot with keypoints (for template image) only ONCE

#Image(s) that need to be searched
folder = "images/test_images" #folder with images that need checking

#iterate through test images and check all of them for stop signs
for filename in os.listdir(folder):
    unknownImg = cv2.imread(os.path.join(folder, filename)) #gets an image in the folder (not necessarily sorted)
    
    #error checking (wrong directory could cause problems)
    if unknownImg is not None:
        unknownImg = cv2.resize(src=unknownImg, dsize=(700, 330)) #resize the image (helps quite a bit with consistency)
        
        #Use this to see the directory image every time (unedited):
        # cv2.imshow(winname="Directory image", mat=unknownImg)
        # cv2.waitKey(delay=0)
        
        #Detection section (unknown image(s) aka training image(s))
        
        #used to detect keypoints (unknownKP) and get the descriptors (unknownDesc) in a single step
        unknownKP, unknownDesc = detector.detectAndCompute(image=unknownImg, mask=None)
        matches = flann.knnMatch(queryDescriptors=templateDesc, trainDescriptors=unknownDesc, k=2) #finds best 2 matches for each unknownDesc


        #Actual object/feature matching section
        reasonableMatch=[] #array that will be used to hold the best matches
        for m, n in matches:
            #if the Euclidean distance is reasonably close (aka L2 metric or L2 norm) for the template image and the image we are checking
            if(m.distance < (0.75*n.distance)):
                reasonableMatch.append(m) #add it

        #If there are more reasonable matches than the minimum that was previously set (aka a stop sign has been found)
        if(len(reasonableMatch) > MIN_MATCH_COUNT):
            temppt=[]
            unkpt=[]
            
            for m in reasonableMatch:
                temppt.append(templateKP[m.queryIdx].pt) #template points (source points)
                unkpt.append(unknownKP[m.trainIdx].pt) #points from image we are checking (destination points)
            temppt, unkpt = np.float32((temppt, unkpt)) #convert all to float32 data type
            
            #Must be 3D and don't know first dimension (not consistent)
            temppt = temppt.reshape(-1, 1, 2)
            unkpt = unkpt.reshape(-1, 1, 2)
            
            #Object/feature detection visualization calculations (using above calculations)
            #Finds a perspective transformation between two planes (aka maps the points in one point to the corresponding point in another image)
            M, mask = cv2.findHomography(srcPoints=temppt, dstPoints=unkpt, method=cv2.RANSAC, ransacReprojThreshold=3.0)
            matchesMask = mask.ravel().tolist()
            
            h, w = templateImg.shape #returns the dimensions of the template image (height = h, width = w)
            points = np.float32([[[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]]).reshape(-1, 1, 2) #convert all to float32 data type
            destination = cv2.perspectiveTransform(src=points, m=M) #used to align image detection
            
            #Bounding box options (all can coincide to compare, but ideally just choose one)
            print("Stop sign has been found in " + filename)
            
            #Rotated rectangle option
            # rot_rect = cv2.minAreaRect(destination)
            # box = cv2.boxPoints(rot_rect)
            # box = np.int0(box)
            # cv2.drawContours(unknownImg, [box], 0, (0,0,255), 2)
            # cv2.imshow(winname="Stop sign detection", mat=unknownImg)
            # cv2.waitKey(delay=0)
            
            #Straight rectangle option
            x,y,w,h = cv2.boundingRect(array=destination)
            cv2.rectangle(img=unknownImg, pt1=(x,y), pt2=(x+w, y+h), color=(0,255,0), thickness=2)
            cv2.imshow(winname="Stop sign found", mat=unknownImg)
            cv2.waitKey(delay=0)
            
            #Matched image visualization
            draw_params = dict(matchColor=(0,255,0), singlePointColor=None, matchesMask=matchesMask, flags=2) #specifying custom parameters (flag = 2 means only matched keypoints shown)
            matchedImg = cv2.drawMatches(templateImg, templateKP, unknownImg, unknownKP, reasonableMatch, None, **draw_params) #draw keypoints and connectors
            plt.imshow(X=matchedImg, cmap='gray')
            plt.show()
        
        #if not enough matches (aka a stop sign has not been found)
        else:
            print("No stop signs have been found in " + filename)
            matchesMask=None
            cv2.imshow(winname="No match found", mat=unknownImg)
            cv2.waitKey(delay=0)

cv2.destroyAllWindows() #will close any windows left open if the program is forcibly stopped