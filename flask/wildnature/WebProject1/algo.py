#################################################################
# Computer Vision functionality
#################################################################
import os, sys, datetime, time
sys.path.append('libraries/')
from pabuehle_utilities_CV_v1 import *
from pabuehle_utilities_general_v0 import *



#####################################
# PARAMETERS
#####################################
projectOxfordKey = "52cea468d3874a4c86add441e113f4ed"

#default params - hog descriptor
blockSize   = (16,16)   #only 16x16 supported
blockStride = (8,8)     #has to be multiple of cellSize
cellSize    = (8,8)     #only 8x8 supported
nbins       = 9
derivAperture     = 1   #?size of sobel kernel used to calculate derivative
winSigma          = -1  #4
histogramNormType = 0   #0 == L2Hys
#L2HysThreshold    = 0.2 
#gammaCorrection   = 0  #binary variable, looks like this refers to taking the sqrt of the pixel values before computing gradients
nlevels           = 64
signedGradients = False #default=False. Very new feature, only in OpenCV3

#specific parameters
datasetName = "giraffe_v2"
rootDir = "./resources/"
targetWidth = 96    
targetHeight = 96   
bboxGrowScale = 1.1 
L2HysThreshold    = 10
gammaCorrection   = 1

#directories and files
imgDir = rootDir + "data/" + datasetName + "/"
procDir = rootDir + "proc/" + datasetName + "/"
resultsDir = rootDir + "results/" + datasetName + "/"
learnerPath = resultsDir + "learner.pickle"





#####################################
# HELPER FUNCTIONS
#####################################
#print text and add to log
logMsgs = []
def printLogMsg(msg):
    msg = str(datetime.datetime.now()) + ":" + msg
    print msg
    logMsgs.append(msg)

def resetLog():
    if len(logMsgs) > 100000:
        del logMsgs[:]

#####################################
# CV IMPLEMENTATION
#####################################
def poolTableDetectAndGetCoordinates(imgPath):
    ############
    # Parameters
    ############
    minRelWidth = 0.05
    imresizeWidth = 1500
    hogDetectParams = {'hitThreshold': 0, 'finalThreshold': -2, 'winStride': (8, 8), 'useMeanshiftGrouping': False} #, 'scale0': 1.0}
    boVisualizeDetections = True
    boEvaluateModel = False


    ############
    # Main
    ############
    frame = imread(imgPath)
    h,w,c = frame.shape
    print frame.shape

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    BORDER_COLOR = 0
    def flood_fill(image, x, y, value):
        count = 1
        points = [(x, y)]
        "Flood fill on a region of non-BORDER_COLOR pixels."
        if x >= image.shape[0] or y >= image.shape[1] or image[x,y] == BORDER_COLOR:
            return None, None
        edge = [(x, y)]
        image[x, y] = value

        while edge:
            newedge = []
            for (x, y) in edge:
                for (s, t) in ((x+1, y), (x-1, y), (x, y+1), (x, y-1)):
                    if s < image.shape[0] and t < image.shape[1] and \
                	    image[s, t] not in (BORDER_COLOR, value):
                        image[s, t] = value
                        points.append((s, t))
                        count += 1
                        newedge.append((s, t))
            edge = newedge

        return count, points

    # thresholds for different balls / background
    low_bkg = np.array([15, 40, 50], dtype=np.uint8)
    high_bkg = np.array([40, 190, 200], dtype=np.uint8)

    low_blue = np.array([110,80,80], dtype=np.uint8)
    high_blue = np.array([130,255,255], dtype=np.uint8)

    low_yellow = np.array([20, 150, 150], dtype=np.uint8)
    high_yellow = np.array([30, 255, 255], dtype=np.uint8)

    low_red = np.array([160, 100, 100], dtype=np.uint8)
    high_red = np.array([180, 255, 255], dtype=np.uint8)

    # mask out the background
    bkg_mask = cv2.inRange(hsv, low_bkg, high_bkg)
    img_bkg_mask = cv2.bitwise_and(frame,frame, mask=bkg_mask)
    # bkg_mask /= 255.
    bkg_mask = np.invert(bkg_mask)

    # Bitwise-AND bkg_mask and original image
    objects = cv2.bitwise_and(frame,frame, mask=bkg_mask)

    hsv = cv2.cvtColor(objects, cv2.COLOR_BGR2HSV)

    # mask the yellow balls
    yellow_mask = cv2.inRange(hsv, low_yellow, high_yellow)
    
    # mask the red balls
    red_mask = cv2.inRange(hsv, low_red, high_red)
    
    # mask the blue balls
    blue_mask = cv2.inRange(hsv, low_blue, high_blue)

    yellows = cv2.bitwise_and(objects, objects, mask=yellow_mask)
    reds = cv2.bitwise_and(objects, objects, mask=red_mask)
    blues = cv2.bitwise_and(objects, objects, mask=blue_mask)

    # find the biggest cloud of 1's in the yellow mask
    yellow_image = yellow_mask / 255.
    red_image = red_mask / 255.
    blue_image = blue_mask / 255.
    
    def findBiggestCloud(image):
        biggest_cloud = []
        biggest_count = 0
        components = np.where(image == 1)
        while len(components[0]) > biggest_count:
            loc = components
            y = loc[0][0]
            x = loc[1][0]
            count, cloud = flood_fill(image, y, x, 2)
            if count > biggest_count:
                print count
                biggest_count = count
                biggest_cloud = cloud
            components = np.where(image == 1)

        # print biggest_cloud
        print biggest_count

        totalY=0.0
        totalX=0.0
        for i in range (0,count):
            totalY += biggest_cloud[i][0]
            totalX += biggest_cloud[i][1]
        ycoord = totalY / count
        xcoord = totalX / count
        return xcoord, ycoord

    def findBiggestAndSecondBiggestCloud(image):
        second_biggest_cloud = []
        second_biggest_count = 0
        biggest_cloud = []
        biggest_count = 0
        components = np.where(image == 1)
        while len(components[0]) > second_biggest_count:
            y = components[0][0]
            x = components[1][0]
            count, cloud = flood_fill(image, y, x, 2)
            if count > biggest_count:
                print count
                second_biggest_count = biggest_count
                second_biggest_cloud = biggest_cloud
                biggest_count = count
                biggest_cloud = cloud
            elif count > second_biggest_count:
                print count
                second_biggest_count = biggest_count
                second_biggest_cloud = biggest_cloud
            components = np.where(image == 1)

        # print biggest_cloud
        print biggest_count
        # print second_biggest_cloud
        print second_biggest_count

        totalY=0.0
        totalX=0.0
        for i in range (0,biggest_count):
            totalY += biggest_cloud[i][0]
            totalX += biggest_cloud[i][1]
        ycoord = totalY / biggest_count
        xcoord = totalX / biggest_count

        
        sectotalY=0.0
        sectotalX=0.0
        for i in range (0,second_biggest_count):
            sectotalY += second_biggest_cloud[i][0]
            sectotalX += second_biggest_cloud[i][1]
        secycoord = sectotalY / second_biggest_count
        secxcoord = sectotalX / second_biggest_count
        return xcoord, ycoord, secxcoord, secycoord
    
    xyel, yyel, xyel2, yyel2 = findBiggestAndSecondBiggestCloud(yellow_image)
    cv2.circle(frame, (int(xyel), int(yyel)), 30, (0,255,64), -1)
    cv2.circle(frame, (int(xyel2), int(yyel2)), 30, (0,255,64), -1)
    xred, yred = findBiggestCloud(red_image)
    cv2.circle(frame, (int(xred), int(yred)), 30, (0,255,64), -1)
    xblue, yblue = findBiggestCloud(blue_image)
    cv2.circle(frame, (int(xblue), int(yblue)), 30, (0,255,64), -1)

    cv2.imwrite('mask.jpg', bkg_mask)
    cv2.imwrite('img_bkg_mask.jpg', img_bkg_mask)
    cv2.imwrite('yellows.jpg', yellows)
    cv2.imwrite('reds.jpg', reds)
    cv2.imwrite('blues.jpg', blues)
    cv2.imwrite('frame.jpg', frame)

    return frame

#####################################
# ENTRY FUNCTIONS
#####################################
def findAndDrawGiraffeBody(imgPath):
    printLogMsg("Running giraffe body detection...")
    # imgResult,confidence,bboxes = findAndDrawGiraffeBodyImpl(imgPath)    
    return imgResult,confidence,bboxes
