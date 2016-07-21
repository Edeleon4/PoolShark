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
                    if s < image.shape[0] and t < image.shape[1] and s >= 0 and t >= 0 and \
                	    image[s, t] not in (BORDER_COLOR, value):
                        image[s, t] = value
                        points.append((s, t))
                        count += 1
                        newedge.append((s, t))
            edge = newedge

        return count, points

    # thresholds for different balls / background
    low_white = np.array([30, 0, 210], dtype=np.uint8)
    high_white = np.array([120, 40, 255], dtype=np.uint8)

    low_bkg = np.array([15, 40, 50], dtype=np.uint8)
    high_bkg = np.array([40, 190, 200], dtype=np.uint8)

    low_blue = np.array([110,80,80], dtype=np.uint8)
    high_blue = np.array([120,255,255], dtype=np.uint8)

    low_purple = np.array([110,0,0], dtype=np.uint8)
    high_purple = np.array([150,100,100], dtype=np.uint8)

    low_yellow = np.array([20, 150, 150], dtype=np.uint8)
    high_yellow = np.array([30, 255, 255], dtype=np.uint8)

    low_red = np.array([160, 100, 100], dtype=np.uint8)
    high_red = np.array([180, 255, 255], dtype=np.uint8)
    
    low_orange = np.array([5, 200, 110], dtype=np.uint8)
    high_orange = np.array([15, 255, 255], dtype=np.uint8)
    
    low_brown = np.array([0, 90, 10], dtype=np.uint8)
    high_brown = np.array([15, 120, 65], dtype=np.uint8)

    # mask out the background
    bkg_mask = cv2.inRange(hsv, low_bkg, high_bkg)
    # img_bkg_mask = cv2.bitwise_and(frame,frame, mask=bkg_mask)
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

    # mask the purple balls
    purple_mask = cv2.inRange(hsv, low_purple, high_purple)
    
    # mask the orange balls
    orange_mask = cv2.inRange(hsv, low_orange, high_orange)
    
    # mask the brown balls
    brown_mask = cv2.inRange(hsv, low_brown, high_brown)

    # mask the cue ball
    white_mask = cv2.inRange(hsv, low_white, high_white)
    #
    yellows = cv2.bitwise_and(objects, objects, mask=yellow_mask)
    reds = cv2.bitwise_and(objects, objects, mask=red_mask)
    blues = cv2.bitwise_and(objects, objects, mask=blue_mask)
    purples = cv2.bitwise_and(objects, objects, mask=purple_mask)
    whites = cv2.bitwise_and(objects, objects, mask=white_mask)
    oranges = cv2.bitwise_and(objects, objects, mask=orange_mask)
    browns = cv2.bitwise_and(objects, objects, mask=brown_mask)

    cv2.imwrite('whites.jpg', whites)
    del whites
    cv2.imwrite('purples.jpg', purples)
    del purples
    cv2.imwrite('oranges.jpg', oranges)
    del oranges
    cv2.imwrite('mask.jpg', bkg_mask)
    cv2.imwrite('yellows.jpg', yellows)
    del yellows
    cv2.imwrite('reds.jpg', reds)
    del reds
    cv2.imwrite('blues.jpg', blues)
    del blues
    cv2.imwrite('browns.jpg', browns)
    del browns
    #
    # find the biggest cloud of 1's in the yellow mask
    yellow_image = yellow_mask# / 255.
    del yellow_mask
    red_image = red_mask# / 255.
    del red_mask
    blue_image = blue_mask# / 255.
    del blue_mask
    white_image = white_mask / 255.
    del white_mask
    purple_image = purple_mask# / 255.
    del purple_mask
    orange_image = orange_mask# / 255.
    del orange_mask
    brown_image = brown_mask# / 255.
    del brown_mask
    
    def findBiggestCloud(image):
        biggest_cloud = []
        biggest_count = 0
        # components = np.where(image == 1)
        # bwImage = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY);
        contours = cv2.findContours(image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)[0]
        centerx = 0
        centery = 0

        for contour in contours:
            # Approximates rectangles
            bound_rect = cv2.boundingRect(contour)
            count = bound_rect[2] * bound_rect[3];
            if count > biggest_count:
                centerx = bound_rect[0] + bound_rect[2] / 2
                centery = bound_rect[1] + bound_rect[3] / 2
                biggest_count = count
        return centerx, centery, biggest_count

    def findBiggestAndSecondBiggestCloudFast(image):
        biggest_count = 0
        second_biggest_count = 0
        contours = cv2.findContours(image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)[0]
        centerx = 0
        centery = 0
        second_centerx = 0
        second_centery = 0

        for contour in contours:
            # Approximates rectangles
            bound_rect = cv2.boundingRect(contour)
            count = bound_rect[2] * bound_rect[3];
            if count > biggest_count:
                second_biggest_count = biggest_count
                second_centerx = centerx
                second_centery = centery
                centerx = bound_rect[0] + bound_rect[2] / 2
                centery = bound_rect[1] + bound_rect[3] / 2
                biggest_count = count
            elif count > second_biggest_count:
                second_biggest_count = count
                second_centerx = bound_rect[0] + bound_rect[2] / 2
                second_centery = bound_rect[1] + bound_rect[3] / 2
        return centerx, centery, biggest_count, second_centerx, second_centery, second_biggest_count

    def findBiggestWhiteCloud(image):
        biggest_cloud = []
        biggest_count = 0
        components = np.where(image == 1)

        while len(components[0]) > 1000 and len(components[0]) > biggest_count:
            y = components[0][0]
            x = components[1][0]
            
            count, cloud = flood_fill(image, y, x, 2)
            if count > biggest_count:
                print count
                biggest_count = count
                biggest_cloud = cloud
            del components
            components = np.where(image == 1)

        print biggest_count

        totalY=0.0
        totalX=0.0
        for i in range (0,biggest_count):
            totalY += biggest_cloud[i][0]
            totalX += biggest_cloud[i][1]
        ycoord = totalY / biggest_count
        xcoord = totalX / biggest_count
        return xcoord, ycoord, biggest_count

    def findBiggestAndSecondBiggestCloud(image):
        second_biggest_cloud = []
        second_biggest_count = 0
        biggest_cloud = []
        biggest_count = 0
        components = np.where(image == 1)
        while len(components[0]) > 1000 and len(components[0]) > second_biggest_count:
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
                second_biggest_count = count
                second_biggest_cloud = cloud
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
        return xcoord, ycoord, biggest_count, secxcoord, secycoord, second_biggest_count
    
    
    xwht, ywht, whtcount = findBiggestWhiteCloud(white_image)
    cv2.circle(frame, (int(xwht), int(ywht)), 30, (255,0,255), -1)
    # give list of points to predict next shot - first white ball, then next set of balls
    listOfPointData = [(xwht, ywht)]

    xyel, yyel, yelcount, xyel2, yyel2, yelcount2 = findBiggestAndSecondBiggestCloudFast(yellow_image)
    thresholdCount = 1000
    if yelcount > thresholdCount:
        cv2.circle(frame, (int(xyel), int(yyel)), 30, (0,255,64), -1)
        listOfPointData.append((xyel, yyel))

    xblue, yblue, bluecount = findBiggestCloud(blue_image)
    if bluecount > thresholdCount:
        cv2.circle(frame, (int(xblue), int(yblue)), 30, (0,255,64), -1)
        listOfPointData.append((xblue, yblue))

    xred, yred, redcount = findBiggestCloud(red_image)
    if redcount > thresholdCount:
        cv2.circle(frame, (int(xred), int(yred)), 30, (0,255,64), -1)
        listOfPointData.append((xred, yred))
        
    xpurple, ypurple, purplecount = findBiggestCloud(purple_image)
    if purplecount > thresholdCount:
        cv2.circle(frame, (int(xpurple), int(ypurple)), 30, (0,255,64), -1)
        listOfPointData.append((xpurple, ypurple))

    xorange, yorange, orangecount = findBiggestCloud(orange_image)
    if orangecount > thresholdCount:
        cv2.circle(frame, (int(xorange), int(yorange)), 30, (0,255,64), -1)
        listOfPointData.append((xorange, yorange))
        
    xbrown, ybrown, browncount = findBiggestCloud(brown_image)
    if browncount > thresholdCount:
        cv2.circle(frame, (int(xbrown), int(ybrown)), 30, (0,255,64), -1)
        listOfPointData.append((xbrown, ybrown))

    if yelcount2 > thresholdCount:
        cv2.circle(frame, (int(xyel2), int(yyel2)), 30, (0,255,64), -1)
        listOfPointData.append((xyel2, yyel2))

    #cv2.imwrite('frame.jpg', frame)

    # send listOfPointData to Eddi's function

    return frame

#####################################
# ENTRY FUNCTIONS
#####################################
def findAndDrawGiraffeBody(imgPath):
    printLogMsg("Running giraffe body detection...")
    # imgResult,confidence,bboxes = findAndDrawGiraffeBodyImpl(imgPath)    
    return imgResult,confidence,bboxes

