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

    lower_blue = np.array([110,50,50], dtype=np.uint8)
    upper_blue = np.array([130,255,255], dtype=np.uint8)

    low_yellow = np.array([20, 30, 30], dtype=np.uint8)
    high_yellow = np.array([30, 255, 255], dtype=np.uint8)


    # mask out the background
    mask = cv2.inRange(hsv, low_bkg, high_bkg)
    mask = np.invert(mask)

    # Bitwise-AND mask and original image
    objects = cv2.bitwise_and(frame,frame, mask= mask)

    hsv = cv2.cvtColor(objects, cv2.COLOR_BGR2HSV)

    # mask the yellow balls
    mask = cv2.inRange(hsv, low_yellow, high_yellow)

    yellows = cv2.bitwise_and(objects, objects, mask=mask)

    # find the biggest cloud of 1's in the yellow mask
    biggest_cloud = []
    biggest_count = 0

    image = mask / 255.

    while len(np.where(image == 1)[0]) > biggest_count:
        loc = np.where(image == 1)
        y = loc[0][0]
        x = loc[1][0]
        count, cloud = flood_fill(image, y, x, 2)
        if count > biggest_count:
            print count
            biggest_count = count
            biggest_cloud = cloud

    print biggest_cloud
    print biggest_count

    cv2.imwrite('mask.jpg', mask)
    cv2.imwrite('yellows.jpg', yellows)
    cv2.imwrite('frame.jpg', frame)

    return

#####################################
# ENTRY FUNCTIONS
#####################################
def findAndDrawGiraffeBody(imgPath):
    printLogMsg("Running giraffe body detection...")
    # imgResult,confidence,bboxes = findAndDrawGiraffeBodyImpl(imgPath)    
    return imgResult,confidence,bboxes


#find faces using Project Oxford
#PLEASE DO NOT MODIFY THIS FUNCTION
def findAndDrawFaces(imgPath, subscriptionKey = projectOxfordKey):
    printLogMsg("Calling project Oxford Face API...")
    jsonString = "[]"
    time.sleep(1)
    #jsonString = callFaceDetectionAPI(imgPath, subscriptionKey)
    #jsonString = '[{"faceId":"aa593c2e-f82e-4325-b694-90cd1bc86469","faceRectangle":{"top":172,"left":275,"width":164,"height":164},"faceLandmarks":{"pupilLeft":{"x":323.7,"y":218.1},"pupilRight":{"x":392.9,"y":213.2},"noseTip":{"x":337.6,"y":259.7},"mouthLeft":{"x":333.3,"y":295.5},"mouthRight":{"x":396.8,"y":290.6},"eyebrowLeftOuter":{"x":302.7,"y":200.1},"eyebrowLeftInner":{"x":333.4,"y":200.0},"eyeLeftOuter":{"x":314.0,"y":219.1},"eyeLeftTop":{"x":323.0,"y":213.3},"eyeLeftBottom":{"x":323.8,"y":223.4},"eyeLeftInner":{"x":335.1,"y":218.2},"eyebrowRightInner":{"x":366.4,"y":196.2},"eyebrowRightOuter":{"x":426.1,"y":198.6},"eyeRightInner":{"x":382.4,"y":214.5},"eyeRightTop":{"x":393.8,"y":208.9},"eyeRightBottom":{"x":394.4,"y":218.0},"eyeRightOuter":{"x":404.3,"y":212.7},"noseRootLeft":{"x":341.4,"y":218.0},"noseRootRight":{"x":362.5,"y":217.3},"noseLeftAlarTop":{"x":331.9,"y":246.7},"noseRightAlarTop":{"x":363.4,"y":246.5},"noseLeftAlarOutTip":{"x":328.5,"y":261.1},"noseRightAlarOutTip":{"x":374.9,"y":262.4},"upperLipTop":{"x":352.0,"y":289.1},"upperLipBottom":{"x":351.7,"y":297.1},"underLipTop":{"x":354.0,"y":300.1},"underLipBottom":{"x":354.5,"y":309.4}},"attributes":{"headPose":{"pitch":0.0,"roll":-0.1,"yaw":-30.9},"gender":"male","age":40}}]'
    
    printLogMsg("Drawing detected faces...")
    img = Image.open(imgPath)
    faceInfos = parseFaceJson(jsonString)
    drawFaceLandmarks(img, faceInfos)
    img = imconvertPil2Cv(img)
    text = "nrFacesFound={0}".format(len(faceInfos))
    return img,text