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
def findAndDrawGiraffeBodyImpl(imgPath):
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
    #init
    winSize = (targetWidth,targetHeight) #Detection window size. Align to block size and block stride.
    minRelHeight = minRelWidth * targetHeight / targetWidth
    learner = loadFromPickle(learnerPath)
    hogObj = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                               histogramNormType,L2HysThreshold,gammaCorrection,nlevels) #, signedGradients)
    hogObj.setSVMDetector(learner.coef_[0])

    #load image
    img = imread(imgPath)
    imgFilename = os.path.basename(imgPath)
    imresizeScale = 1.0 * imresizeWidth / imWidth(img)
    if imresizeScale > 1:  #never upscale image
        imresizeScale = 1.0
    img = imresize(img, imresizeScale)
    imgWidth, imgHeight = imWidthHeight(img)

    #run detection and reject small detections
    tStart = datetime.datetime.now()
    detections, scores = hogObj.detectMultiScale(img, **hogDetectParams)
    printLogMsg("Found {0} objects in image {1} (duration = {2} sec).".format(len(detections), imgFilename, (datetime.datetime.now() - tStart).total_seconds()))
    bboxes = [Bbox(x, y, x + w, y + h) for (x, y, w, h) in detections]
    keepIndices = [i for i,bbox in enumerate(bboxes) if bbox.width()> minRelWidth * imgWidth and bbox.height()> minRelHeight * imgHeight]
    bboxes = [bboxes[i] for i in keepIndices]
    scores = [scores[i] for i in keepIndices]
    printLogMsg("{0} detections left after removing boxes that are too small.".format(len(bboxes)))

    #only keep the highest scoring detection
    if len(bboxes) > 1:
        maxVal,indices = pbMax(scores)
        bboxes = [bboxes[indices[0]]]
        scores = [scores[indices[0]]]

    if len(bboxes) > 0:
        #classfier was trained on enlarged bboxes. need to downsize detection rectangle.
        for i in range(len(bboxes)):
            bbox = bboxes[i]
            centerPt = bbox.center()
            newWidth = bbox.width() / (bboxGrowScale)
            newHeight = bbox.height() / (bboxGrowScale)
            left   = centerPt[0] - 0.5 * newWidth
            right  = centerPt[0] + 0.5 * newWidth
            top    = centerPt[1] - 0.5 * newHeight
            bottom = centerPt[1] + 0.5 * newHeight
            bboxes[i] = Bbox(left, top, right, bottom)
        bboxes = [bbox.crop(imWidth, imHeight) for bbox in bboxes]
    else:
        bboxes = []

    #draw debug info
    if boVisualizeDetections:
        thickness = int(ceil((imgHeight + imgWidth) / 700.0))
        for bbox in bboxes:
            cv2.rectangle(img, tuple(bbox.leftTop()), tuple(bbox.rightBottom()), (255, 0, 0), 4 * thickness)

    #detection result
    if len(bboxes) > 0:
        bboxes = [bbox.scale(1.0/imresizeScale) for bbox in bboxes]
        bboxes = [bbox.crop(imWidth(imgPath), imHeight(imgPath)) for bbox in bboxes]
        score = numToString(scores[0][0].astype('float64'), 4)
    else:
        score = -1
    return img,score,bboxes




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