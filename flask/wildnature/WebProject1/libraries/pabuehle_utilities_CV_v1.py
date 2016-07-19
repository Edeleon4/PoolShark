# -*- coding: utf-8 -*-

###############################################################################
# Description:
#    This is a collection of utility / helper functions for computer vision tasks.
#    Note that most of these functions are not well tested, but are
#    prototyping implementations.
#
# Typical meaning of variable names:
#    pt                     = 2D point (column,row)
#    img                    = image
#    width,height (or w/h)  = image dimensions
#    bbox                   = bbox object
#    rect                   = rectangle (order: left, top, right, bottom)
#    angle                  = rotation angle in degree
#    scale                  = image up/downscaling factor
#
# TODO for v2:
# -
#
# NOTE:
# - All points are (column,row order). This is similar to OpenCV and other packages.
#   However, OpenCV indexes images as img[row,col] (but using OpenCVs Point class it's: img[Point(x,y)] )
# - all rotations are counter-clockwise, all angles are in degree
###############################################################################

import cv2, textwrap, pdb, copy, httplib, urllib, json, sys
from PIL import Image, ImageDraw, ImageColor, ImageFont, ExifTags
from PIL.ExifTags import TAGS
from skimage import data, exposure, color, transform, img_as_ubyte
from skimage.feature import hog
from math import *
import matplotlib.pyplot as plt
import numpy as np
sys.path.append('C:\Users\pabuehle\Desktop\PROJECTS\pythonLibrary')
from pabuehle_utilities_general_v0 import *










####################################
# CONSTANTS
####################################
COLORS = [ [255,0,0], [0,255,0], [0,0,255], [255,255,0], [255,0,255] ] #, (255,255,255), (0,0,0) )
for i in range(5):
    for dim in range(0,3):
        for s in (0.25, 0.5, 0.75):
            if COLORS[i][dim] != 0:
                newColor = copy.deepcopy(COLORS[i])
                newColor[dim] = int(round(newColor[dim] * s))
                COLORS.append(newColor)






####################################
# Image transformation
####################################
def imread(imgPath):
    if not os.path.exists(imgPath):
        "ERROR: image path does not exist."
        error
    img = cv2.imread(imgPath)
    TAGSinverted = {v: k for k, v in TAGS.items()}
    orientationExifId = TAGSinverted['Orientation']
    try:
        imageExifTags = Image.open(imgPath)._getexif()
    except:
        imageExifTags = None

    #rotate the image if orientation exif tag is present
    if imageExifTags != None and orientationExifId != None and orientationExifId in imageExifTags:
        orientation = imageExifTags[orientationExifId]
        #print "orientation = " + str(imageExifTags[orientationExifId])
        if orientation == 1 or orientation == 0:
            pass #no need to do anything
        elif orientation == 8:
            img = imrotate(img, 90)
        else:
            print "ERROR: orientation = " + str(orientation) + " not_supported!"
            error
    return img


def imresize(img, scale):
    return cv2.resize(img, (0,0), fx=scale, fy=scale)

def imresizeToSize(img, targetWidth, targetHeight):
    return cv2.resize(img, (targetWidth,targetHeight))

def imresizeMaxPixels(img, maxNrPixels):
    nrPixels = (img.shape[0] * img.shape[1])
    scale = min(1.0,  1.0 * maxNrPixels / nrPixels)
    if scale < 1:
        img = imresize(img, scale)
    return img, scale


def imresizeMaxDim(img, maxDim):
    scale = min(1.0, 1.0 * maxDim / max(img.shape[:2]))
    if scale < 1:
        img = imresize(img, scale)
    return img, scale


def imresizeAndPad(img, targetWidth, targetHeight, paddingValue = 0):
    scale = min(1.0 * targetHeight / imHeight(img), 1.0 * targetWidth / imWidth(img))
    imgScaled = cv2.resize(img, (0,0), fx=scale, fy=scale)
    #imgScaled = transform.rescale(img, scale)
    newImg = paddingValue * np.ones((targetHeight, targetWidth, img.ndim), img.dtype)
    newImg[:imgScaled.shape[0], :imgScaled.shape[1], :] = imgScaled
    return newImg


def imrotate(img, angle, centerPt = None):
    if centerPt == None:
        imgPil = imconvertCv2Pil(img)
        imgPil = imgPil.rotate(angle, expand = True)
        return imconvertPil2Cv(imgPil)
    else:
        error


def imRigidTransform(img, srcPts, dstPts):
    srcPts = np.array([srcPts], np.int)
    dstPts = np.array([dstPts], np.int)
    M = cv2.estimateRigidTransform(srcPts, dstPts, False)
    if transformation is not None:
        return cv2.warpAffine(img, M)
    else:
        return None


def imConcat(img1, img2):
    h1,w1 = img1.shape[:2]
    h2,w2 = img2.shape[:2]
    if len(img2.shape) == 3:
        newImg = np.zeros((max(h1, h2), w1+w2, img1.shape[2]), img1.dtype)
        newImg[0:h1,0:w1,:] = img1
        newImg[0:h2,w1:w1+w2,:] = img2
    else:
        newImg = np.zeros((max(h1, h2), w1+w2), img1.dtype)
        newImg[0:h1,0:w1] = img1
        newImg[0:h2,w1:w1+w2] = img2
    return newImg


def imconvertPil2Cv(pilImg):
    return imconvertPil2Numpy(pilImg)[:, :, ::-1]

def imconvertCv2Pil(img):
    #rgb = pilImg.convert('RGB')
    #return np.array(rgb).copy()[:, :, ::-1]
    cv2_im = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(cv2_im)
    return pil_im

def imconvertPil2Numpy(pilImg):
    rgb = pilImg.convert('RGB')
    return np.array(rgb).copy()

def imconvertSki2Cv(skiImg):
    return img_as_ubyte(skiImg).copy()

def imconvertCv2Numpy(img):
    (b,g,r) = cv2.split(img)
    return cv2.merge([r,g,b])







####################################
# Image info
####################################
def imWidth(input):
    return imWidthHeight(input)[0]


def imHeight(input):
    return imWidthHeight(input)[1]


def imWidthHeight(input):
    if type(input) is str or type(input) is unicode:
        width, height = Image.open(input).size #this does not load the full image
    else:
        width =  input.shape[1]
        height = input.shape[0]
    return width,height







####################################
# Visualization
####################################
def imshow(img, waitDuration=0, maxDim = None, windowName = 'img'):
    if isinstance(img, basestring): #test if 'img' is a string
        img = cv2.imread(img)
    if maxDim is not None:
        scaleVal = 1.0 * maxDim / max(img.shape[:2])
        if scaleVal < 1:
            img = imresize(img, scaleVal)
    cv2.imshow(windowName, img)
    cv2.waitKey(waitDuration)


def drawLine(img, pt1, pt2, color = (0, 255, 0), thickness = 2):
    cv2.line(img, tuple(np.int0(pt1)), tuple(np.int0(pt2)), color, thickness)


def drawLines(img, pt1s, pt2s, color = (0, 255, 0), thickness = 2):
    for pt1,pt2 in zip(pt1s,pt2s):
        drawLine(img, pt1, pt2, color, thickness)


def drawPolygon(img, pts, boCloseShape = False, color = (0, 255, 0), thickness = 2):
    for i in range(len(pts) - 1):
        drawLine(img, pts[i], pts[i+1], color = color, thickness = thickness)
    if boCloseShape:
        drawLine(img, pts[len(pts)-1], pts[0], color = color, thickness = thickness)


def drawRectangles(img, rects, color = (0, 255, 0), thickness = 2):
    for rect in rects:
        pt1 = tuple(ToIntegers(rect[0:2]))
        pt2 = tuple(ToIntegers(rect[2:]))
        cv2.rectangle(img, pt1, pt2, color, thickness)


def drawCircles(img, centerPts, radius, color = (0, 255, 0), thickness = 2):
    for centerPt in centerPts:
        centerPt = tuple(ToIntegers(centerPt))
        radius = int(round(radius))
        cv2.circle(img, centerPt, radius, color, thickness)


def drawCrossbar(img, pt):
    (x,y) = pt
    cv2.rectangle(img, (0, y), (x, y), (255, 255, 0), 1)
    cv2.rectangle(img, (x, 0), (x, y), (255, 255, 0), 1)
    cv2.rectangle(img, (img.shape[1],y), (x, y), (255, 255, 0), 1)
    cv2.rectangle(img, (x, img.shape[0]), (x, y), (255, 255, 0), 1)


def pilDrawText(img, pt, text, textWidth, color = (255,255,255), font = ImageFont.truetype("arial.ttf", 16)):
    draw = ImageDraw.Draw(img)
    lines = textwrap.wrap(text, width=textWidth)
    for line in lines:
        width, height = font.getsize(line)
        draw.text(pt, line, fill = color, font = font)
        textY += height


def pilDrawPoints(img, pts, color=(0,255,0), thickness=2):
    draw = ImageDraw.Draw(img)
    for pt in pts:
        (x,y) = pt
        draw.rectangle((x-thickness, y-thickness, x+thickness, y+thickness), fill=color)









####################################
# Bounding box and rectangle
####################################
class Bbox:
    MAX_VALID_DIM = 100000
    left = top = right = bottom = None

    def __init__(self, left, top, right, bottom):
        self.left   = int(round(float(left)))
        self.top    = int(round(float(top)))
        self.right  = int(round(float(right)))
        self.bottom = int(round(float(bottom)))
        self.standardize()

    def __str__(self):
        return ("Bbox object: left = {0}, top = {1}, right = {2}, bottom = {3}".format(self.left, self.top, self.right, self.bottom))

    def __repr__(self):
        return str(self)

    def setRect(self, rect):
        rect = [int(round(float(i))) for i in rect]
        self.left, self.top, self.right, self.bottom  = rect
        self.standardize()

    def rect(self):
        return [self.left, self.top, self.right, self.bottom]

    # def upperLeft(self):
    #     return [self.top, self.left]
    #
    # def bottomRight(self):
    #     return [self.bottom, self.right]

    def leftTop(self):
        return [self.left, self.top]

    def rightTop(self):
        return [self.right, self.top]

    def leftBottom(self):
        return [self.left, self.bottom]

    def rightBottom(self):
        return [self.right, self.bottom]

    def center(self):
        col = 0.5 * (self.left + self.right)
        row = 0.5 * (self.top + self.bottom)
        return (col, row)

    def width(self):
        width  = self.right - self.left + 1
        assert(width>=0)
        return width

    def height(self):
        height = self.bottom - self.top + 1
        assert(height>=0)
        return height

    def widthHeight(self):
        return self.width(), self.height()

    def aspectRatio(self):
        return 1.0 * self.width() / self.height()

    def surfaceArea(self):
        return self.width() * self.height()

    def getOverlapBbox(self, bbox):
        left1, top1, right1, bottom1 = self.rect()
        left2, top2, right2, bottom2 = bbox.rect()
        overlapLeft = max(left1, left2)
        overlapTop = max(top1, top2)
        overlapRight = min(right1, right2)
        overlapBottom = min(bottom1, bottom2)
        return Bbox(overlapLeft, overlapTop, overlapRight, overlapBottom)

    def standardize(self): #NOTE: every setter method should call standardize
        leftNew   = min(self.left, self.right)
        topNew    = min(self.top, self.bottom)
        rightNew  = max(self.left, self.right)
        bottomNew = max(self.top, self.bottom)
        self.left = leftNew
        self.top = topNew
        self.right = rightNew
        self.bottom = bottomNew

    def crop(self, maxWidth, maxHeight):
        leftNew   = min(max(self.left,   0), maxWidth)
        topNew    = min(max(self.top,    0), maxHeight)
        rightNew  = min(max(self.right,  0), maxWidth)
        bottomNew = min(max(self.bottom, 0), maxHeight)
        return Bbox(leftNew, topNew, rightNew, bottomNew)

    def scale(self, scale):
        left = self.left * scale
        top = self.top * scale
        right = self.right * scale
        bottom = self.bottom * scale
        return Bbox(left, top, right, bottom)

    def expandAspectRatio(self, targetAspectRatio):
        width, height = self.width(), self.height()
        if self.aspectRatio() > targetAspectRatio:
            targetHeight = round(width / targetAspectRatio)
            centerHeight = 0.5 * (self.top + self.bottom)
            newTop = round(centerHeight - 0.5 * targetHeight)
            newBottom = newTop + targetHeight - 1
            return Bbox(self.left, newTop, self.right, newBottom)
        else:
            targetWidth = round(height * targetAspectRatio)
            centerWidth = 0.5 * (self.left + self.right)
            newLeft = round(centerWidth - 0.5 * targetWidth)
            newRight = newLeft + targetWidth - 1
            return Bbox(newLeft, self.top, newRight, self.bottom)

    def rotate(self, angle, centerPt = []):
        notTestedYet
        if centerPt == []:
            centerPt = self.center();
        leftTopRot,rightTopRot, leftBottomRot, rightBottomRot = rectRotate(self.getRect(), angle, centerPt)
        return getEnclosingBbox([leftTopRot,rightTopRot, leftBottomRot, rightBottomRot])

    def transform(self, transformMat):
        if transformMat is None:
            return []
        else:
            pts = [self.leftTop(), self.rightTop(), self.rightBottom(), self.leftBottom()]
            pts = np.float32(pts).reshape(-1,1,2)
            transPts = cv2.transform(pts, transformMat)
            left, right = min(transPts[:,:,0]), max(transPts[:,:,0])
            top, bottom = min(transPts[:,:,1]), max(transPts[:,:,1])
            return Bbox(left, top, right, bottom)

    def isInsideRegion(self, maxWidth, maxHeight):
        if self.left >= 0 and self.top >= 0 and self.right < maxWidth and self.bottom < maxHeight:
            return True
        else:
            return False

    def isValid(self):
        if self.left>=self.right or self.top>=self.bottom:
            return False
        if min(self.rect()) < -self.MAX_VALID_DIM or max(self.rect()) > self.MAX_VALID_DIM:
            return False
        return True



def getEnclosingBbox(pts):
    left = top = float('inf')
    right = bottom = float('-inf')
    for pt in pts:
        left   = min(left,   pt[0])
        top    = min(top,    pt[1])
        right  = max(right,  pt[0])
        bottom = max(bottom, pt[1])
    return Bbox(left, top, right, bottom)


def ptRotate(pt, angle, centerPt = [0,0]):
    #while angle < 0: angle += 360
    #while angle >= 360: angle -= 360
    theta = - angle / 180.0 * pi
    ptRot = [0,0]
    ptRot[0] = cos(theta) * (pt[0]-centerPt[0]) - sin(theta) * (pt[1]-centerPt[1]) + centerPt[0]
    ptRot[1] = sin(theta) * (pt[0]-centerPt[0]) + cos(theta) * (pt[1]-centerPt[1]) + centerPt[1]
    return ptRot


def rectRotate(rect, angle, centerPt = []):
    left, top, right, bottom = rect
    if centerPt == []:
        centerPt = [0.5 * (left + right), 0.5 * (top + bottom)]
    leftTopRot     = ptRotate([left,top],     angle, centerPt)
    rightTopRot    = ptRotate([right,top],    angle, centerPt)
    leftBottomRot  = ptRotate([left,bottom],  angle, centerPt)
    rightBottomRot = ptRotate([right,bottom], angle, centerPt)
    return [leftTopRot, rightTopRot, leftBottomRot, rightBottomRot]


def bboxRotate(bbox, angle):
    centerPt = bbox.center()
    leftTopRot,rightTopRot, leftBottomRot, rightBottomRot = rectRotate(bbox.rect(), angle, centerPt)
    rotatedPolygon = [leftTopRot, rightTopRot, rightBottomRot, leftBottomRot]
    return rotatedPolygon


def bboxAndImgRotate(bbox, angle, img):
    rotatedPolygon = bboxRotate(bbox, angle)
    bboxRot = getEnclosingBbox(rotatedPolygon)
    imgRot = imrotate(img, angle, bbox.center())
    maskRot = np.array(np.ones(img.shape[:2]), np.uint8)
    maskRot = imrotate(maskRot, angle, bbox.center())
    left, top, right, bottom = bboxRot.rect()
    if bboxRot.isInsideRegion(*imWidthHeight(img)) and maskRot[top,left]==1 and maskRot[top,right]==1 and maskRot[bottom,left]==1 and maskRot[bottom,right]==1:
        bboxRotInsideOriginalImage = True;
    else:
        bboxRotInsideOriginalImage = False
        imshow(imgRot)
    return bboxRot, imgRot, bboxRotInsideOriginalImage, rotatedPolygon, maskRot


def bboxComputeOverlapVoc(bbox1, bbox2):
    surfaceRect1 = bbox1.surfaceArea()
    surfaceRect2 = bbox2.surfaceArea()
    surfaceOverlap = bbox1.getOverlapBbox(bbox2).surfaceArea()
    return max(0, 1.0 * surfaceOverlap / (surfaceRect1 + surfaceRect2 - surfaceOverlap))






####################################
# SIFT / SURF / ORB
# (mostly high-level functions)
####################################
def serializeKeypoints(keyPoints):
    return [[kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id] for kp in keyPoints]


def deserializeKeypoints(kpInfos):
    return [cv2.KeyPoint(x=kpInfo[0][0],y=kpInfo[0][1],_size=kpInfo[1], _angle=kpInfo[2], _response=kpInfo[3], _octave=kpInfo[4], _class_id=kpInfo[5]) for kpInfo in kpInfos]


def loadKeypointsAndDescriptors(featDir, imgFilename, keyLUT = None, descLUT = None):
    keyPointsPath = featDir + "/" + imgFilename[:-4] + ".keyPoints.tsv"
    descriptorsPath = featDir + "/" + imgFilename[:-4] + ".descriptors.tsv"
    if keyLUT is None or keyPointsPath not in keyLUT.keys():
        keyPointsSerialized = loadFromPickle(keyPointsPath)
        keyPoints = deserializeKeypoints(keyPointsSerialized)
        descriptors = loadFromPickle(descriptorsPath)
    elif keyLUT is not None and keyPointsPath in keyLUT.keys():
        keyPoints = keyLUT[keyPointsPath]
        descriptors = descLUT[descriptorsPath]
    elif keyLUT is not None and keyPointsPath not in keyLUT.keys():
        keyLUT[keyPointsPath] = keyPoints
        descLUT[descriptorsPath] = descriptors
    return keyPoints,descriptors


def orbDetectAndCompute(img, orbObject, orbImresizeScale, orbImresizeMaxDim):
    if orbImresizeScale != 1: #obtained better results for tiny objects if upscaling img
        scale = orbImresizeScale
        if max(img.shape) * scale > orbImresizeMaxDim:
            scale = orbImresizeMaxDim / max(img.shape)
        img = imresize(img, scale)
    else:
        scale = 1.0
    #mask = np.zeros(img.shape[:2], np.uint8)
    #mask[3*800:3*1200, 3*650:3*1400] = 1
    keyPoints, descriptors = orbObject.detectAndCompute(img, None)
    if scale != 1: #scale back to original image dimension
        for i in range(len(keyPoints)):
            keyPoints[i].pt = (round(keyPoints[i].pt[0]/scale), round(keyPoints[i].pt[1]/scale))
            keyPoints[i].size = round(keyPoints[i].size/scale)
    return keyPoints, descriptors


def findGoodMatches(matcher, test_desc, train_desc, train_kp, ratioThres, matchDistanceThres, maxNrMatchesPerLocation):
    goodMatches = []
    goodMatchesPts = dict()
    matches = matcher.knnMatch(test_desc, train_desc, k=2)
    for match in matches:
        if len(match) > 1:
            bestMatch, secondBestMatch = match
            if secondBestMatch.distance == 0:
                secondBestMatch.distance = 0.0001 #avoid division by zero errors
            if bestMatch.distance < matchDistanceThres and bestMatch.distance / secondBestMatch.distance < ratioThres:
                key = str(Round(train_kp[bestMatch.trainIdx].pt))
                if key not in goodMatchesPts:
                    goodMatchesPts[key] = 1
                if goodMatchesPts[key] <= maxNrMatchesPerLocation:
                    goodMatches.append(bestMatch)
                    goodMatchesPts[key] += 1
    return goodMatches


def estimateRigidTransform(srcPts, dstPts, inlierThreshold, outlierCoordScale = 1.0, fullAffine = False):
    srcPts = np.float32(srcPts).reshape(-1,1,2)
    dstPts = np.float32(dstPts).reshape(-1,1,2)
    M = cv2.estimateRigidTransform(srcPts, dstPts, fullAffine = fullAffine)
    print M
    if M is None:
        inlierMask = np.zeros(len(srcPts))
    else:
        inlierMask = []
        mappedPts = cv2.transform(srcPts, M)
        for mappedPt,dstPt in zip(mappedPts, dstPts):
            dist = np.linalg.norm(mappedPt/outlierCoordScale - dstPt/outlierCoordScale)
            inlierMask.append(int(dist < inlierThreshold))
        inlierMask = np.array(inlierMask)
    return M, inlierMask


def matchesGeometricVerification(goodMatches, train_kp, test_kp, outlierCoordScale, projectionDistanceThres):
    srcPts = [train_kp[m.trainIdx].pt for m in goodMatches]
    dstPts = [ test_kp[m.queryIdx].pt for m in goodMatches]
    transformMat, inlierMask = estimateRigidTransform(srcPts, dstPts, projectionDistanceThres, outlierCoordScale)
    nrInliers = sum(inlierMask)
    inlierRatio = 1.0 * nrInliers / len(goodMatches)
    return (nrInliers, inlierRatio, inlierMask, transformMat)


def visualizeMatchingResult(bbox, testImg, trainImg, test_kp, train_kp, goodMatches, inlierMask):
    trainImgScale = 3.0 #1.0
    if trainImg != [] and trainImgScale != 1.0:
        kpInfos = serializeKeypoints(train_kp)
        for i in range(len(kpInfos)):
            kpInfos[i][0] = (kpInfos[i][0][0]*trainImgScale, kpInfos[i][0][1]*trainImgScale)
        train_kp = deserializeKeypoints(kpInfos)
        trainImg = imresize(trainImg, trainImgScale)
        #for i in range(len(train_kp_scaled)):
        #    pt = train_kp_scaled[i].pt
        #    train_kp_scaled[i].pt = (pt[0] * trainImgScale, pt[1] * trainImgScale)
    if trainImg == []:
        newImg = testImg[:]
        lineThickness = 1 #max(2, int(ceil(max(testImg.shape) / 600.0)))
        drawCircles(newImg,  [x.pt for x in test_kp],  lineThickness, color = (255, 0, 0), thickness = lineThickness)
        #newImg  = cv2.drawKeypoints(newImg,  test_kp,  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    else:
        lineThickness = int(max(2, max(ceil(max(trainImg.shape) / 1000.0), int(max(testImg.shape) / 1000.0))))
        drawCircles(trainImg, [x.pt for x in train_kp], lineThickness, color = (255, 0, 0), thickness = lineThickness)
        drawCircles(testImg,  [x.pt for x in test_kp],  lineThickness, color = (255, 0, 0), thickness = lineThickness)
        #trainImg = cv2.drawKeypoints(trainImg, train_kp) #, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        #testImg  = cv2.drawKeypoints(testImg,  test_kp) #,  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        #draw all matches
        offsetX = trainImg.shape[1]
        newImg = imConcat(trainImg, testImg)
        for loopIndex,match in enumerate(goodMatches):
            posTrain = train_kp[match.trainIdx].pt
            posTest  = test_kp[match.queryIdx].pt
            posTest  = (posTest[0] + offsetX, posTest[1])
            if inlierMask[loopIndex]:
                lineColor = [0, 255, 0]
            else:
                lineColor = [0, 0, 255]
            drawLine(newImg, posTest, posTrain, lineColor, lineThickness)

        #draw detected bounding box
        if bbox != []:
            bboxDraw = bbox[:]
            bboxDraw[0] += offsetX
            bboxDraw[2] += offsetX
            drawRectangles(newImg, [bboxDraw], color = (0, 255, 0), thickness = 4* lineThickness)
    return newImg


def orbFindBestMatch(matcher, testImgWidth, testImgHeight, testKp, testDesc, orbTrainInfos, MIN_MATCH_COUNT, MATCH_RATIO_THRES, DISTANCE_THRES, MAX_NR_MATCHES_PER_LOCATION, PROJECTION_DISTANCE_THRESHOLD, MIN_INLIER_RATIO):
    bboxRect = []
    bestNrInliers = 0
    bestTrainImgInfo = ([], "", "", [])
    bestTransformInfo = ([], [], [], -1)

    #loop over all training images
    for trainDataIndex, orbTrainInfo in enumerate(orbTrainInfos):
        displayProgressBarPrompt(1.0 * trainDataIndex / len(orbTrainInfos))
        trainObjectName, trainImgFilename, trainKp, trainDesc, trainBbox = orbTrainInfo

        #find good matches
        if trainDesc is not None and trainDesc != []:
            goodMatches = findGoodMatches(matcher, testDesc, trainDesc, trainKp, MATCH_RATIO_THRES, DISTANCE_THRES, MAX_NR_MATCHES_PER_LOCATION)
            print "len(goodMatches) = " + str(len(goodMatches))

            #run geometric verification
            if len(goodMatches) > bestNrInliers and len(goodMatches) > MIN_MATCH_COUNT:
                nrInliers, inlierRatio, inlierMask, transformMat = matchesGeometricVerification(goodMatches, trainKp, testKp, 0.5 *(testImgWidth + testImgHeight), PROJECTION_DISTANCE_THRESHOLD)
                print "nrInliers = " + str(nrInliers)

                # remember best matching training image
                if nrInliers > bestNrInliers and (1.0 * nrInliers / len(goodMatches)) > MIN_INLIER_RATIO:
                    bestNrInliers = nrInliers
                    bestTrainImgInfo = [trainKp, trainObjectName, trainImgFilename, trainBbox]
                    bestTransformInfo = [goodMatches, inlierMask, transformMat, inlierRatio]

    #compute bounding box
    if bestNrInliers > 0:
        trainObjectName, trainImgFilename, trainBbox = bestTrainImgInfo[1:]
        transformMat = bestTransformInfo[2]
        bboxRect = [0, 0, trainBbox.width(), trainBbox.height()]
        bboxRect = Bbox(*bboxRect).transform(transformMat).crop(testImgWidth, testImgHeight).rect()
    return (bestNrInliers, bboxRect, bestTrainImgInfo, bestTransformInfo)



####################################
# Project Oxford Face API
####################################
def callFaceDetectionAPI(imgPath, subscriptionKey):
    #specify image from url of from file
    #body = "{'url':'https://c.s-microsoft.com/en-us/CMSImages/ImgMmnt_Ignite_768x768_EN_US.png?version=7b019640-7544-8e3d-06a2-43654307ae07'}"
    #headers = {'Content-type': 'application/json'}
    body = readBinaryFile(imgPath)
    headers = {'Content-type': 'application/octet-stream'}

    #call API
    conn = httplib.HTTPSConnection('api.projectoxford.ai')
    params = urllib.urlencode({'subscription-key': subscriptionKey,
        'analyzesFaceLandmarks': 'true',
        'analyzesAge': 'true',
        'analyzesGender': 'true',
        'analyzesHeadPose': 'true'})
    conn.request("POST", "/face/v0/detections?%s" % params, body, headers)
    response = conn.getresponse("")
    jsonSring = response.read()
    conn.close()
    return jsonSring


def parseFaceJson(jsonString):
    detectedFaces = json.loads(jsonString)
    faceInfos = []
    faceInfo = dict()
    for detectedFace in detectedFaces:
        faceInfo['gender']      = detectedFace['attributes']['gender']
        faceInfo['age']         = detectedFace['attributes']['age']
        faceInfo['yaw']         = detectedFace['attributes']['headPose']['yaw']
        faceInfo['roll']        = detectedFace['attributes']['headPose']['roll']
        faceInfo['pitch']       = detectedFace['attributes']['headPose']['pitch']
        faceInfo['faceid']      = detectedFace['faceId']
        faceInfo['width']       = detectedFace['faceRectangle']['width']
        faceInfo['top']         = detectedFace['faceRectangle']['top']
        faceInfo['height']      = detectedFace['faceRectangle']['height']
        faceInfo['left']        = detectedFace['faceRectangle']['left']
        faceInfo['faceLandmarks']   = detectedFace['faceLandmarks']
        faceInfo['eyeLeftOuter']    = detectedFace['faceLandmarks']['eyeLeftOuter']
        faceInfo['eyeLeftInner']    = detectedFace['faceLandmarks']['eyeLeftInner']
        faceInfo['eyeLeftTop']      = detectedFace['faceLandmarks']['eyeLeftTop']
        faceInfo['eyeLeftBottom']   = detectedFace['faceLandmarks']['eyeLeftBottom']
        faceInfo['eyeRightOuter']   = detectedFace['faceLandmarks']['eyeRightOuter']
        faceInfo['eyeRightInner']   = detectedFace['faceLandmarks']['eyeRightInner']
        faceInfo['eyeRightTop']     = detectedFace['faceLandmarks']['eyeRightTop']
        faceInfo['eyeRightBottom']  = detectedFace['faceLandmarks']['eyeRightBottom']
        faceInfo['upperLipBottom']  = detectedFace['faceLandmarks']['upperLipBottom']
        faceInfo['underLipBottom']  = detectedFace['faceLandmarks']['underLipBottom']
        faceInfo['mouthLeft']       = detectedFace['faceLandmarks']['mouthLeft']
        faceInfo['mouthRight']      = detectedFace['faceLandmarks']['mouthRight']
        faceInfos.append(faceInfo)
        #assert(detectedFace['attributes']['gender'] =='female' or detectedFace['attributes']['gender'] == 'male')
    return faceInfos


def getEyePosition(faceInfo):
    eyePosLeft_x =  round(0.25 * (faceInfo['eyeLeftOuter']['x']  + faceInfo['eyeLeftInner']['x']  + faceInfo['eyeLeftTop']['x']  + faceInfo['eyeLeftBottom']['x']))
    eyePosLeft_y =  round(0.25 * (faceInfo['eyeLeftOuter']['y']  + faceInfo['eyeLeftInner']['y']  + faceInfo['eyeLeftTop']['y']  + faceInfo['eyeLeftBottom']['y']))
    eyePosRight_x = round(0.25 * (faceInfo['eyeRightOuter']['x'] + faceInfo['eyeRightInner']['x'] + faceInfo['eyeRightTop']['x'] + faceInfo['eyeRightBottom']['x']))
    eyePosRight_y = round(0.25 * (faceInfo['eyeRightOuter']['y'] + faceInfo['eyeRightInner']['y'] + faceInfo['eyeRightTop']['y'] + faceInfo['eyeRightBottom']['y']))
    return ((eyePosLeft_x, eyePosLeft_y), (eyePosRight_x, eyePosRight_y))


def getMouthPosition(faceInfo):
    mouthPos_x = round(0.5 * (faceInfo['mouthLeft']['x'] + faceInfo['mouthRight']['x']) )
    mouthPos_y = round(0.5 * (faceInfo['mouthLeft']['y'] + faceInfo['mouthRight']['y']) )
    return (mouthPos_x, mouthPos_y)


def getFaceCoordinates(faceInfo):
    w, h = faceInfo['width'], faceInfo['height']
    faceLU = (faceInfo['left'], faceInfo['top'])
    faceRU = (faceLU[0] + w, faceLU[1])
    faceLB = (faceLU[0]    , faceLU[1] + h)
    faceRB = (faceLU[0] + w, faceLU[1] + h)
    return (faceLU, faceRU, faceLB, faceRB)


def drawFaceRectangle(img, faceInfos, color = (0,255,0)):
    for faceInfo in faceInfos:
        (faceLU, faceRU, faceLB, faceRB) = getFaceCoordinates(faceInfo)
        cv2.rectangle(img, faceLU, faceRB, color, 3)


def plotFaceRectangle(faceInfos):
    for faceInfo in faceInfos:
        (faceLU, faceRU, faceLB, faceRB) = getFaceCoordinates(faceInfo)
        plt.plot((faceLU[0], faceRU[0], faceRB[0], faceLB[0], faceLU[0]), (faceLU[1], faceRU[1], faceRB[1], faceLB[1], faceLU[1]),  'r-')


def drawFaceLandmarks(img, faceInfos):
    draw = ImageDraw.Draw(img)
    for faceInfo in faceInfos:
        for (key,value) in faceInfo['faceLandmarks'].items():
            x = value['x']
            y = value['y']
            draw.rectangle((x-2, y-2, x+2, y+2), fill=(255, 255, 0))


def plotFaceLandmarks(faceInfos):
    for faceInfo in faceInfos:
        for (key,value) in faceInfo['faceLandmarks'].items():
            x = value['x']
            y = value['y']
            plt.plot(x,y,"o")





####################################
# Feature computation
####################################
def skiComputeHOGs(imgPaths, boVisualize = False, targetWidth = [], targetHeight = [], orientations=8, pixels_per_cell=(16, 16)):
    hogs = []
    for index,imgPath in enumerate(imgPaths):
        displayProgressBarPrompt(1.0 * index / len(imgPaths))
        hogs.append(computeHOG(imgPath, boVisualize, targetWidth, targetHeight, orientations, pixels_per_cell))
    return hogs


def skiComputeHOG(imgOrPath, boVisualize = False, targetWidth = [], targetHeight = [],  orientations=8, pixels_per_cell=(16, 16)):
    if type(imgOrPath) == str:
        img = cv2.imread(imgOrPath)
    else:
        img = imgOrPath
    if targetWidth != [] or targetHeight != []:
        imgRescaled = imresizeAndPad(img, targetWidth, targetHeight)
    else:
        imgRescaled = img
    imgGray = cv2.cvtColor(imgRescaled, cv2.COLOR_BGR2GRAY)
    imgGray = imgGray.astype(float) / 256.0   #skiimage expects image in range [0,1]
    #assert(max(imgGray.ravel() <= 1))
    hog_out = hog(imgGray, orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=(1, 1), visualise=boVisualize)

    if not boVisualize:
        hogVec = hog_out
    else:
        hogVec, hog_image = hog_out
        plt.close()
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1.axis('off')
        ax1.imshow(img)
        ax1.set_title('Input image')
        ax3.axis('off')
        ax3.imshow(imgGray, cmap=plt.cm.gray)
        ax3.set_title('gray image')
        ax2.axis('off')
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.1))
        ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
        ax2.set_title('Histogram of Oriented Gradients')
        plt.show() #block=False
    return hogVec


def parseHogDescriptor(hogFeat, winSize, cellSize, nbins):
    assert(len(hogFeat) == (winSize[0]/8-1) * (winSize[1]/8-1) * (2*2) * nbins)
    nrCellsHoriz = winSize[0] / cellSize[0]
    nrCellsVert  = winSize[1] / cellSize[1]
    counters = np.zeros((nrCellsVert, nrCellsHoriz), np.float32)
    hogGradients = np.zeros((nrCellsVert, nrCellsHoriz, nbins), np.float32)

    #loop over all blocks (each one cell apart, and block consists of 2x2 cells)
    hogFeatIndex = 0;
    for blockX in range(nrCellsHoriz - 1):
        for blockY in range(nrCellsVert - 1):
            for cellNr in range(4):
                #compute cell index
                cellX, cellY = blockX, blockY
                if (cellNr == 1 or cellNr == 3): cellY+=1
                if (cellNr == 2 or cellNr == 3): cellX+=1

                for bin in range(nbins):
                    hogGradients[cellY][cellX][bin] += hogFeat[hogFeatIndex]
                    counters[cellY][cellX] += 1
                    hogFeatIndex += 1

    #compute average gradient strengths per cell x,y location
    for cellX in range(nrCellsHoriz):
        for cellY in range(nrCellsVert):
            for bin in range(nbins):
                hogGradients[cellY][cellX][bin] /= counters[cellY][cellX]
    return hogGradients


def drawHogDescriptor(img, hogGradients, cellSize, nbins, signedGradients = False, imgScale = 1.0, boNormalizeGradients = True):
    if signedGradients:
        anglePerBin = 2*np.pi / nbins
    else:
        anglePerBin = np.pi / nbins
    if boNormalizeGradients:
        gradientScale = max(hogGradients.ravel())
    else:
        gradientScale = 1.0

    #plot hog descriptor into image
    hogImg = imresize(img, imgScale)
    for cellX in range(len(hogGradients[0])):
        for cellY in range(len(hogGradients)):
            ptX = (cellX + 0.5) * cellSize[0] #cell center point
            ptY = (cellY + 0.5) * cellSize[1]
            rect = [ptX-cellSize[0]/2.0, ptY-cellSize[1]/2.0, ptX+cellSize[0]/2.0, ptY+cellSize[1]/2.0]
            rect = [f*imgScale for f in rect]
            drawRectangles(hogImg, [rect], color = (255, 0, 0), thickness = 1)

            for bin in range(nbins):
                currRad = (bin + 0.5) * anglePerBin
                dirVecX, dirVecY = cos(currRad), sin(currRad)
                currGradient = hogGradients[cellY][cellX][bin]
                vecLength = currGradient / gradientScale * max(cellSize)/2.0
                assert(currGradient >= 0)

                if signedGradients:
                    x1 = ptX * imgScale
                    y1 = ptY * imgScale
                else:
                    x1 = (ptX - dirVecX * vecLength) * imgScale
                    y1 = (ptY - dirVecY * vecLength) * imgScale
                x2 = (ptX + dirVecX * vecLength) * imgScale
                y2 = (ptY + dirVecY * vecLength) * imgScale
                drawLine(hogImg, [x1, y1], [x2, y2], color = (0, 255, 0), thickness = 2)
    return hogImg


def drawHogSvmWeights(svmWeights, winSize, cellSize, nbins = 9, signedGradients = False, imgScale = 1.0, boNormalizeGradients = True):
    parsedSvmWeights = parseHogDescriptor(svmWeights, winSize, cellSize, nbins)
    parsedSvmWeightsPos = np.fmax(parsedSvmWeights, 0)
    parsedSvmWeightsNeg = np.fmin(parsedSvmWeights, 0)
    svmPosHogImg = drawHogDescriptor(np.zeros((winSize[1],winSize[0],3)), abs(parsedSvmWeightsPos), cellSize, nbins, signedGradients, imgScale, boNormalizeGradients)
    svmNegHogImg = drawHogDescriptor(np.zeros((winSize[1],winSize[0],3)), abs(parsedSvmWeightsNeg), cellSize, nbins, signedGradients, imgScale, boNormalizeGradients)
    plt.close()
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.axis('off')
    ax1.imshow(255*svmPosHogImg)
    ax1.set_title('Positive SVM weights')
    ax2.axis('off')
    ax2.imshow(255*svmNegHogImg)
    ax2.set_title('Negative SVM weight')
    plt.draw()
    plt.show()






####################################
# TLC
####################################
def getTlcInput(labels, feats):
    table = []
    nrFeats = -1
    for i in range(len(labels)):
        items = [int(labels[i].tolist())]
        items += (feats[i].tolist())
        if nrFeats == -1:
            nrFeats = len(items)
        assert(len(items) == nrFeats)
        table.append(items)
    return table


def loadImagenetFeatures(featPaths, boShowProgressBar = False):
    feats = []
    for index,featPath in enumerate(featPaths):
        if boShowProgressBar:
            displayProgressBarPrompt(1.0 * index / len(featPaths), "Loading features ...")
        featString = readFile(featPath)[0]
        featString = featString.split("\t")
        featString = np.array(featString, np.float32)
        #featString = featString / np.linalg.norm(featString,2)
        feats.append(featString)
    return feats





####################################
# High-level helper functions
####################################
def evalBboxesOverlap(gtBboxes, bboxes, overlapThreshold):
    maxOverlaps = []
    for bbox in bboxes:
        overlaps = [bboxComputeOverlapVoc(bbox,gtBbox) for gtBbox in gtBboxes]
        maxOverlap = 0
        if len(overlaps) > 0:
            maxOverlap = max(overlaps)
        maxOverlaps.append(maxOverlap)
    gtMaxOverlaps = []
    for gtBbox in gtBboxes:
        overlaps = [bboxComputeOverlapVoc(bbox,gtBbox) for bbox in bboxes]
        maxOverlap = 0
        if len(overlaps) > 0:
            maxOverlap = max(overlaps)
        gtMaxOverlaps.append(maxOverlap)

    tp = fp = tn = fn = 0
    for overlap in maxOverlaps:
        if overlap > overlapThreshold:
            tp += 1
        else:
            fp += 1
    for overlap in gtMaxOverlaps:
        if overlap < overlapThreshold:
            fn += 1
    if len(gtBboxes)==0 and len(bboxes)==0:
        tn += 1
    return tp, fp, tn, fn, maxOverlaps, gtMaxOverlaps


def getCropsAroundBbox(img, bbox, borderWidth = 1.5, borderHeight = 1.5, minBboxNrPixels = -1, offsetsRel = [0], angles = [0], boZeroPad = False):
    imgCrops = []
    imgCropsInfo = []
    if bbox.surfaceArea() >= minBboxNrPixels:
        #get target crop size
        cropSizeWidth  =  round(max(bbox.widthHeight()) * borderWidth)
        cropSizeHeight =  round(max(bbox.widthHeight()) * borderHeight)

        #add sufficiently large border such that crop is never outside image (similar to zero-padding each crop)
        paddingSizeWidth = 0
        paddingSizeHeight = 0
        #imshow(img, maxDim=800)
        if boZeroPad:
            paddingSizeWidth  = round(1.1 * cropSizeWidth)
            paddingSizeHeight = round(1.1 * cropSizeHeight)
            imgPadded = np.zeros((img.shape[0] + 2*paddingSizeHeight, img.shape[1] + 2*paddingSizeWidth, img.shape[2]), img.dtype)
            imgPadded[paddingSizeHeight:paddingSizeHeight+imHeight(img), paddingSizeWidth:paddingSizeWidth+imWidth(img),:] = img
            img = imgPadded

        #increase bbox size to include neighboring pixels
        centerPt = bbox.center()
        left   = centerPt[0] - cropSizeWidth / 2.0  + paddingSizeWidth
        right  = centerPt[0] + cropSizeWidth / 2.0  + paddingSizeWidth
        top    = centerPt[1] - cropSizeHeight / 2.0 + paddingSizeHeight
        bottom = centerPt[1] + cropSizeHeight / 2.0 + paddingSizeHeight
        #bbox = Bbox(left, top, right, bottom)

        #get crops at different offsets
        offsetsWidth  = sorted(list(set([f * cropSizeWidth  for f in offsetsRel])))
        offsetsHeight = sorted(list(set([f * cropSizeHeight for f in offsetsRel])))
        for offsetWidth in offsetsWidth:
            for offsetHeight in offsetsHeight:
                for angle in angles:
                    #translate bbox
                    #(left, top, right, bottom) = bbox.rect()
                    currBbox = Bbox(left + offsetWidth, top + offsetHeight, right + offsetWidth, bottom + offsetHeight)
                    currBbox = currBbox.crop(*imWidthHeight(img))

                    #rotate bbox
                    imgRot = img
                    if angle != 0:
                        try:
                            currBbox, imgRot, bboxRotInsideOriginalImage, dummy, dummy = bboxAndImgRotate(currBbox, angle, img)
                        except:
                            bboxRotInsideOriginalImage = False
                        if not bboxRotInsideOriginalImage:
                            continue

                    #get image crop
                    (currLeft, currTop, currRight, currBottom) = currBbox.rect()
                    imgCrop = imgRot[currTop:currBottom, currLeft:currRight, :]
                    if boZeroPad and angle == 0: #sanity check
                        assert(imWidth(imgCrop) == cropSizeWidth and imHeight(imgCrop) == cropSizeHeight)
                    imgCrops.append(copy.deepcopy(imgCrop))
                    imgCropsInfo.append((offsetWidth, offsetHeight, angle))
    return imgCrops, imgCropsInfo







####################################
# Random
####################################
class ImgInfo:
    width = height = None
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def __str__(self):
        return ("Iminfo object: width = {0}, height = {1}".format(self.width, self.height))

    def __repr__(self):
        return str(self)


def ptClip(pt, maxWidth, maxHeight):
    pt = list(pt)
    pt[0] = max(pt[0], 0)
    pt[1] = max(pt[1], 0)
    pt[0] = min(pt[0], maxWidth)
    pt[1] = min(pt[1], maxHeight)
    return pt







