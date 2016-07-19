import os, sys
sys.path.append('.\libraries')
#from pabuehle_utilities_CV_v1 import *
from pabuehle_utilities_general_v0 import *
import requests, json, base64, datetime, time, threading
from flask import Flask, jsonify



####################################
# Parameters
####################################
#imgDir = "C:/Users/pabuehle/Desktop/apiCallImages/"
imgDir = "C:/workspace_BostonDSEng/teamTJ/wildnature/WebProject1/examples/"
apiUrl = "http://localhost:64054/api"
#apiUrl = "http://hendrick2.azurewebsites.net/faceDetectionAPI"

#no need to change these
boThreaded = False
outDir = imgDir + "apiOutput/"


####################################
# Helper Functions
####################################
def runWorker(imgIndex,imgFilename):
    print "\n*** Processing image " + str(imgIndex) + ": " + imgFilename + ".. ***"
    imgPath = imgDir + imgFilename

    #call API upload route with image
    print "Making API call.."
    tstart = datetime.datetime.now()
    files = {'file': (imgFilename, open(imgPath, 'rb'), 'image/jpeg', {'Expires': '0'})}
    rv = requests.post(apiUrl, files = files)
    durationMs = (datetime.datetime.now()-tstart).total_seconds() * 1000
    print "Done API call (time = {0}[ms])".format(durationMs)

    #parse output and save to disk
    print rv.content
    response = json.loads(rv.content)
    boGiraffeFound = response['boGiraffeFound'] == 'True'
    confidence = response['confidence']
    debugLog = response['debugLog']
    processingTimeMs = response['processingTimeMs']
    #resultImg = base64.b64decode(response['resultImg'])
    #writeBinaryFile(outDir + imgFilename[:-3] + "resultImg.jpg", resultImg);
    if boGiraffeFound:
        left = int(response['left'])
        top = int(response['top'])
        right = int(response['right'])
        bottom = int(response['bottom'])
        print "Giraffe location: left = {0}, top = {1}, right = {2}, bottom = {3}".format(left, top, right, bottom)

    print "Processing time = " + str(processingTimeMs)
    print "Overhead from API call = {0} [ms]".format(str(durationMs - float(processingTimeMs)))
    print "confidence = " + str(confidence)
    print "debugLog = "
    for s in debugLog.split('<br>'): print "   " + s
    print "\n*** DONE with image " + str(imgIndex) + ": " + imgFilename + ".. ***"



####################################
# Code
####################################
makeDirectory(outDir)
imgFilenames = getFilesInDirectory(imgDir, ".jpg")
tstartAll = datetime.datetime.now()

if boThreaded == False:
    for imgIndex,imgFilename in enumerate(imgFilenames):
        runWorker(imgIndex,imgFilename)

else:
    #start threads
    threads = []
    for imgIndex,imgFilename in enumerate(imgFilenames):
        t = threading.Thread(target=runWorker, name=None, args=(imgIndex,imgFilename))
        threads.append(t)
        t.start()
        #time.sleep(0.1)

    #wait until all threads are done
    for i in range(len(threads)):
        threads[i].join()

print "Done with all API calls (time = {0}[ms])".format((datetime.datetime.now()-tstartAll).total_seconds() * 1000)
