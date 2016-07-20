#################################################################
# This project uses Flask to run python code. See here for good 
# tutorial videos:
# https://mva.microsoft.com/en-US/training-courses/introduction-to-creating-websites-using-python-and-flask-8677?l=2otYpCH1_4104984382
#################################################################
import os, redis, webbrowser, base64, datetime, uuid
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
from werkzeug import secure_filename
from algo import *
#from azure.storage.blob import BlobService

#initialize the Flask application
app = Flask(__name__)

#Make the WSGI interface available at the top level so wfastcgi can get it.
wsgi_app = app.wsgi_app

#parameters
#maxNrAPICalls = 100
UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = set(['bmp', 'tif', 'tiff', 'png', 'jpg', 'jpeg', 'gif'])

#redis cache for persistent data storage
#redisCache = redis.StrictRedis(host = 'giraffeDetection.redis.cache.windows.net', port=6380, db=0, 
#                               password = 'TLmOY5oCUd43et9jENMi4ImV5uHqyq5EKKpqivcAi0Q=', ssl=True)



#####################################
# helper functions
#####################################
#For a given file, return whether it's an allowed type or not
def allowed_file(filename):
    return '.' in filename and filename.lower().rsplit('.', 1)[1] in ALLOWED_EXTENSIONS
        

def saveRequestFile(file):
    imgFilename = file.filename + str(uuid.uuid4()) + ".jpg"
    imgFilename = secure_filename(imgFilename)  
    imgPath = os.path.join(UPLOAD_FOLDER, imgFilename)
    file.save(imgPath)   
    return(imgFilename,imgPath)


def saveResultImg(imgFilename, img):
    resultsImgFilename = imgFilename[:-4] + ".result.jpg"
    resultsImgPath = os.path.join(UPLOAD_FOLDER, resultsImgFilename)
    cv2.imwrite(resultsImgPath, img)
    return resultsImgFilename, resultsImgPath


           

#####################################
# routes
#####################################
#home route
@app.route('/')
def index():
    return render_template('index.html')


#render the showImage html template
@app.route('/showImage/<filename64>_<text64>')
def renderShowImageHtml(filename64, text64):
    return render_template('showImage.html', filename=base64.b64decode(filename64), text=base64.b64decode(text64)) 


#send content of a file from a given directory to the client
@app.route('/sendFile/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


#show log and update constantly
@app.route('/showLog')
def showLog():
    text = "<br>".join(logMsgs)
    return '<head> ' + text + '<meta http-equiv="refresh" content="1"> </head>'


#show number of API calls
@app.route('/showNrApiCalls')
def showNrApiCalls():
    nrApiCalls = -1
    try:
        nrApiCalls = 5 #redisCache['nrApiCalls']
    except:
        nrApiCalls = -1
    msg = "NrApiCalls = " + str(nrApiCalls) + "."
    printLogMsg(msg);
    return msg


#route that handles the file upload
@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    if file and allowed_file(file.filename):
        printLogMsg("Uploading file to local directory...")
        imgFilename, imgPath = saveRequestFile(file)
        printLogMsg("Calling entry CV function...")
        img = imread(imgPath)
        poolTableDetectAndGetCoordinates(imgPath)
        os.remove(imgPath)

        printLogMsg("Generating result site...")
        #if len(bboxes)==0:
        text = "No best shot found"
        #else:
        #     text = "Found giraffe body with confidence {0} at location: {1}".format(confidence, str(bboxes[0])[13:])
        resultsImgFilename,_ = saveResultImg(imgFilename, img)   
        return redirect(url_for('renderShowImageHtml', filename64=base64.b64encode(resultsImgFilename), text64=base64.b64encode(text)))


#provide API interface for input/output
@app.route('/api', methods=['POST'])
def uploadFaceDetectionAPI():
    file = request.files['file']
    # redisCache.incr('nrApiCalls')
    printLogMsg("Total number of api calls: " + 5) #str(redisCache['nrApiCalls']))
    
    #if float(redisCache['nrApiCalls']) > maxNrAPICalls:
    #    printLogMsg("ERROR: already reached the maximum number of API Calls.");
    #    response = {'error': "already reached the maximum number of API Calls"}
    #    return jsonify(response)

    if file and allowed_file(file.filename):
        resetLog()
        tstart = datetime.datetime.now()
        printLogMsg("Uploading file {0} to local directory... ({1} ms)".format(file.filename,(datetime.datetime.now()-tstart).total_seconds() * 1000))
        imgFilename, imgPath = saveRequestFile(file)
        printLogMsg("Calling entry CV function... ({0} ms)".format((datetime.datetime.now()-tstart).total_seconds() * 1000))
        img,confidence,bboxes = findAndDrawGiraffeBody(imgPath)
        
        #prepare API return
        printLogMsg("Generating API response.. ({0} ms)".format((datetime.datetime.now()-tstart).total_seconds() * 1000))
        #_, resultsImgPath = saveResultImg(imgFilename, img) 
        #img64 = base64.b64encode(readBinaryFile(resultsImgPath))
        response = {
                'error':"",
                'boGiraffeFound': str(len(bboxes)>0),
                'confidence': confidence,
                'processingTimeMs': str((datetime.datetime.now()-tstart).total_seconds() * 1000)
                #'resultImg': img64
                #'debugLog': "<br>".join(logMsgs),
            }
        if len(bboxes)>0:
            printLogMsg("Giraffe found at location: " + str(bboxes[0]));
            response['left']   = str(bboxes[0].left)
            response['top']    = str(bboxes[0].top)
            response['right']  = str(bboxes[0].right)
            response['bottom'] = str(bboxes[0].bottom)
        return jsonify(response)


#reset number of API calls
#@app.route('/resetNrApiCalls')
#def resetNrApiCalls():
#    redisCache['nrApiCalls'] = 0
#    msg = "Variable nrApiCalls successfully set to " + str(redisCache['nrApiCalls']) + "."
#    printLogMsg(msg);
#    return msg + " This site can now be closed."


#main
if __name__ == '__main__':
    HOST = os.environ.get('SERVER_HOST', 'localhost')
    try:
        PORT = int(os.environ.get('SERVER_PORT', '5555'))
    except ValueError:
        PORT = 5555
    #if HOST == 'localhost':
    #    PORT = 58018

    app.run(HOST, PORT) #, threaded=True) #, debug=True,)
