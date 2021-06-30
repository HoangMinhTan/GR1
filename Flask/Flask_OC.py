from flask import Flask, render_template
from flask_cors import CORS, cross_origin
from flask import request, url_for, flash, redirect
from werkzeug.utils import secure_filename
import cv2 as cv
import numpy as np
import os


#Khoi tao Flask server
app = Flask(__name__)

UPLOAD_FOLDER = '/static'
#Apply Flask CORS
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


def object_counting(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #blur image
    blur = cv.blur(gray,(3,3))

    edge = cv.Canny(blur, 50, 300, 3)

    kernel = np.ones((22, 22), np.uint8)
    closing = cv.morphologyEx(edge, cv.MORPH_CLOSE, kernel)

    num_labels, labels_im = cv.connectedComponents(closing)

    _, contours, _ = cv.findContours(closing, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(img, contours, -1, (0, 255, 0), 2) # vẽ lại ảnh contour vào ảnh gốc
    number = 'Number of object: ' + str(num_labels-1)
    cv.putText(img, number, (5,25), cv.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv.imwrite(os.path.join('static', 'result.jpg'), img)
    return img

@app.route('/object_counting', methods=['POST'])
@cross_origin(origin='*')
def counting_process():
    #load image
    img = request.files['img'].read()
    np_img= np.frombuffer(img, np.uint8)
    img = cv.imdecode(np_img, cv.IMREAD_COLOR)
    img = object_counting(img)
    return render_template('success.html')

@app.route('/test', methods=['GET'])
@cross_origin(origin='*')
def test():
    return render_template('index.html')


@app.route('/', methods=['GET'])
@cross_origin(origin='*')
def main():
    return render_template('Flask_OC.html')

#Start Server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port='1111', debug=True)
