from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from flask_cors import CORS
from flask import request
from PIL import Image
import numpy as np
import base64
import flask
import cv2
import os
import io

app = flask.Flask(__name__)

CORS(app)

@app.route('/', methods=['GET'])
def index():
    return {"test": "Welcome to the Face Mask Detection Application"}

@app.route('/camera', methods=['POST'])
def postMethod():

    # fetching the json data
    data = request.get_json()

    # seperating the keys, values pair from data recived from angular
    keys, values = zip(*data.items())

    # taking the reqiured value that contain img data
    img = values[0]

    # seperating img data again
    keys, values = zip(*img.items())

    # replaceing redundant data
    newStr = values[2].replace("data:image/jpeg;base64,", "")

    # converting the string to base64 to img and saving it for processing
    save_img(newStr)

    result, image = detect_mask_in_image()

    with open("ScannedImage.jpg", "rb") as output:
        image = base64.b64encode(output.read())
    image = str(image)
    # print(image)
    # print(type(image))

    image = image.replace("b'", "data:image/jpg;base64,")
    image = image.replace("'", "")
    # print(image)

    d = { '_imageAsDataUrl': image }

    try:
        os.remove("ScannedImage.jpg")
        os.remove("checkForMask.jpg")
    except:
        pass

    return {"result": result,
            "output": image}

def save_img(checkForMask):
    imgdata = base64.b64decode(checkForMask)
    filename = 'checkForMask.jpg'
    with open(filename, 'wb') as f:
        f.write(imgdata)

def detect_mask_in_image():

    # print("\033[2;32;m[INFO] loading face detector model...")

    prototxtPath = os.path.sep.join(["face_detector", "deploy.prototxt"])

    weightsPath = os.path.sep.join(["face_detector",
                                    "res10_300x300_ssd_iter_140000.caffemodel"])
    net = cv2.dnn.readNet(prototxtPath, weightsPath)

    # load the face mask detector model from disk
    # print("\033[2;32;m[INFO] loading face mask detector model...")
    model = load_model("mask_detector.model")

    # load the input image from disk, clone it, and grab the image spatial dimensions
    image = cv2.imread("checkForMask.jpg")
    (h, w) = image.shape[:2]

    # construct a blob from the image
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    # print("\033[2;32;m[INFO] computing face detections...")
    net.setInput(blob)
    detections = net.forward()
    result = str

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = image[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            # pass the face through the model to determine if the face
            # has a mask or not
            (mask, withoutMask) = model.predict(face)[0]

            # determine the class label and color we'll use to draw
            # the bounding box and text
            label = "Mask Detected" if mask > withoutMask else "Mask Not Detected"
            color = (80, 200, 80) if label == "Mask Detected" else (15, 15, 180)

            result = label

            # include the probability in the label
            label = "{}".format(label)

            # display the label and bounding box rectangle on the output
            # frame
            cv2.putText(image, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)
            cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

    cv2.imwrite('ScannedImage.jpg', image)
    return result, image

if __name__ == '__main__':
    app.run(debug=True)


"""
request.is_json -> Checks if data from PostMan or anywhere is in JSON or not. 
If not it returns False

request.get_json() -> Gets contents in JSON
If not in JSON, prints 'None'
Else prints the input JSON

"""