# import the necessary packages
import numpy as np
import argparse
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to input image")
ap.add_argument("-p", "--prototxt", required=True,
                help="path to Caffe 'deploy' prototxt file")
#To create a Caffe model you need to define the model architecture in a protocol buffer definition file (prototxt).

#The model well be using in this blog post is a
#Caffe version of the original TensorFlow implementation by Howard et al. and was trained by chuanqi305
ap.add_argument("-m", "--model", required=True,
                help="path to Caffe pre-trained model")
#The MobileNet SSD was first trained on the COCO dataset
# (Common Objects in Context) and was then fine-tuned on PASCAL VOC reaching 72.7% mAP (mean average precision)
ap.add_argument("-c", "--confidence", type=float, default=0.2,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))#mekedi wenne 0th(include) 255th(exclude)
#  athara random numbers thiyana array ekak hadana eka..meke size eka dila thiyenne ilakkam dekakin enisa 2d array ekak hadenne
#classes eke thyana awayawa ganata samata awayawa 3k athi tuples thiyana array ekak.awayawa 0-255 random floats
# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])#dnn eka oni wenne caffe framework eka load krnna
# caffe= GoogLeNet trained network from Caffe model zoo.
#meke prototxt ekakui(architecture eka thiyenne meeke) model ekakui(meka train krla thyenne object detect krnna) denna oni

#meken return karanne net object ekak

# load the input image and construct an input blob for the image
# by resizing to a fixed 300x300 pixels and then normalizing it
# (note: normalization is done via the authors of the MobileNet SSD
# implementation)

image = cv2.imread(args["image"])
(h, w) = image.shape[:2]#[Shape of image is accessed by img.shape. It returns a tuple of number
# of rows, columns and channels (if image is color):
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843,
	(300, 300), 127.5)

# pass the blob through the network and obtain the detections and
# predictions
print("[INFO] computing object detections...")
net.setInput(blob)
detections = net.forward()#forward method eken return wennet blob ekak= blob for first output of specified layer.
b= detections.shape[2]
print "number of people"+str( b)
# loop over the detections
for i in np.arange(0, detections.shape[2]):#shape eken return wenne array eke dimention eka shape[2] kiyanne eke 3weni attribute
    #  eka detect kragatta objects gana
    # extract the confidence (i.e., probability) associated with the
    # prediction
    confidence = detections[0, 0, i, 2]#2 wenne i kiyana object eke accuracy eka

    # filter out weak detections by ensuring the `confidence` is
    # greater than the minimum confidence
    if confidence > args["confidence"]:
        # extract the index of the class label from the `detections`,
        # then compute the (x, y)-coordinates of the bounding box for
        # the object
        idx = int(detections[0, 0, i, 1])#1 wenne i kiyana object eke wargaya class eke position eka
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])#3:7 wenne i kiyana object eke x1=3,y1=4,x2=5,y2=6 points 4
        (startX, startY, endX, endY) = box.astype("int")

        # display the prediction
        label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
        print("[INFO] {}".format(label))
        cv2.rectangle(image, (startX, startY), (endX, endY),#rectangle eka andeema
                      COLORS[idx], 2)
        y = startY - 15 if startY - 15 > 15 else startY + 15
        cv2.putText(image, label, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)


# show the output image
cv2.imshow("Output", image)
cv2.waitKey(0)