import streamlit as st
from cv2 import cv2
import math
from matplotlib import pyplot as plt
from cv2 import threshold, drawContours
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imshow, imread
from skimage.color import rgb2hsv, hsv2rgb
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import git
import os
st.header("Fish Weight Prediction App")
st.text_input("Enter your Name: ", key="name")


#if st.checkbox('Show Training Dataframe'):
    #data
st.subheader("Please Enter the Image ")
#filename = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

import streamlit as st
import requests
import base64
import os
from git import Repo

# Use file uploader to get an image from the user
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
st.image(uploaded_file,caption="Your image")
import numpy as np
import cv2

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
if img is not None:
    cv2.imwrite('new_image.jpg', img)
    st.success('Image saved successfully!')

img = cv2.imread('new_image.jpg')

# Apply a Gaussian blur filter to the input image
blur = cv2.GaussianBlur(img, (51, 51), 0)

# Create a mask to highlight the foreground object
mask = cv2.inRange(blur, (0, 0, 0), (100, 100, 100))

# Apply the mask to the input image to obtain the foreground object
fg = cv2.bitwise_and(img, img, mask=mask)

# Invert the mask to highlight the background region
mask_inv = cv2.bitwise_not(mask)

# Apply the inverted mask to the blurred image to obtain the background
bg = cv2.bitwise_and(blur, blur, mask=mask_inv)

# Combine the foreground and background images
result = cv2.add(fg, bg)

# Display the result
cv2.imwrite('Result.jpg', result)
 

W = 600
oriimg = cv2.imread('Result.jpg')
height, width, depth = oriimg.shape
imgScale = W/width
newX,newY = oriimg.shape[1]*imgScale, oriimg.shape[0]*imgScale
newimg = cv2.resize(oriimg,(int(newX),int(newY)))
#cv2.imshow("Show by CV2",newimg)
#cv2.waitKey(0)
cv2.imwrite('resized.jpg',newimg)
img = cv2.imread('resized.jpg')
imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(imgray,40,60)

# load the dataset
df = pd.read_csv('Fish.csv')
df_clean = df.drop(columns=['Species','Length1'],)
# Splitting data into training and testing sets
x = df_clean.drop(columns='Weight')
y = df_clean['Weight']


# create polynomial features
poly = PolynomialFeatures(degree=11)
X_poly = poly.fit_transform(x)

# train linear regression model on polynomial features
model2 = LinearRegression().fit(X_poly, y)

def midpoint(ptA, ptB):
  return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

image = cv2.imread("resized.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (7, 7), 0)
cv2.imwrite('ref0.jpg', gray)

# perform edge detection, then perform a dilation + erosion to
# close gaps in between object edges

edged = cv2.Canny(gray, 30,40)
edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)
#cv2.imshow("Im", edged)
cv2.imwrite('ref1.jpg', edged)

# find contours in the edge map
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
  cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# sort the contours from left-to-right and initialize the
# 'pixels per metric' calibration variable
(cnts, _) = contours.sort_contours(cnts)
pixelsPerMetric = None

orig = image.copy()

# loop over the contours individually
for c in cnts:
  # if the contour is not sufficiently large, ignore it
  if cv2.contourArea(c) < 100:
    continue
 
  # compute the rotated bounding box of the contour
  box = cv2.minAreaRect(c)
  box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
  box = np.array(box, dtype="int")
 
  # order the points in the contour such that they appear
  # in top-left, top-right, bottom-right, and bottom-left
  # order, then draw the outline of the rotated bounding
  # box
  box = perspective.order_points(box)
  cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
  cv2.imwrite('ref2.jpg', cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2))
 
  # loop over the original points and draw them
  for (x, y) in box:
    cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

  # unpack the ordered bounding box, then compute the midpoint
  # between the top-left and top-right coordinates, followed by
  # the midpoint between bottom-left and bottom-right coordinates
  (tl, tr, br, bl) = box
  (tltrX, tltrY) = midpoint(tl, tr)
  (blbrX, blbrY) = midpoint(bl, br)
 
  # compute the midpoint between the top-left and top-right points,
  # followed by the midpoint between the top-righ and bottom-right
  (tlblX, tlblY) = midpoint(tl, bl)
  (trbrX, trbrY) = midpoint(tr, br)
 
  # draw the midpoints on the image
  cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
  cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
  cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
  cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
 
  # draw lines between the midpoints
  cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
    (255, 0, 255), 2)
  cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
    (255, 0, 255), 2)

  cv2.imwrite("ref3.jpg",orig)
  # compute the Euclidean distance between the midpoints
  dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
  dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
 
  # if the pixels per metric has not been initialized, then
  # compute it as the ratio of pixels to supplied metric
  # (in this case, inches)
  if pixelsPerMetric is None:
    pixelsPerMetric = dB / 1

  # compute the size of the object
  dimA = dA / pixelsPerMetric
  dimB = dB / pixelsPerMetric
  if dimA<dimB:
    dimA,dimB = dimB,dimA
  dimA = 2.54*dimA
  dimB = 2.54*dimB
  dimC = math.sqrt((dimA**2)+(dimB**2))
  input_str = "{},{},{},{}".format(dimA,dimC,dimB,1)
  input_list = input_str.split(',')
  input_arr = np.array(input_list).astype(float)

# Reshape the input array to a row vector
  input_arr = input_arr.reshape(1, -1)

# Transform input array into polynomial features
  input_poly = poly.transform(input_arr)

# Make prediction using the trained model
  weight = model2.predict(input_poly)
  
#  weight_poly=poly.fit_transform([dimA,dimC,dimB,1])
#  weight = model2.predict(weight_poly)

  #width = 1
  if weight[0]<0:
    weight = [4.0]
  st.write("Length: ",dimA,"Breadth :",dimB,"Cross : ",dimC,"Weight : ",weight[0])
 
  # draw the object sizes on the image
  cv2.putText(orig, "{:.1f}cm".format(dimB),
    (int(tltrX), int(tltrY)), cv2.FONT_HERSHEY_SIMPLEX,
    0.65, (0,0,139), 2)
  cv2.putText(orig, "{:.1f}cm".format(dimA),
    (int(trbrX-50), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
    0.65, (0,0,139), 2)
  cv2.putText(orig, "{:.1f}cm".format(dimC),
    (int(trbrX+10), int(trbrY-10)), cv2.FONT_HERSHEY_SIMPLEX,
    0.65, (0,0,139), 2)
  cv2.putText(orig, "{}g".format(weight),
    (int(trbrX-100), int(trbrY+50)), cv2.FONT_HERSHEY_SIMPLEX,
    0.5, (0,0,13),2)
  

  

  # show the output image
  #cv2.imshow("Image", orig)
  cv2.imwrite('Final.jpg', orig)                                                  
  #cv2.waitKey(0)

if st.button('Predict Fish Weight'):
    st.image(orig, caption='Weight and dimension predicted')
    
    #st.write(f"Your fish weight is: {np.squeeze(prediction, -1):.2f}g")

    st.write(f"Thank you {st.session_state.name}!")
    


