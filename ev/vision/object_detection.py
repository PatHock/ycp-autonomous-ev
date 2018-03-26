"""
object_detection.py

"""
#https://towardsdatascience.com/finding-lane-lines-on-the-road-30cf016a1165
# Identify yellow and white road markings
# mask out anything above the horizon line
# identify the edge of the road and mask out objects outside of the road

import cv2
import numpy as np

# img = cv2.imread('road_country.jpg', 1)
# if img is None:
#     raise Exception("could not load image !")

cap = cv2.VideoCapture('solidWhiteRight.mp4')
if not cap.isOpened():
    print "Couldn't load video :("

# define range of color white in HSV
lower_white = np.array([0, 0, 120])
upper_white = np.array([255, 15, 255])

# define range of color yellow in HSV
lower_yellow = np.array([20, 130, 100])
upper_yellow = np.array([40, 255, 255])


while True:

    # Take each frame
    ret, frame = cap.read()
    if ret is True:

        # Convert BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gaussian_hsv = cv2.GaussianBlur(hsv, (5, 5), 0)

        # Threshold the HSV image
        maskYellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        maskWhite = cv2.inRange(hsv, lower_white, upper_white)

        maskYellowGaussian = cv2.inRange(gaussian_hsv, lower_yellow, upper_yellow)
        maskWhiteGaussian = cv2.inRange(gaussian_hsv, lower_white, upper_white)

        # Bitwise-AND mask and original image
        # resWhite = cv2.bitwise_and(frame, frame, mask=maskWhite)
        # resYellow = cv2.bitwise_and(frame, frame, mask=maskYellow)

        cv2.imshow('frame', frame)
        cv2.imshow('maskWhite', maskWhite)
        cv2.imshow('gaussian white mask', maskWhiteGaussian)
        cv2.imshow(' gaussian yellow mask', maskYellowGaussian)

        k = cv2.waitKey(50) & 0xFF
        if k == 27:
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
