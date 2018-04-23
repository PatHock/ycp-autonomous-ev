"""
object_detection.py

"""
# https://towardsdatascience.com/finding-lane-lines-on-the-road-30cf016a1165
# Identify yellow and white road markings
# mask out anything above the horizon line
# identify the edge of the road and mask out objects outside of the road

import cv2
import numpy as np

# frame = cv2.imread('road_country.jpg', 1)
# if frame is None:
#     raise Exception("could not load image !")

cap = cv2.VideoCapture('roadvideo.mp4')
if not cap.isOpened():
    print "Couldn't load video :("


def hough_transform(image):
    min_line_length = 25
    max_line_gap = 250
    rho = 5
    threshold = 100
    theta = np.pi / 180
    return cv2.HoughLinesP(image, rho=rho, theta=theta, threshold=threshold,
                           minLineLength=min_line_length, maxLineGap=max_line_gap)


def mask_color(image):
    # define range of color white in HSV
    lower_white = np.array([0, 0, 120])
    upper_white = np.array([255, 25, 255])

    # define range of color yellow in HSV
    lower_yellow = np.array([15, 100, 80])
    upper_yellow = np.array([45, 255, 255])

    # Convert BGR to HSV
    img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    img = cv2.GaussianBlur(img, (7, 7), 0)

    # Color masks for yellow, white (for lanes)
    mask_yellow = cv2.inRange(img, lower_yellow, upper_yellow)
    mask_white = cv2.inRange(img, lower_white, upper_white)

    return cv2.bitwise_and(img, img, mask=(mask_white + mask_yellow))


def mask_roi_overlay(image, mask_bounds_arr):
    # video is 1280*720
    # create a mask to ignore parts of the image that are not the road

    mask_bounds_arr = mask_bounds_arr.reshape((-1, 1, 2))
    cv2.polylines(image, [mask_bounds_arr], True, (255, 0, 0), 3)  # Show region of interest on output
    return image


def apply_perspective_transform(image):
    dst = np.float32([[0, (TOP_HEIGHT - BOTTOM_HEIGHT)*height], [0, 0], [BOTTOM_WIDTH*width, 0], [BOTTOM_WIDTH * width, (TOP_HEIGHT-BOTTOM_HEIGHT)*height]])
    src = roiBounds.astype(np.float32, copy=False)
    m = cv2.getPerspectiveTransform(src, dst)

    return cv2.warpPerspective(image, m, (int(BOTTOM_WIDTH * width), int((TOP_HEIGHT - BOTTOM_HEIGHT) * height)))

def undo_perspective_transform(image):
    # dst = np.float32([[0, height - 1], [0, 0], [width - 1, 0],
    #                   [width - 1, height - 1]])
    dst = np.float32([[0, (TOP_HEIGHT - BOTTOM_HEIGHT)*height], [0, 0], [BOTTOM_WIDTH*width, 0], [BOTTOM_WIDTH * width, (TOP_HEIGHT-BOTTOM_HEIGHT)*height]])
    src = roiBounds.astype(np.float32, copy=False)
    m = cv2.getPerspectiveTransform(src, dst)
    m = np.linalg.inv(m)

    return cv2.warpPerspective(image, m, (width, height))



def display_images():
    cv2.imshow('frame', frameRoiOverlay)
    # cv2.imshow('lines', frameLinesHough)
    cv2.imshow('Canny', frameCanny)
    cv2.imshow('transform', frameTransPersp)


def canny_edge_det(image):
    threshold_1 = 100
    threshold_2 = 200
    aperture_size = 3

    return cv2.Canny(image=image, threshold1=threshold_1,
                     threshold2=threshold_2, apertureSize=aperture_size)


while True:

    # Take each frame
    ret, frame = cap.read()
    if ret is True:
        height = int(frame.shape[0])
        width = int(frame.shape[1])

        # ROI trapezoid points
        HORIZ_OFFSET = 0.01
        BOTTOM_HEIGHT = 0.05
        BOTTOM_WIDTH = 0.85
        TOP_HEIGHT = 0.38
        TOP_WIDTH = 0.10

        BOTTOM_LEFT = [width * (0.5 * (1 - BOTTOM_WIDTH) + HORIZ_OFFSET), (1 - BOTTOM_HEIGHT) * height]
        BOTTOM_RIGHT = [width * (0.5 * (1 + BOTTOM_WIDTH) + HORIZ_OFFSET), (1 - BOTTOM_HEIGHT) * height]
        TOP_LEFT = [width * (0.5 * (1 - TOP_WIDTH) + HORIZ_OFFSET), (1 - TOP_HEIGHT) * height]
        TOP_RIGHT = [width * (0.5 * (1 + TOP_WIDTH) + HORIZ_OFFSET), (1 - TOP_HEIGHT) * height]

        roiBounds = np.array([BOTTOM_LEFT, TOP_LEFT, TOP_RIGHT, BOTTOM_RIGHT], np.int32)

        frameColorMasked = mask_color(frame)

        frameRoiOverlay = mask_roi_overlay(frame, roiBounds)

        frameTransPersp = apply_perspective_transform(frameColorMasked)

        frameCanny = canny_edge_det(frameTransPersp)
        grayscale = cv2.cvtColor(cv2.cvtColor(frameTransPersp, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2GRAY)

        lines = hough_transform(frameCanny)

        frameLinesHough = np.zeros(frameTransPersp.shape)

        if lines is not None:
            for i in range(0, len(lines)):
                for x1, y1, x2, y2 in lines[i]:
                    dx = x2 - x1
                    dy = y2 - y1
                    theta = np.arctan2(dy, dx) * 180 / np.pi
                    if 30 < abs(theta) < 80:
                        cv2.line(frameLinesHough, (x1, y1), (x2, y2), (0, 0, 255), 20)

        test = undo_perspective_transform(frameLinesHough)

        frameRoiOverlay = cv2.addWeighted(frameRoiOverlay, 0.5, test.astype(np.uint8, copy=False), 0.5, 0)


        display_images()

        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
