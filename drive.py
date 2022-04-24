# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

import asyncio
import base64
import json
import time
from io import BytesIO
from multiprocessing import Process, Queue

import cv2
import numpy as np
import websockets
from PIL import Image

def birdview_transform(img):
    """Apply bird-view transform to the image
    """
    IMAGE_H = 480
    IMAGE_W = 640
    src = np.float32([[0, IMAGE_H], [640, IMAGE_H], [0, IMAGE_H * 0.4], [IMAGE_W, IMAGE_H * 0.4]])
    dst = np.float32([[240, IMAGE_H], [640 - 240, IMAGE_H], [-160, 0], [IMAGE_W+160, 0]])
    M = cv2.getPerspectiveTransform(src, dst) # The transformation matrix
    warped_img = cv2.warpPerspective(img, M, (IMAGE_W, IMAGE_H)) # Image warping
    return warped_img

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * (3/5))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []

    for line in lines:
        line.reshape(4)
        x1, y1, x2, y2 = line.reshape(4)
        parameter = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameter[0]
        intercept = parameter[1]
        if x1 < 320:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    left_fit_average = np.median(left_fit, axis = 0)
    right_fit_average = np.median(right_fit, axis = 0)
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])


def crop_view(image):
    height = image.shape[0]
    left_border = 0 #220
    right_border = 640 #420
    top_border = 100
    polygons = np.array([[left_border, height], [right_border, height], [right_border, top_border], [left_border, top_border]])
    mask = np.zeros_like(image, dtype='uint8')
    cv2.fillPoly(mask, np.int32([polygons]), 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def canny(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny_img = cv2.Canny(blur, 50, 150)
    return canny_img

def display_lines(img, lines):
    lines_image = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(lines_image, (x1, y1), (x2, y2), (255, 0, 0), 5)
    return lines_image

def filter_image(img):
    mid_color = img[]
    threshold, thresh = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
def image_process(image):
    img = image.copy()

    canny_img = canny(img)

    birdview = birdview_transform(canny_img)
    birdview_image = birdview_transform(image.copy())

    cropped_image = crop_view(birdview)

    lines = cv2.HoughLinesP(cropped_image, 2, 1*np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)

    average_lines = average_slope_intercept(birdview_image, lines)

    lines_image = display_lines(birdview_image, lines)

    image_with_line = cv2.addWeighted(birdview_image, 0.8, lines_image, 1, 1)

    return image_with_line


# frame = cv2.imread("D:/Stupid Robot/testImg/ngu5.png")
# frame = cv2.resize(frame, (640, 480), interpolation = cv2.INTER_AREA)

# result = image_process(frame.copy())   
# cv2.imshow("result", result)

# # Show the plot of the image
# # plt.imshow(result)
# # plt.show()

# cv2.waitKey(0)
async def process_image(websocket, path):
    async for message in websocket:
        # Get image from simulation
        data = json.loads(message)
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        image = np.asarray(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
        # Prepare visualization image
        draw = image_process(image.copy())

        # Show the result to a window
        cv2.imshow("image", image)
        cv2.imshow("draw", draw)
        cv2.waitKey(1)


async def main():
    async with websockets.serve(process_image, "0.0.0.0", 4567, ping_interval=None):
        await asyncio.Future()  # run forever

if __name__ == '__main__':
    asyncio.run(main())

