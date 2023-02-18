import cv2
import numpy as np
import os


def create():
    return cv2.VideoCapture(0)


def capture(camera):
    return camera.read()[1]


def destroy(camera):
    camera.release()
    cv2.destroyAllWindows()


def resize(image, ratio):
    return cv2.resize(image, dsize=None, fx=ratio, fy=ratio)


def transform(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([100, 50, 50]), np.array([120, 255, 255]))
    masked_hsv = cv2.bitwise_and(hsv, hsv, mask=mask)
    bgr = cv2.cvtColor(masked_hsv, cv2.COLOR_HSV2BGR)
    return bgr


def detect(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(gray, 2, 255, cv2.THRESH_BINARY)[1]

    hist = np.sum(binary, axis=1).cumsum()
    rows = np.where((hist > hist[-1] / 2))[0]

    return rows[0] if len(rows) != 0 else 0


def height(image):
    return image.shape[0]


def line(image, row):
    image[int(row)][:] = [255, 255, 255]


def speak(message):
    os.system("say '{}'".format(message))


def keyboard():
    return cv2.waitKey(1000)


ESC = 27


if __name__ == '__main__':
    camera = create()

    while True:
        image = capture(camera)

        image = resize(image, 0.2)

        image = transform(image)

        position = detect(image)
        threshold = height(image) / 3

        line(image, position)
        line(image, threshold)

        cv2.imshow('Capture', image)

        if position >= threshold:
            speak('Heads up, Eric!')

        if keyboard() == ESC:
            break

    destroy(camera)

