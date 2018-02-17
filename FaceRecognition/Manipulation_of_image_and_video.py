# mouse callback function
import cv2
import math

import numpy as np
from IPython.display import clear_output

def using_mask_to_uncover_img(url=0, name_video='name video'):
    def draw_circle_move(event, x, y, flags, param):
        global x_in, y_in
        if event == cv2.EVENT_LBUTTONDOWN:
            x_in = x
            y_in = y
        elif event == cv2.EVENT_LBUTTONUP:
            cv2.circle(mask, (int((x + x_in) / 2), int((y + y_in) / 2)),
                       int(math.sqrt((y - y_in) ** 2 + (x - x_in) ** 2) / 2), (255, 255, 255), -1)

    def draw_circle_static(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(mask, (x, y),
                       50, (255, 255, 255), -1)

    cv2.namedWindow(name_video)
    cv2.setMouseCallback(name_video, draw_circle_static)

    webcam = cv2.VideoCapture(url)
    _, frame = webcam.read()
    mask = np.zeros_like(frame)

    while True:
        _, frame = webcam.read()
        frame = np.bitwise_and(frame, mask)
        cv2.imshow('name video', frame)
        if cv2.waitKey(40) & 0xff == 27:
            break

    webcam.release()
    cv2.destroyAllWindows()


def reading_a_video(url=0, name_video='name video'):
    video = cv2.VideoCapture(url)

    try:
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            width = frame.shape[1]

            frame = frame[:, width/3:width, :]
            cv2.imshow(name_video, frame)
            clear_output(wait=True)
    except KeyboardInterrupt:
        print("video Interrupted")

    video.release()

def rectangle_and_typing(url=0, name_video='name video'):
    video = cv2.VideoCapture(url)
    cv2.namedWindow(name_video, cv2.WINDOW_AUTOSIZE)
    message = ""

    while video.isOpened():
        _, frame = video.read()
        cv2.rectangle(frame, (100, 100), (543, 400), (155,255,255), 2)
        cv2.putText(frame, message, (95, 95), cv2.FONT_HERSHEY_SIMPLEX, .7, (155,255,255), 2)

        cv2.imshow(name_video, frame)
        key = cv2.waitKey(40) & 0xFF
        if key not in [255, 27]:
            message += chr(key)
        elif key == 27:
            break

    video.release()
    cv2.destroyAllWindows()

def view_throw_circle(url=0, name_video='name video'):
    video = cv2.VideoCapture(url)
    cv2.namedWindow(name_video, cv2.WINDOW_AUTOSIZE)

    while video.isOpened():
        _, frame = video.read()
        mask = np.zeros_like(frame)
        height, width, _ = frame.shape

        cv2.circle(mask, (int(width/2), int(height/2)), 200, (255, 255, 255), -1)
        frame = np.bitwise_and(frame, mask)

        cv2.imshow(name_video, frame)
        if cv2.waitKey(40) & 0xff == 27:
            break

    video.release()
    cv2.destroyAllWindows()



