import random

import cv2
import os

import numpy as np

from FaceRecognition.FaceDetector import *
from FaceRecognition.VideoCamera import *


def detect_face(url=0, show_image=False, saved_parts=False, saved_complete=False):
    # https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(url)
    ret, frame = cap.read()
    cap.release()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    scale_factor = 1.3
    min_neighbor = 5
    min_size = (130, 130)
    biggest_only = True

    flags = cv2.CASCADE_FIND_BIGGEST_OBJECT | cv2.CASCADE_DO_ROUGH_SEARCH if biggest_only else cv2.CASCADE_SCALE_IMAGE

    faces_coordinates = face_cascade.detectMultiScale(frame, scaleFactor=scale_factor, minNeighbors=min_neighbor,
                                                      minSize=min_size, flags=flags)

    print("Type: {}".format(type(faces_coordinates)))
    print("Length: {}".format(len(faces_coordinates)))


def detect_face_video(url=0, show_image=False, saved_parts=False, saved_complete=False):
    video = VideoCamera(url)
    detector = FaceDetector('haarcascade_frontalface_default.xml')

    while True:
        frame = video.get_frame()
        faces_coordinates = detector.detect(frame)
        for (top, right, bottom, left) in faces_coordinates:
            cv2.rectangle(frame, (top, right), (top + bottom, right + left), (255, 255, 255), 2)

        cv2.imshow("live!!!", frame)

        if cv2.waitKey(1) & 0xff == 27:
            break

    video.__del__()
    cv2.destroyAllWindows()


def detect_and_cut_face(url=0, show_image=False, saved_parts=False, saved_complete=False):
    video = VideoCamera(url)
    detector = FaceDetector('haarcascade_frontalface_default.xml')

    while True:
        frame = video.get_frame()
        faces_coordinates = detector.detect(frame)
        if len(faces_coordinates):
            faces = cut_faces(frame, faces_coordinates)
            faces = normalize_intensity(faces)
            faces = resize(faces)
            cv2.imshow("live!!!", faces[0])

        if cv2.waitKey(1) & 0xff == 27:
            break

    video.__del__()
    cv2.destroyAllWindows()


def cut_faces(image, faces_coordinates):
    faces = []

    for (top, right, bottom, left) in faces_coordinates:
        bottom_rm = int(0.2 * bottom / 2)
        faces.append(image[right:right + left, top + bottom_rm: top + bottom - bottom_rm])
    return faces

