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


def normalize_intensity(images):
    images_norm = []
    for image in images:
        is_color = len(image.shape) == 3
        if is_color:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        images_norm.append(cv2.equalizeHist(image))
    return images_norm


def resize(images, size=(50, 50)):
    images_norm = []
    for image in images:
        if image.shape < size:
            image_norm = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        else:
            image_norm = cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)
        images_norm.append(image_norm)
    return images_norm


def normalize_faces(image, faces_coordinates):
    faces = cut_faces(image, faces_coordinates)
    faces = normalize_intensity(faces)
    faces = resize(faces)
    return faces


def drew_rectangle(image, faces_coordinates):
    for (top, right, bottom, left) in faces_coordinates:
        bottom_rm = int(0.2 * bottom / 2)
        cv2.rectangle(image, (top + bottom_rm, right), (top + bottom - bottom_rm, right + left), (255, 255, 255), 2)


def build_data_set(url=0, win_name="live!"):
    video = VideoCamera(url)
    detector = FaceDetector('haarcascade_frontalface_default.xml')
    # name = raw_input('Person: ').lower()
    # folder = "people/" + name  # input name
    # cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(win_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    name = ""
    number_of_pic = 10

    while True:
        frame = video.get_frame()
        mask = np.zeros_like(frame)
        height, width, _ = frame.shape

        cv2.circle(mask, (int(width / 2), int(height / 2)), 200, (255, 255, 255), -1)
        frame = np.bitwise_and(frame, mask)

        cv2.putText(frame, "Hey, Press Enter key to start taking picture", (5, frame.shape[0] - 5),
                    cv2.FONT_HERSHEY_PLAIN, 1.3, (66, 53, 243), 2)
        cv2.putText(frame, "Can you enter your name: {}".format(name), (95, 95), cv2.FONT_HERSHEY_SIMPLEX, .7,
                    (155, 55, 55), 2)
        cv2.imshow(win_name, frame)

        key = cv2.waitKey(40) & 0xFF
        if key not in [8, 13, 27, 255]:
            name += chr(key)
        elif key == 8:
            name = name[:-1]
        elif key == 27:
            return
        elif key == 13:
            folder = "people/" + name.lower()  # input name
            if not os.path.exists(folder):
                break
            else:
                while True:
                    frame = video.get_frame()
                    mask = np.zeros_like(frame)
                    height, width, _ = frame.shape

                    cv2.circle(mask, (int(width / 2), int(height / 2)), 200, (255, 255, 255), -1)
                    frame = np.bitwise_and(frame, mask)

                    cv2.putText(frame, "The name {} is already in the system,".format(name), (95, 85),
                                cv2.FONT_HERSHEY_SIMPLEX, .7, (155, 55, 55), 2)
                    cv2.putText(frame, "would you like to add more pic?".format(name), (105, 115),
                                cv2.FONT_HERSHEY_SIMPLEX, .7, (155, 55, 55), 2)
                    cv2.putText(frame, "Press 'Y' key to start taking picture or 'N' ".format(name), (95, 145),
                                cv2.FONT_HERSHEY_SIMPLEX, .7, (66, 53, 243), 2)
                    cv2.putText(frame, "ESC to exit", (5, frame.shape[0] - 5), cv2.FONT_HERSHEY_PLAIN, 1.3,
                                (66, 53, 243), 2)
                    cv2.imshow(win_name, frame)

                    key = cv2.waitKey(40) & 0xFF
                    if key in [89, 78, 121, 110]:
                        break
                    elif key == 27:
                        return
                if key in [89, 121]:
                    break

    if not os.path.exists(folder):
        os.mkdir(folder)
    init_pic = len(os.listdir(folder))
    counter = init_pic
    timer = 0
    while counter < number_of_pic + init_pic:  # take 10 photo
        frame = video.get_frame()
        faces_coordinates = detector.detect(frame)
        if len(faces_coordinates) and timer % 700 == 50:
            faces = normalize_faces(frame, faces_coordinates)
            cv2.imwrite(folder + '/' + str(counter) + '.jpg', faces[0])
            counter += 1
        drew_rectangle(frame, faces_coordinates)
        cv2.putText(frame,
                    "Hey {}, You managed to take {} from {} pictures".format(name, counter, number_of_pic + init_pic),
                    (5, frame.shape[0] - 5), cv2.FONT_HERSHEY_PLAIN, 1.3, (266, 53, 43), 2)
        cv2.imshow(win_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

        timer += 50
    cv2.destroyAllWindows()


def collect_dat_set():
    images = []
    labels = []
    labels_dic = {}
    people = [person for person in os.listdir("people/")]
    for i, person in enumerate(people):
        labels_dic[i] = person
        for image in os.listdir("people/" + person):
            images.append(cv2.imread("people/" + person + '/' + image, 0))
            labels.append(i)
    return images, np.array(labels), labels_dic


def train_models():
    images, labels, labels_dic = collect_dat_set()

    rec_eig = cv2.face.EigenFaceRecognizer_create()
    if images:
        rec_eig.train(images, labels)

    rec_fisher = cv2.face.FisherFaceRecognizer_create()
    if len(set(labels)) > 1:
        rec_fisher.train(images, labels)

    rec_lbph = cv2.face.LBPHFaceRecognizer_create()
    if images:
        rec_lbph.train(images, labels)

    print("Models Trained Successful")
    return rec_eig, rec_fisher, rec_lbph, labels_dic


def prediction_3_models(url=0, show_image=False, saved_parts=False, saved_complete=False):
    images, labels, labels_dic = collect_dat_set()

    rec_eig = cv2.face.EigenFaceRecognizer_create()
    rec_eig.train(images, labels)

    rec_fisher = cv2.face.FisherFaceRecognizer_create()
    rec_fisher.train(images, labels)

    rec_lbph = cv2.face.LBPHFaceRecognizer_create()
    rec_lbph.train(images, labels)

    collector = cv2.face.StandardCollector_create()

    video = VideoCamera()
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

    pred, conf = rec_eig.predict(faces[0])
    print("Eigen faces -> Prediction: {}, Confidence: {}.".format(labels_dic[pred].capitalize(), conf))

    pred, conf = rec_fisher.predict(faces[0])
    print("Fisher faces -> Prediction: {}, Confidence: {}.".format(labels_dic[pred].capitalize(), conf))

    pred, conf = rec_lbph.predict(faces[0])
    print("Lbph faces -> Prediction: {}, Confidence: {}.".format(labels_dic[pred].capitalize(), conf))


def face_recognition(url=0, win_name="live!"):
    rec_eig, rec_fisher, rec_lbph, labels_dic = train_models()

    video = VideoCamera(url)
    detector = FaceDetector('haarcascade_frontalface_default.xml')
    cv2.namedWindow(win_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        frame = video.get_frame()
        faces_coordinates = detector.detect(frame)
        if len(faces_coordinates):
            faces = normalize_faces(frame, faces_coordinates)
            for i, face in enumerate(faces):
                pred, conf = rec_lbph.predict(faces[i])
                threshold = 110
                if conf > threshold:
                    print("Prediction: {}, Confidence: {}.".format(labels_dic[pred].capitalize(), conf))
                    cv2.putText(frame, labels_dic[pred].capitalize(),
                                (faces_coordinates[i][0], faces_coordinates[i][1] - 10), cv2.FONT_HERSHEY_PLAIN, 3,
                                getColor(i), 2)
            drew_rectangle(frame, faces_coordinates)
        cv2.putText(frame, "ESC to exit", (5, frame.shape[0] - 5), cv2.FONT_HERSHEY_PLAIN, 1.3, (66, 53, 243), 2)
        cv2.imshow(win_name, frame)

        if cv2.waitKey(1) & 0xff == 27:
            break

    video.__del__()
    cv2.destroyAllWindows()


def getColor(index):
    colors = [(76, 63, 243), (37, 210, 8), (137, 20, 210), (137, 148, 67)]
    # return colors[2]
    if index < 3:
        return colors[index]
    else:
        return colors[3]
