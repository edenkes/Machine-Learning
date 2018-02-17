import cv2
from PIL import Image


def start_detection_face(url: 0):
    # multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

    # https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

    cap = cv2.VideoCapture(url)

    while 1:
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]

            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        cv2.imshow('img', img)
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


def saved_detection_face(url):
    # https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

    cap = cv2.VideoCapture(url)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

    while (cap.isOpened()):
        ret, img = cap.read()
        if ret == True:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = img[y:y + h, x:x + w]

                eyes = eye_cascade.detectMultiScale(roi_gray)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

            # write the flipped frame
            out.write(img)
            cv2.imshow('img', img)
            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break
        else:
            break

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def detection_face(url=0, show_image=False, saved_parts=False, saved_complete=False):
    # https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

    cap = cv2.VideoCapture(url)

    ret, img = cap.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    i = 0
    for (top, right, bottom, left) in faces:
        if saved_parts:
            print("face {} located at Top:{}, Left: {}, Bottom: {}, Right: {}.".format(i, top, left, bottom, right))
            face_image = img[right:right + left, top:top + bottom]
            pil_image = Image.fromarray(face_image)
            pil_image.save('out{}_{}'.format(i, url))
            i += 1

        cv2.rectangle(img, (top, right), (top + bottom, right + left), (255,255,255), 2)
        roi_gray = gray[right:right + left, top:top + bottom]
        roi_color = img[right:right + left, top:top + bottom]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 0), 2)

    if saved_complete:
        cv2.imwrite('Out_{}'.format(url), img)

    while show_image:
        cv2.imshow('img', img)
        if cv2.waitKey(1) & 0xff == 27:
            break