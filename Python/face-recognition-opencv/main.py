# Implementation of tutorial by Justin from CodingEnterpreneurs
# https://www.youtube.com/watch?v=PmZ29Vta7Vc
# https://www.codingforentrepreneurs.com/

import numpy as np
import cv2
import pickle

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

# Import Labels
labels = {}
with open("face-labels.pickle", "rb") as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

# Image capturing device
cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.5, minNeighbors = 1)

    for (x, y, w, h) in faces:
        print(x, y, w, h)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Make Prediction
        id_, conf = recognizer.predict(roi_gray)
        if conf >= 50 and conf <= 100:
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_] + " " + str(round(conf, 2)) + "%"
            color = (255, 255, 255)
            stroke = 2
            cv2.putText(frame, name, (x, y), font, 1, color, stroke) #, cv2.LINE_AA)

        # Display Rectangular Box
        width = x + w
        height = y + h
        color = (255, 0, 0)
        stroke = 2
        cv2.rectangle(frame, (x, y), (width, height), color, stroke)

    # Display the resulting framec
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()