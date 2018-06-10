import cv2
import os
import numpy as np
from PIL import Image
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')

current_id = 0
label_ids = {}
y_labels = []
x_train = []

for root, dirs, files in os.walk(image_dir):
	for file in files:
		if file.endswith("png") or file.endswith("jpg"):

			# Get Labels ( folder name )
			path = os.path.join(root, file)
			label = os.path.basename(root).replace(" ", "-").lower()
			if not label in label_ids:
				label_ids[label] = current_id
				current_id += 1
			id_ = label_ids[label]

			# Convert Image to Grayscale and its pixels into numpy array
			pil_image = Image.open(path).convert("L")
			size = (400, 400)
			final_image = pil_image.resize(size, Image.ANTIALIAS)
			image_array = np.array(final_image, "uint8")

			# Find Face from the Image
			faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)
			for (x, y, w, h) in faces:
				roi = image_array[y:y+h, x:x+w]
				x_train.append(roi)
				y_labels.append(id_)

with open("face-labels.pickle", 'wb') as f:
	pickle.dump(label_ids, f)

# Train the model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(x_train, np.array(y_labels))
recognizer.write("trainer.yml")