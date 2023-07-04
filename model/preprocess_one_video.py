import cv2
import dlib
import numpy as np
import os
from model.paths import UTILS_PATH

class Preprocess:

    def __init__(self):
        haar_classifier_path = os.path.join(UTILS_PATH, 'used_models', 'haarcascade_frontalface_default.xml')
        shape_predictor = os.path.join(UTILS_PATH, 'used_models', 'shape_predictor_68_face_landmarks.dat')
        self.face_cascade = cv2.CascadeClassifier(haar_classifier_path)
        self.predictor = dlib.shape_predictor(shape_predictor)

    def normalize(self, data):
        no_of_frames = np.array(data).shape[0]
        all_landmarks = [landmark for landmarks_list in data for landmark in landmarks_list]
        mean = np.mean(all_landmarks, axis=0)
        all_landmarks = all_landmarks - mean
        std = np.std(all_landmarks, axis=0)
        utils_pth = os.path.join(UTILS_PATH, 'normalization')
        mean = np.load(os.path.join(utils_pth, 'mean.npy'))
        std = np.load(os.path.join(utils_pth, 'std.npy'))
        data -= mean
        data /= std
        data = np.array(data)
        return data.reshape((no_of_frames, 68, 2))

    def detect_faces(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        cropped_faces = []
        for (x, y, w, h) in faces:
            cropped_faces.append(image[y:y + h, x:x + w])
        return faces, cropped_faces

    def extract_facial_landmarks(self, face):
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        landmarks = self.predictor(gray, dlib.rectangle(0, 0, face.shape[0], face.shape[1]))
        landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])
        return landmarks

    def align_faces(self, image, face_locations):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        aligned_faces = []
        facial_landmarks = []
        for (x, y, w, h) in face_locations:
            landmarks = self.predictor(gray, dlib.rectangle(x, y, x + w, y + h))
            aligned_face = dlib.get_face_chip(image, landmarks)
            aligned_faces.append(aligned_face)
            facial_landmarks.append(self.extract_facial_landmarks(aligned_face))
        return facial_landmarks

    def frame_extraction_landmarks(self, video_path):
        x = []
        interval = 10
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            print(f"Error opening video file {video_path}")
            raise Exception("Video isn't save correctly.")
        count = 0
        landmarks_list = []
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            count += 1
            if count % interval == 0:
                normalized_frame = cv2.normalize(frame, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
                face_locations, cropped_faces = self.detect_faces(normalized_frame)
                if len(face_locations) > 1:
                    areas = [w * h for (_, _, w, h) in face_locations]
                    largest_index = np.argmax(areas)
                    face_locations = face_locations[largest_index:largest_index + 1]
                landmarks = self.align_faces(normalized_frame, face_locations)
                if len(landmarks) < 1 or landmarks[0].shape[0] != 68:
                    print(f"Skipping frame {count} in video {video_path}: No faces detected.")
                    continue
                x.append(count)
                landmarks_list.append(landmarks)
        return landmarks_list

