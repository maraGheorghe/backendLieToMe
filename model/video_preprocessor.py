import threading

import cv2
import dlib
import numpy as np
import os

from model.paths import UTILS_PATH


class VideoProcessor(threading.Thread):
    def __init__(self, name_of_thread, video_path, output_list,
                 start_frame, end_frame, interval):
        threading.Thread.__init__(self)
        self.__name = int(name_of_thread)
        self.video_path = video_path
        self.output_list = output_list
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.interval = interval
        haar_classifier_path = os.path.join(UTILS_PATH, 'used_models', 'haarcascade_frontalface_default.xml')
        shape_predictor = os.path.join(UTILS_PATH, 'used_models', 'shape_predictor_68_face_landmarks.dat')
        self.face_cascade = cv2.CascadeClassifier(haar_classifier_path)
        self.predictor = dlib.shape_predictor(shape_predictor)

    def detect_face(self, gray):
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return max(faces, key=lambda rect: rect[2] * rect[3]) if faces != () else []

    def extract_facial_landmarks(self, face):
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        landmarks = self.predictor(gray, dlib.rectangle(0, 0, face.shape[0], face.shape[1]))
        landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])
        return landmarks

    def align_faces(self, image, greyscale_image, face_locations):
        (x, y, w, h) = face_locations
        landmarks = self.predictor(greyscale_image, dlib.rectangle(x, y, x + w, y + h))
        aligned_face = dlib.get_face_chip(image, landmarks)
        return self.extract_facial_landmarks(aligned_face)

    def normalize(self, data):
        all_landmarks = [landmark for landmarks_list in data for landmark in landmarks_list]
        mean = np.mean(all_landmarks, axis=0)
        all_landmarks = all_landmarks - mean
        std = np.std(all_landmarks, axis=0)
        data -= mean
        data /= std
        data = np.array(data)
        return data.reshape((max(data.shape), 68, 2))

    def run(self):
        x = []
        landmarks_list = []
        video = cv2.VideoCapture(self.video_path)
        video.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
        count = self.start_frame

        while count <= self.end_frame:
            success, frame = video.read()
            if not success:
                break

            count += 1
            if count % self.interval == 0:
                normalized_frame = cv2.normalize(frame, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
                grey_scale_image = cv2.cvtColor(normalized_frame, cv2.COLOR_BGR2GRAY)
                face_location = self.detect_face(grey_scale_image)
                if face_location == []:
                    print(f"Skipping {count}.")
                    continue
                landmarks = self.align_faces(normalized_frame, grey_scale_image, face_location)
                if len(landmarks) < 1 or landmarks.shape[0] != 68:
                    print(f"Skipping thread {self.__name} frame {count} in video {self.video_path}: No faces detected.")
                else:
                    x.append(count)
                    landmarks_list.append(landmarks)

        self.output_list[self.__name] = landmarks_list

        video.release()
