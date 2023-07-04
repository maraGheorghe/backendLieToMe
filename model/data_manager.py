import numpy as np
from sklearn.model_selection import train_test_split

from data_loader import DataLoader


class DataManager:

    def __init__(self, data_loader_truth: DataLoader, data_loader_lie: DataLoader):
        self.__data_loader_truth = data_loader_truth
        self.__data_loader_lie = data_loader_lie

    def normalize(self):
        all_landmarks = [x.frames_landmarks for x in self.__data_loader_truth.data + self.__data_loader_lie.data]
        all_landmarks = [landmark for landmarks_list in all_landmarks for landmark in landmarks_list]
        mean = np.mean(all_landmarks, axis=0)
        all_landmarks = all_landmarks - mean
        std = np.std(all_landmarks, axis=0)
        for video in self.__data_loader_truth.data + self.__data_loader_lie.data:
            video.frames_landmarks -= mean
            video.frames_landmarks = video.frames_landmarks / std

    def __find_most_relevant_landmarks_for_a_video(self, video, size):
        landmarks = np.array(video.frames_landmarks)
        # Măsurarea gradului de schimbare a repelor între cadrele consecutive din video.
        intensity_measures = np.linalg.norm(np.diff(landmarks, axis=0), axis=2)
        top_indices = np.argsort(np.max(intensity_measures, axis=1))[-size:]
        return landmarks[top_indices]

    def __find_echidistant_landmarks_foar_a_video(self, video, size):
        landmarks = np.array(video.frames_landmarks)
        indexes = np.linspace(0, len(landmarks) - 1, num=size, dtype=int)
        return landmarks[indexes]

    def get_part_from_video(self, video, label, size=0.5):
        size = int(len(video.frames_landmarks) * size)
        if label == 1:
            return self.__find_echidistant_landmarks_foar_a_video(video, size)
        return self.__find_most_relevant_landmarks_for_a_video(video, size)

    def return_data_unsplit(self):
        X = self.__data_loader_truth.data + self.__data_loader_lie.data
        y = np.concatenate([np.ones(len(self.__data_loader_truth.data)), np.zeros(len(self.__data_loader_lie.data))])
        X_processed = np.empty((0, 68, 2))
        y_processed = np.empty((0,))
        for video, label in zip(X, y):
            X_processed = np.concatenate((X_processed, video.frames_landmarks))
            y_processed = np.concatenate((y_processed, np.full(len(video.frames_landmarks), label)))
        shuffle_idx = np.random.permutation(X_processed.shape[0])
        X_processed = X_processed[shuffle_idx]
        y_processed = y_processed[shuffle_idx]
        return X_processed, y_processed

        # X = self.__data_loader_truth.data + self.__data_loader_lie.data
        # y = np.concatenate(
        #     [np.ones(len(self.__data_loader_truth.data)), np.zeros(len(self.__data_loader_lie.data))])
        # shuffle_idx = np.random.permutation(len(X))
        # X = [X[i] for i in shuffle_idx]
        # y = [y[i] for i in shuffle_idx]
        # videos_per_fold = len(X) // 10
        # current_fold = 0
        # videos_in_current_fold = 0
        # folds = np.empty((0,))
        # X_processed = np.empty((0, 68, 2))
        # y_processed = np.empty((0,))
        # for video, label in zip(X, y):
        #     X_processed = np.concatenate((X_processed, video.frames_landmarks))
        #     folds = np.concatenate((folds, np.full(len(video.frames_landmarks), current_fold)))
        #     y_processed = np.concatenate((y_processed, np.full(len(video.frames_landmarks), label)))
        #     videos_in_current_fold += 1
        #     if videos_in_current_fold == videos_per_fold:
        #         current_fold += 1
        #         videos_in_current_fold = 0
        # return X_processed, y_processed, folds

    def split_data_smart(self):
        X = self.__data_loader_truth.data + self.__data_loader_lie.data
        y = np.concatenate([np.ones(len(self.__data_loader_truth.data)), np.zeros(len(self.__data_loader_lie.data))])
        # X_train_val, X_test, y_train_val, y_test = \
        #     train_test_split(X, y, test_size=0.05, random_state=42, shuffle=True)
        # X_train_videos, X_val_videos, y_train_videos, y_val_videos = \
        #     train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42, shuffle=True)
        X_train = np.empty((0, 68, 2))
        y_train = np.empty((0,))
        for video, label in zip(X, y):
            landmarks = self.get_part_from_video(video, label)
            X_train = np.concatenate((X_train, landmarks))
            y_train = np.concatenate((y_train, np.full(len(landmarks), label)))
        # X_val = np.empty((0, 68, 2))
        # y_val = np.empty((0,))
        # for video, label in zip(X_val_videos, y_val_videos):
        #     landmarks = self.get_part_from_video(video, label)
        #     X_val = np.concatenate((X_val, landmarks))
        #     y_val = np.concatenate((y_val, np.full(len(landmarks), label)))
        X_train = np.array(X_train).reshape((len(X_train), -1, 2))
        # X_val = np.array(X_val).reshape((len(X_val), -1, 2))
        return X_train, np.array(y_train)\
            # , X_val, np.array(y_val), X_test, y_test
