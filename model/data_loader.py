import os

import numpy as np
import pandas as pd

from video import Video


class DataLoader:

    def __init__(self, csv_dir):
        self.__csv_dir = csv_dir
        self.__data = []

    @property
    def data(self):
        return self.__data

    def load_data(self):
        self.__data = []
        for filename in os.listdir(self.__csv_dir):
            if not filename.endswith('.csv'):
                continue
            video_name = ''
            frames_number = []
            frames_landmarks = []
            csv = pd.read_csv(os.path.join(self.__csv_dir, filename), header=None)
            for row in np.array(csv):
                video_name = row[0]
                frames_number.append(row[1])
                landmarks = row[2].strip().split('],')
                landmarks = [np.array(x.strip(' [] ').strip().split(',')).astype('float64') for x in landmarks]
                frames_landmarks.append(landmarks)
            self.__data.append(Video(video_name, frames_number, frames_landmarks))
