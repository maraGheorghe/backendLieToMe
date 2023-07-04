class Video:
    def __init__(self, name, frames_number, frames_landmarks):
        self.__name = name
        self.__frames_number = frames_number
        self.__frames_landmarks = frames_landmarks

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, new_name):
        self.__name = new_name

    @property
    def frames_number(self):
        return self.__frames_number

    @frames_number.setter
    def frames_number(self, new_frames_number):
        self.__frames_number = new_frames_number

    @property
    def frames_landmarks(self):
        return self.__frames_landmarks

    @frames_landmarks.setter
    def frames_landmarks(self, new_frames_landmarks):
        self.__frames_landmarks = new_frames_landmarks
