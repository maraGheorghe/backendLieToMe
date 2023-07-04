import tensorflow as tf
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import os
import numpy as np
import subprocess
from moviepy.editor import VideoFileClip
import cv2
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

from model.preprocess_one_video import Preprocess
from model.video_preprocessor import VideoProcessor
from model.adnotate_video import AnnotateVideo
from model.paths import OUTPUT_PATH, INPUT_PATH, UTILS_PATH

MODEL_PATH = os.path.join(UTILS_PATH, 'used_models', 'model_final_v2.h5')

app = Flask(__name__)
CORS(app)
model = tf.keras.models.load_model(MODEL_PATH)
videos_clients = {}
preprocess = Preprocess()

current_dir = os.path.dirname(os.path.abspath(__file__))
print(f"The current directory is: {current_dir}")

def preprocess_video(filepath):
    interval = 5
    video = cv2.VideoCapture(filepath)
    if not video.isOpened():
        print(f"Error opening video file {filepath}")
        raise Exception("Video isn't save correctly.")
    num_threads = 2
    threads = []
    final_list = [None] * num_threads
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video.release()
    frames_per_thread = total_frames // num_threads
    for i in range(num_threads):
        start_frame = i * frames_per_thread
        end_frame = start_frame + frames_per_thread - 1
        if i == num_threads - 1:
            end_frame = total_frames - 1
        thread = VideoProcessor(i, filepath, final_list, start_frame, end_frame, interval)
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()
    landmark_list = []
    for landmark_from_thread in final_list:
        landmark_list += landmark_from_thread
    return landmark_list

@app.route('/videos', methods=['POST'])
def upload():
    video_file = request.files['file']
    file_path = os.path.join(INPUT_PATH, video_file.filename)
    video_file.save(file_path)

    data = preprocess_video(file_path)
    data = np.expand_dims(np.array(data, dtype=np.float64), axis=1)
    data = preprocess.normalize(data)
    data_path = f"{os.path.splitext(file_path)[0]}.npy"
    np.save(data_path, data)
    return data_path

def get_data_and_video_path(filename):
    video_path = os.path.join(INPUT_PATH, filename)
    data_path = f"{os.path.splitext(video_path)[0]}.npy"
    data = np.load(data_path)
    return data, video_path

@app.route('/videos/detect/<filename>', methods=['GET'])
def detect(filename):
    data, video_path = get_data_and_video_path(filename)
    predictions = model.predict(data)
    threshold = 0.5
    annotated_frames = AnnotateVideo().annotate_frames(video_path, np.where(predictions > threshold, 1, 0).flatten())
    output_path = os.path.join(OUTPUT_PATH, filename.strip().split('.')[0] + '_annotated.mp4')
    final_frames = []
    for frame in annotated_frames:
        final_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    original_video = VideoFileClip(video_path)
    audio = original_video.audio
    final_clip = ImageSequenceClip(final_frames, fps=original_video.fps)
    final_clip = final_clip.set_audio(audio.subclip(0, final_clip.duration))
    output_path = os.path.join(OUTPUT_PATH, filename.strip().split('.')[0] + '_annotated_with_sound.mp4')
    final_clip.write_videofile(output_path)
    output_path = os.path.abspath(output_path)
    return send_file(output_path, mimetype='video/mp4', as_attachment=True)



