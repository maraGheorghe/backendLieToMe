import cv2


class AnnotateVideo:
    def __init__(self):
        pass

    def annotate_frames(self, video_path, predictions):
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        frame_number = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // 10
        annotated_frames = []
        chunk_size = 1
        i = 0
        last_prediction, last_color = None, None
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if (frame_count - 1) % 10 != 0 or len(predictions) == i:
                annotated_frame = cv2.putText(
                    img=frame,
                    text=last_prediction,
                    org=(20, 50),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=2,
                    color=last_color,
                    thickness=5
                )
                annotated_frames.append(annotated_frame)
                continue
            frame_annotation = "truthful"  # Valoarea implicită este "truthful" pentru frame-uri izolate
            color = (0, 255, 0)
            if predictions[i] == 0:
                start_index = max(0, i - chunk_size // 2)
                end_index = min(frame_number, i + chunk_size // 2 + 1)
                window_predictions = predictions[start_index:end_index]

                if len(window_predictions) == chunk_size and all(
                        prediction == 0 for prediction in window_predictions):
                    frame_annotation = "deceptive"
                    color = (0, 0, 255)
            annotated_frame = cv2.putText(
                img=frame,
                text=frame_annotation,
                org=(20, 50),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=2,
                color=color,
                thickness=5
            )
            annotated_frames.append(annotated_frame)
            last_prediction = frame_annotation
            last_color = color
            i += 1
        cap.release()
        return annotated_frames


