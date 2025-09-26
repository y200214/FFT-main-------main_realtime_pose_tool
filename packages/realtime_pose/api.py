import cv2
from deepface import DeepFace
import mediapipe as mp
import numpy as np
from ultralytics import YOLO
import logging
from collections import OrderedDict
from scipy.spatial import distance as dist

# MediaPipe Holisticモデルをインポート
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
PoseLandmark = mp.solutions.pose.PoseLandmark

class CentroidTracker:
    def __init__(self, maxDisappeared=50):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.rects = OrderedDict()
        self.maxDisappeared = maxDisappeared

    def register(self, centroid, rect):
        self.objects[self.nextObjectID] = centroid
        self.rects[self.nextObjectID] = rect
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]
        del self.rects[objectID]

    def update(self, rects):
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.rects

        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(len(inputCentroids)):
                self.register(inputCentroids[i], rects[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            usedRows = set()
            usedCols = set()
            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.rects[objectID] = rects[col]
                self.disappeared[objectID] = 0
                usedRows.add(row)
                usedCols.add(col)
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)
            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col], rects[col])
        return self.rects


class RealtimePose:
    """
    YOLOv8, MediaPipe, DeepFaceを使用して、特徴量を抽出するクラス。
    """
    def __init__(self, model_path='yolov8n-face.pt'):
        try:
            self.detector_model = YOLO(model_path)
            self.tracker = CentroidTracker(maxDisappeared=80) 
        except Exception as e:
            logging.error(f"モデルの読み込みに失敗しました: {e}")
            raise
            
        self.holistic = mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def track_video(self, video_source):
        """
        映像ソースからフレームを処理し、結果をyieldで返すイテレータ。
        """
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            logging.error(f"映像ソース {video_source} を開けませんでした。")
            return

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 1フレームごとの処理はprocess_frameに任せる
            annotated_frame, keypoints = self.process_frame(frame)
            
            # 処理結果をパイプライン側に返す
            yield annotated_frame, keypoints
        
        cap.release()
        self.close()

    def process_frame(self, frame):
        """
        フレームを処理し、描画済みフレームとキーポイント辞書を返す。
        """
        annotated_frame = frame.copy()
        keypoints_per_person = {}

        face_results = self.detector_model.predict(frame, verbose=False, conf=0.6, iou=0.4)
        rects = [box.xyxy[0].cpu().numpy().astype(int) for r in face_results for box in r.boxes]
        
        tracked_faces = self.tracker.update(rects)

        for (track_id, box) in tracked_faces.items():
            x1, y1, x2, y2 = box
            
            face_roi = frame[y1:y2, x1:x2]
            if face_roi.size == 0:
                continue

            #  感情分析を実行 
            emotions = {}
            try:
                analysis = DeepFace.analyze(
                    img_path=face_roi,
                    actions=['emotion'],
                    enforce_detection=False,
                    silent=True
                )
                if isinstance(analysis, list) and len(analysis) > 0:
                    emotions = analysis[0]['emotion']
            except Exception as e:
                logging.warning(f"DeepFace analysis failed on ROI: {e}")

            padding = 30
            exp_y1 = max(0, y1 - padding)
            exp_y2 = min(frame.shape[0], y2 + padding)
            exp_x1 = max(0, x1 - padding)
            exp_x2 = min(frame.shape[1], x2 + padding)
            expanded_roi = frame[exp_y1:exp_y2, exp_x1:exp_x2]

            holistic_results = None
            if expanded_roi.size > 0:
                roi_rgb = cv2.cvtColor(expanded_roi, cv2.COLOR_BGR2RGB)
                holistic_results = self.holistic.process(roi_rgb)
                
                if holistic_results and holistic_results.pose_landmarks:
                    person_keypoints = []
                    for landmark in holistic_results.pose_landmarks.landmark:
                        person_keypoints.append([landmark.x, landmark.y, landmark.visibility])
                    keypoints_per_person[track_id] = person_keypoints

                custom_drawing_spec = mp_drawing.DrawingSpec(color=(0, 255, 127), thickness=1, circle_radius=1)
                mp_drawing.draw_landmarks(
                    expanded_roi, holistic_results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                    custom_drawing_spec, custom_drawing_spec)
                mp_drawing.draw_landmarks(
                    expanded_roi, holistic_results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                    custom_drawing_spec, custom_drawing_spec)
                annotated_frame[exp_y1:exp_y2, exp_x1:exp_x2] = expanded_roi

            if emotions:
                emotion_styles = {
                    'happy':    {'name': 'Happy',    'color': (34, 139, 34), 'text': (255, 255, 255)},
                    'sad':      {'name': 'Sad',      'color': (139, 0, 0),   'text': (255, 255, 255)},
                    'surprise': {'name': 'Surprise', 'color': (255, 165, 0), 'text': (0, 0, 0)},
                    'neutral':  {'name': 'Neutral',  'color': (128, 128, 128), 'text': (255, 255, 255)},
                    'angry':    {'name': 'Angry',    'color': (255, 0, 0),   'text': (255, 255, 255)},
                    'disgust':  {'name': 'Disgust',  'color': (0, 100, 0),   'text': (255, 255, 255)},
                    'fear':     {'name': 'Fear',     'color': (75, 0, 130),  'text': (255, 255, 255)},
                    'contempt': {'name': 'Contempt', 'color': (210, 105, 30), 'text': (255, 255, 255)},
                }
                dominant_emotion = max(emotions, key=emotions.get)
                style = emotion_styles.get(dominant_emotion, {'name': 'Unknown', 'color': (0,0,0), 'text': (255,255,255)})
                display_text = f"ID:{track_id} {style['name']}"
                bg_color = style['color']
                text_color = style['text']
            else:
                display_text = f"ID:{track_id}"
                bg_color = (128, 128, 128)
                text_color = (255, 255, 255)

            (w, h), _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            bg_start = (x2 - w - 10, y1 - h - 15)
            bg_end = (x2, y1 - 10)
            text_pos = (x2 - w - 5, y1 - 15)
            
            cv2.rectangle(annotated_frame, bg_start, bg_end, bg_color, -1)
            cv2.putText(annotated_frame, display_text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, text_color, 2, cv2.LINE_AA)

        return annotated_frame, keypoints_per_person

    def close(self):
        self.holistic.close()