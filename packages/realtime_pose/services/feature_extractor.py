import cv2
from deepface import DeepFace
import mediapipe as mp
import numpy as np
from ultralytics import YOLO
import logging
from math import hypot
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

class FeatureExtractor:
    """
    YOLOv8とMediaPipe Holisticを使用して、特徴量を抽出するクラス。
    """
    def __init__(self, model_path='yolov8n.pt'):
        try:
            # メインの検出器を顔モデル一本に絞ります
            self.detector_model = YOLO('yolov8n-face.pt')
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

    def process_frame(self, frame):
        annotated_frame = frame.copy()
        all_features = []
        
        # 1. YOLOv8で「顔」を直接検出
        face_results = self.detector_model.predict(
            frame, 
            verbose=False,
            conf=0.6,
            iou=0.4
        )
        rects = [box.xyxy[0].cpu().numpy().astype(int) for r in face_results for box in r.boxes]
        
        # 2. 検出された「顔」を追跡
        tracked_faces = self.tracker.update(rects)

        # 3. 追跡中の各「顔」に対してループ処理
        for (track_id, box) in tracked_faces.items():
            x1, y1, x2, y2 = box
            
            face_roi = frame[y1:y2, x1:x2]
            if face_roi.size == 0:
                continue

            # 感情分析の実行
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

            # ステップ1：先に骨格やメッシュを描画
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
                
                custom_drawing_spec = mp_drawing.DrawingSpec(color=(0, 255, 127), thickness=1, circle_radius=1)
                mp_drawing.draw_landmarks(
                    expanded_roi, holistic_results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                    custom_drawing_spec, custom_drawing_spec)
                mp_drawing.draw_landmarks(
                    expanded_roi, holistic_results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                    custom_drawing_spec, custom_drawing_spec)
                annotated_frame[exp_y1:exp_y2, exp_x1:exp_x2] = expanded_roi

            # ステップ2：一番上にテキストを描画
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
                display_text = f"ID:{track_id} (No Face)"
                bg_color = (128, 128, 128)
                text_color = (255, 255, 255)

            (w, h), _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            bg_start = (x2 - w - 10, y1 - h - 15)
            bg_end = (x2, y1 - 10)
            text_pos = (x2 - w - 5, y1 - 15)
            
            cv2.rectangle(annotated_frame, bg_start, bg_end, bg_color, -1)
            cv2.putText(annotated_frame, display_text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, text_color, 2, cv2.LINE_AA)

            # 7. 特徴量を抽出
            if holistic_results:
                features = self._extract_features(holistic_results, emotions)
                features['person_id'] = track_id
                all_features.append(features)

        return annotated_frame, all_features
    
    def _extract_features(self, holistic_results, emotions):
        features = {}

        emotion_map = {
            'happy': 'happy_value', 'surprise': 'surprise_value', 'neutral': 'neutral_value',
            'sad': 'sad_value', 'disgust': 'disgust_value', 'fear': 'fear_value',
            'angry': 'anger_value', 'contempt': 'contempt_value'
        }
        for api_name, feature_name in emotion_map.items():
            score = emotions.get(api_name, 0.0) / 100.0
            features[feature_name] = score

        if not holistic_results or not holistic_results.pose_world_landmarks or not holistic_results.face_landmarks:
            default_keys = [
                "lips_value", "left_eye_value", "right_eye_value", "head_value",
                "left_shoulder_value", "right_shoulder_value", "left_hand_value",
                "right_hand_value", "roll_value", "pitch_value", "yaw_value"
            ]
            for key in default_keys:
                features[key] = 0.0
            return features

        pose_landmarks = holistic_results.pose_world_landmarks.landmark
        face_landmarks = holistic_results.face_landmarks.landmark

        features["lips_value"] = self._calculate_lip_opening(face_landmarks)
        features["left_eye_value"] = self._calculate_eye_aspect_ratio(face_landmarks, is_left=True)
        features["right_eye_value"] = self._calculate_eye_aspect_ratio(face_landmarks, is_left=False)

        features["head_value"] = pose_landmarks[PoseLandmark.NOSE.value].y
        features["left_shoulder_value"] = pose_landmarks[PoseLandmark.LEFT_SHOULDER.value].y
        features["right_shoulder_value"] = pose_landmarks[PoseLandmark.RIGHT_SHOULDER.value].y
        features["left_hand_value"] = pose_landmarks[PoseLandmark.LEFT_WRIST.value].y
        features["right_hand_value"] = pose_landmarks[PoseLandmark.RIGHT_WRIST.value].y

        roll, pitch, yaw = self._calculate_head_pose(pose_landmarks)
        features["roll_value"] = roll
        features["pitch_value"] = pitch
        features["yaw_value"] = yaw
        
        return features
    
    def _calculate_lip_opening(self, face_landmarks):
        try:
            top_lip = face_landmarks[13]
            bottom_lip = face_landmarks[14]
            return abs(top_lip.y - bottom_lip.y)
        except IndexError:
            return 0.0

    def _calculate_eye_aspect_ratio(self, face_landmarks, is_left):
        try:
            if is_left:
                p = [face_landmarks[p] for p in [362, 385, 387, 263, 373, 380]]
            else:
                p = [face_landmarks[p] for p in [33, 160, 158, 133, 153, 144]]
            
            v1 = hypot(p[1].x - p[5].x, p[1].y - p[5].y)
            v2 = hypot(p[2].x - p[4].x, p[2].y - p[4].y)
            h = hypot(p[0].x - p[3].x, p[0].y - p[3].y)
            return (v1 + v2) / (2.0 * h) if h > 0 else 0.0
        except IndexError:
            return 0.0

    def _calculate_head_pose(self, pose_landmarks):
        try:
            nose = np.array([pose_landmarks[PoseLandmark.NOSE].x, pose_landmarks[PoseLandmark.NOSE].y, pose_landmarks[PoseLandmark.NOSE].z])
            left_eye = np.array([pose_landmarks[PoseLandmark.LEFT_EYE_INNER].x, pose_landmarks[PoseLandmark.LEFT_EYE_INNER].y, pose_landmarks[PoseLandmark.LEFT_EYE_INNER].z])
            right_eye = np.array([pose_landmarks[PoseLandmark.RIGHT_EYE_INNER].x, pose_landmarks[PoseLandmark.RIGHT_EYE_INNER].y, pose_landmarks[PoseLandmark.RIGHT_EYE_INNER].z])

            eye_center = (left_eye + right_eye) / 2.0
            yaw_vec = nose - eye_center
            yaw = np.degrees(np.arctan2(yaw_vec[0], -yaw_vec[2]))
            pitch = np.degrees(np.arctan2(-yaw_vec[1], np.sqrt(yaw_vec[0]**2 + yaw_vec[2]**2)))
            roll_vec = right_eye - left_eye
            roll = np.degrees(np.arctan2(roll_vec[1], roll_vec[0]))
            return roll, pitch, yaw
        except Exception:
            return 0.0, 0.0, 0.0

    def close(self):
        self.holistic.close()