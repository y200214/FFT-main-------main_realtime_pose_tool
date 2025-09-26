# ファイル名: services/feature_extractor.py

import cv2
import mediapipe as mp
import logging
import numpy as np
import time
from deepface import DeepFace
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import holistic as mp_holistic

from .analysis_utils import calculate_emotion_features, calculate_holistic_features
from .person_tracker import PersonTracker 

logger = logging.getLogger(__name__)

class FeatureExtractor:
    """
    YOLOv8とMediaPipe Holisticを使用して、特徴量を抽出するクラス。
    """
    def __init__(self, config):
        self.config = config
        self.tracker = PersonTracker(
            model_path=self.config['yolo_model_path'],
            device=self.config['device']
        )
        self.holistic = mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        logger.info("FeatureExtractorの初期化が完了しました。")

    def process_frame(self, frame):
        """
        1フレームを処理し、描画済みフレームと特徴量リストを返す。
        """
        annotated_frame = frame.copy()
        
        # 1. 人物追跡
        tracked_persons, annotated_frame = self.tracker.track(annotated_frame)
        all_features = []

        if not tracked_persons:
            return annotated_frame, all_features
            
        # 2. 各人物の特徴量抽出
        for person in tracked_persons:
            person_id = person['id']
            box = person['box']
            x1, y1, x2, y2 = box
            
            padding = 30
            exp_y1 = max(0, y1 - padding)
            exp_y2 = min(frame.shape[0], y2 + padding)
            exp_x1 = max(0, x1 - padding)
            exp_x2 = min(frame.shape[1], x2 + padding)
            expanded_roi = frame[exp_y1:exp_y2, exp_x1:exp_x2]
            
            if expanded_roi.size == 0:
                continue

            # 感情分析
            emotions = {}
            try:
                face_roi = frame[y1:y2, x1:x2]
                if face_roi.size > 0:
                    analysis = DeepFace.analyze(
                        img_path=face_roi,
                        actions=['emotion'],
                        enforce_detection=False,
                        silent=True
                    )
                    if isinstance(analysis, list) and len(analysis) > 0:
                        emotions = analysis[0]['emotion']
            except Exception as e:
                pass

            # 全身の骨格・顔メッシュ解析
            roi_rgb = cv2.cvtColor(expanded_roi, cv2.COLOR_BGR2RGB)
            holistic_results = self.holistic.process(roi_rgb)

            # ランドマークを描画
            if holistic_results:
                mp_drawing.draw_landmarks(
                    expanded_roi, holistic_results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                    mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                    mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                )
                mp_drawing.draw_landmarks(
                    expanded_roi, holistic_results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                )
                annotated_frame[exp_y1:exp_y2, exp_x1:exp_x2] = expanded_roi

            # 特徴量計算
            emotion_features = calculate_emotion_features(emotions)
            holistic_features = calculate_holistic_features(holistic_results, expanded_roi.shape)
            
            all_person_features = {**emotion_features, **holistic_features}

            all_features.append({
                'id': person_id,
                'features': all_person_features
            })

        return annotated_frame, all_features
    
    def close(self):
        self.holistic.close()