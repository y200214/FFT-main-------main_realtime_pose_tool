# ファイル名: services/analysis_utils.py

import numpy as np
from constants import ALL_VARIABLES
from mediapipe.python.solutions.pose import PoseLandmark
from math import hypot
import cv2

def calculate_emotion_features(emotions_dict):
    """
    DeepFaceの出力辞書から感情に関する特徴量を正規化して計算する。
    """
    features = {}
    if not emotions_dict:
        return {key: 0.0 for key in ['happy', 'sad', 'surprise', 'neutral', 'angry', 'disgust', 'fear', 'contempt']}

    emotion_map = {
        'happy': 'happy', 'surprise': 'surprise', 'neutral': 'neutral',
        'sad': 'sad', 'disgust': 'disgust', 'fear': 'fear',
        'angry': 'anger', 'contempt': 'contempt'
    }
    for api_name, feature_name in emotion_map.items():
        score = emotions_dict.get(api_name, 0.0) / 100.0 # 100で割って正規化
        features[feature_name] = score
    return features

def calculate_holistic_features(holistic_results, frame_shape):
    """
    MediaPipe Holisticの結果から行動に関する特徴量を計算する。
    """
    features = {key: 0.0 for key in ['lips', 'left_eye', 'right_eye', 'head', 'left_shoulder', 'right_shoulder', 'left_hand', 'right_hand', 'roll', 'pitch', 'yaw']}
    
    if not holistic_results:
        return features
        
    # 顔のランドマークから特徴量計算
    if holistic_results.face_landmarks:
        face_lm = holistic_results.face_landmarks.landmark
        features["lips_value"] = _calculate_lip_opening(face_lm)
        features["left_eye_value"] = _calculate_eye_aspect_ratio(face_lm, is_left=True)
        features["right_eye_value"] = _calculate_eye_aspect_ratio(face_lm, is_left=False)

    # 姿勢のランドマークから特徴量計算
    if holistic_results.pose_landmarks:
        pose_lm = holistic_results.pose_landmarks.landmark
        h, w, _ = frame_shape
        
        features["head_value"] = pose_lm[PoseLandmark.NOSE.value].y
        features["left_shoulder_value"] = pose_lm[PoseLandmark.LEFT_SHOULDER.value].y
        features["right_shoulder_value"] = pose_lm[PoseLandmark.RIGHT_SHOULDER.value].y
        features["left_hand_value"] = pose_lm[PoseLandmark.LEFT_WRIST.value].y
        features["right_hand_value"] = pose_lm[PoseLandmark.RIGHT_WRIST.value].y

        # 頭の向きを計算
        image_points = np.array([
            (pose_lm[PoseLandmark.NOSE.value].x * w, pose_lm[PoseLandmark.NOSE.value].y * h),
            (pose_lm[PoseLandmark.CHIN.value].x * w, pose_lm[PoseLandmark.CHIN.value].y * h),
            (pose_lm[PoseLandmark.LEFT_EYE_INNER.value].x * w, pose_lm[PoseLandmark.LEFT_EYE_INNER.value].y * h),
            (pose_lm[PoseLandmark.RIGHT_EYE_INNER.value].x * w, pose_lm[PoseLandmark.RIGHT_EYE_INNER.value].y * h),
            (pose_lm[PoseLandmark.LEFT_MOUTH.value].x * w, pose_lm[PoseLandmark.LEFT_MOUTH.value].y * h),
            (pose_lm[PoseLandmark.RIGHT_MOUTH.value].x * w, pose_lm[PoseLandmark.RIGHT_MOUTH.value].y * h)
        ], dtype="double")
        
        model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left Mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ])

        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )
        dist_coeffs = np.zeros((4, 1)) # Assuming no lens distortion
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

        if success:
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            sy = np.sqrt(rotation_matrix[0, 0] * rotation_matrix[0, 0] + rotation_matrix[1, 0] * rotation_matrix[1, 0])
            singular = sy < 1e-6
            if not singular:
                x = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
                y = np.arctan2(-rotation_matrix[2, 0], sy)
                z = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
            else:
                x = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
                y = np.arctan2(-rotation_matrix[2, 0], sy)
                z = 0
            
            features["roll"] = np.degrees(x)
            features["pitch"] = np.degrees(y)
            features["yaw"] = np.degrees(z)

    return features


def _calculate_lip_opening(face_landmarks):
    try:
        top_lip = face_landmarks[13]
        bottom_lip = face_landmarks[14]
        return abs(top_lip.y - bottom_lip.y)
    except IndexError:
        return 0.0

def _calculate_eye_aspect_ratio(face_landmarks, is_left):
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