# app/video_processing.py

import cv2
import time
import logging
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage
import numpy as np
# 1つ上の階層にあるservicesを参照するため
import sys
sys.path.append('..')
from services.feature_extractor import FeatureExtractor
from utils.camera_utils import create_fisheye_to_equirectangular_map, create_ptz_maps 

class VideoProcessorThread(QThread):
    """
    カメラ映像の処理をバックグラウンドで行うスレッド。
    """
    # GUIに渡すためのシグナルを定義
    change_pixmap_signal = pyqtSignal(np.ndarray)
    processing_fps_signal = pyqtSignal(float)
    
    def __init__(self, camera_index, parent=None):
        """
        コンストラクタ。

        Args:
            camera_index (int): 使用するカメラのインデックス。
        """
        super().__init__(parent)
        self.camera_index = camera_index
        self._is_running = True
        self.feature_extractor = None
        self.recorded_data = []

    def run(self):
            """
            スレッドのメイン処理。
            """
            logging.info(f"ビデオ処理スレッドを開始します (カメラインデックス: {self.camera_index})")
            
            try:
                self.feature_extractor = FeatureExtractor()
            except Exception as e:
                logging.error(f"FeatureExtractorの初期化に失敗しました: {e}")
                return

            cap = cv2.VideoCapture(self.camera_index)
            if not cap.isOpened():
                logging.error(f"カメラ {self.camera_index} を開けませんでした。")
                return
                
            frame_count = 0
            start_time = time.time()

            while self._is_running:
                ret, frame = cap.read()
                if not ret:
                    logging.warning("カメラからフレームを取得できませんでした。")
                    break

                # 特徴量抽出と描画
                annotated_frame, features = self.feature_extractor.process_frame(frame)
                
                if features:
                    # タイムスタンプを追加
                    current_time = time.time()
                    for person_features in features:
                        person_features['timestamp'] = current_time
                        self.recorded_data.append(person_features)
                
                # FPS計算
                frame_count += 1
                elapsed_time = time.time() - start_time
                if elapsed_time > 1.0:
                    fps = frame_count / elapsed_time
                    self.processing_fps_signal.emit(fps)
                    # リセット
                    frame_count = 0
                    start_time = time.time()

                # 描画されたフレームをGUIに送信
                self.change_pixmap_signal.emit(annotated_frame)
            
            # クリーンアップ
            cap.release()
            if self.feature_extractor:
                self.feature_extractor.close()
            logging.info("ビデオ処理スレッドを終了します。")



    def stop(self):
        """
        スレッドを安全に停止します。
        """
        self._is_running = False
        self.wait() # スレッドが完全に終了するまで待つ

    def get_recorded_data(self):
        """
        記録された特徴量データを取得します。
        
        Returns:
            list of dict: 記録されたデータのリスト。
        """
        return self.recorded_data