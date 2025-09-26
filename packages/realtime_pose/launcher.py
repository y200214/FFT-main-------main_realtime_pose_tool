# launcher.py

import sys
import logging
import cv2
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,QHBoxLayout, QPushButton, QLabel, QComboBox, QMessageBox,QFileDialog)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, pyqtSlot
from utils.camera_utils import get_available_cameras, get_camera_max_resolution
from app.video_processing import VideoProcessorThread
# 他のモジュールをインポート
from utils.camera_utils import get_available_cameras
from app.video_processing import VideoProcessorThread

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MainWindow(QMainWindow):
    """
    メインウィンドウクラス。
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("リアルタイム骨格点列解析ツール")
        self.setGeometry(100, 100, 1280, 720) # ウィンドウサイズを少し大きめに

        # --- UI要素の初期化 ---
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # 映像表示ラベル
        self.video_label = QLabel("カメラを選択して「開始」ボタンを押してください。")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; color: white;")
        self.layout.addWidget(self.video_label, 1) # サイズ変更時に伸長するように

        # 下部のコントロールパネル
        control_layout = QHBoxLayout()
        
        self.camera_combo = QComboBox()
        self.populate_camera_list()
        control_layout.addWidget(QLabel("カメラ:"))
        control_layout.addWidget(self.camera_combo)

        self.start_button = QPushButton("開始")
        self.start_button.clicked.connect(self.start_video_processing)
        control_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("停止")
        self.stop_button.clicked.connect(self.stop_video_processing)
        self.stop_button.setEnabled(False)
        control_layout.addWidget(self.stop_button)

        self.save_button = QPushButton("保存")
        self.save_button.clicked.connect(self.trigger_save_csv)
        self.save_button.setEnabled(False) # 初期状態では無効
        control_layout.addWidget(self.save_button)

        self.fps_label = QLabel("FPS: -")
        control_layout.addStretch()
        control_layout.addWidget(self.fps_label)

        self.layout.addLayout(control_layout)

        # --- メンバ変数の初期化 ---
        self.video_thread = None
        self.available_cameras = []
        self.recorded_data_buffer = []        

    def populate_camera_list(self):
        """
        利用可能なカメラを検出し、コンボボックスに追加します。
        """
        self.camera_combo.clear()
        try:
            self.available_cameras = get_available_cameras()
            if self.available_cameras:
                for index, name in self.available_cameras:
                    self.camera_combo.addItem(f"{name} (Index: {index})", index)
            else:
                self.camera_combo.addItem("カメラが見つかりません")
                self.camera_combo.setEnabled(False)
                self.start_button.setEnabled(False)
        except Exception as e:
            logging.error(f"カメラの検出中にエラーが発生しました: {e}")
            self.show_error_message(f"カメラの検出に失敗しました:\n{e}")

    def start_video_processing(self):
        """
        「開始」ボタンが押されたときの処理。
        """
        selected_index = self.camera_combo.currentData()
        if selected_index is None:
            self.show_error_message("有効なカメラが選択されていません。")
            return
            
        logging.info(f"カメラ {selected_index} で処理を開始します。")

        # 映像処理スレッドを作成して開始
        self.video_thread = VideoProcessorThread(camera_index=selected_index)
        self.video_thread.change_pixmap_signal.connect(self.update_image)
        self.video_thread.processing_fps_signal.connect(self.update_fps)
        self.video_thread.start()

        # UIの状態を更新
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.camera_combo.setEnabled(False)
        self.save_button.setEnabled(False) 

    def stop_video_processing(self):
        """
        「停止」ボタンが押されたときの処理。
        """
        if self.video_thread:
            # スレッドからデータを取得し、クラスのバッファに直接保存
            self.recorded_data_buffer = self.video_thread.get_recorded_data()
            self.video_thread.stop()
            self.video_thread = None
        
        # UIの状態を更新
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.camera_combo.setEnabled(True)
        self.video_label.setText("処理が停止しました。")
        self.fps_label.setText("FPS: -")

        # バッファにデータがあれば保存ボタンを有効化
        if self.recorded_data_buffer:
            self.save_button.setEnabled(True)

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """
        映像処理スレッドからフレームを受け取り、UIを更新します。
        """
        qt_img = self.convert_cv_qt(cv_img)
        self.video_label.setPixmap(qt_img)

    @pyqtSlot(float)
    def update_fps(self, fps):
        """
        FPS表示を更新します。
        """
        self.fps_label.setText(f"FPS: {fps:.2f}")

    def convert_cv_qt(self, cv_img):
            """
            OpenCVの画像(np.ndarray)をPyQtのQPixmapに変換します。
            """
            rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            # --- この行を変更 ---
            # ウィンドウの幅を基準いっぱいに広がるようにスケーリング設定を変更
            scaled_img = convert_to_Qt_format.scaledToWidth(self.video_label.width(), Qt.SmoothTransformation)
            return QPixmap.fromImage(scaled_img)

    def closeEvent(self, event):
        """
        ウィンドウが閉じられるときの処理。
        """
        if self.video_thread and self.video_thread.isRunning():
            self.stop_video_processing()
        
        # 記録されたデータをCSVに保存するか確認
        if self.video_thread and self.video_thread.get_recorded_data():
            reply = QMessageBox.question(self, '確認', 
                "記録されたデータをCSVファイルに保存しますか？", 
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

            if reply == QMessageBox.Yes:
                self.save_data_to_csv(self.video_thread.get_recorded_data())

        event.accept()


    def save_data_to_csv(self, data):
        """
        記録されたデータをCSVファイルに保存します。
        ID別にファイルを分割して出力します。
        """
        if not data:
            logging.warning("保存するデータがありません。")
            return
            
        options = QFileDialog.Options()
        # ユーザーに「ベースとなるファイル名」を選んでもらう
        file_path, _ = QFileDialog.getSaveFileName(self, "CSVファイルを保存 (ベース名)", "","CSV Files (*.csv);;All Files (*)", options=options)
        
        if file_path:
            try:
                df = pd.DataFrame(data)
                
                # --- ここからが改良部分 ---

                # 記録された全IDを重複なく取得
                unique_ids = df['person_id'].unique()
                
                # ベースのファイルパスから名前と拡張子を分離
                import os
                base_name, extension = os.path.splitext(file_path)

                # IDごとにループしてファイルを保存
                for person_id in unique_ids:
                    # 特定のIDのデータだけを抽出
                    person_df = df[df['person_id'] == person_id]
                    
                    # ID別のファイルパスを作成 (例: analysis_id_1.csv)
                    output_path = f"{base_name}_id_{person_id}{extension}"
                    
                    # CSVの列の順番を指定
                    column_order = [
                        'timestamp', 'person_id', 'happy_value', 'surprise_value',
                        'neutral_value', 'sad_value', 'contempt_value', 'disgust_value', 
                        'fear_value', 'anger_value', 'lips_value', 'left_eye_value', 
                        'right_eye_value', 'head_value', 'left_shoulder_value', 
                        'right_shoulder_value', 'left_hand_value', 'right_hand_value', 
                        'roll_value', 'pitch_value', 'yaw_value'
                    ]
                    
                    # DataFrameの列を並べ替え
                    person_df_ordered = person_df.reindex(columns=column_order)
                    
                    # ID別のCSVファイルを保存
                    person_df_ordered.to_csv(output_path, index=False, header=True)
                    logging.info(f"ID {person_id} のデータを {output_path} に保存しました。")

                # --- 改良ここまで ---

                QMessageBox.information(self, "成功", f"{len(unique_ids)}人分のデータを正常に保存しました。")
            except Exception as e:
                logging.error(f"CSVファイルの保存中にエラーが発生しました: {e}")
                self.show_error_message(f"ファイルの保存に失敗しました:\n{e}")

    def show_error_message(self, message):
        """
        エラーメッセージダイアログを表示します。
        """
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setText("エラー")
        msg.setInformativeText(message)
        msg.setWindowTitle("エラー")
        msg.exec_()

    def trigger_save_csv(self):
        """
        「保存」ボタンのクリックイベント。バッファからデータを保存する。
        """
        if hasattr(self, 'recorded_data_buffer') and self.recorded_data_buffer:
            self.save_data_to_csv(self.recorded_data_buffer)
        else:
            QMessageBox.information(self, '情報', 
                '保存するデータがありません。\n解析を開始し、「停止」ボタンを押した後に保存してください。')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())