# ファイル名: utils/camera_utils.py

import cv2
import logging
from pygrabber.dshow_graph import FilterGraph
import threading
import time

logger = logging.getLogger(__name__)

def get_available_cameras():
    """
    システムで利用可能なカメラデバイスのインデックスと名前を辞書で返す。
    """
    try:
        devices = FilterGraph().get_input_devices()
        available_cameras = {}
        for device_index, device_name in enumerate(devices):
            available_cameras[device_index] = device_name
        
        logger.info(f"利用可能なカメラ: {available_cameras}")
        return available_cameras
    except Exception as e:
        logger.error(f"カメラの検出に失敗しました (pygrabber): {e}")
        return {} # エラー時は空の辞書を返す

class CameraMonitor(threading.Thread):
    """カメラの接続状況をバックグラウンドで監視するスレッド"""
    def __init__(self, callback, check_interval=2):
        super().__init__(daemon=True)
        self.callback = callback
        self.check_interval = check_interval
        self._stop_event = threading.Event()
        self.last_cameras = {}

    def run(self):
        """スレッドのメイン処理"""
        while not self._stop_event.is_set():
            try:
                current_cameras = get_available_cameras()
                if current_cameras != self.last_cameras:
                    logger.info(f"カメラの変更を検出: {current_cameras}")
                    self.last_cameras = current_cameras
                    self.callback(self.last_cameras)
            except Exception as e:
                logger.error(f"カメラ監視スレッドでエラーが発生しました: {e}")
                # エラーが発生しても監視を続ける
                self.last_cameras = {} # リセット
            
            # stop_event.waitを使うことで、stop()が呼ばれたらすぐにループを抜けられる
            self._stop_event.wait(self.check_interval)

    def stop(self):
        """スレッドを停止する"""
        self._stop_event.set()