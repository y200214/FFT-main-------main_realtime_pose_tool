# ファイル名: app/mode_handler/realtime_handler.py

import queue
import multiprocessing
import time
from .mode_handler_base import ModeHandlerBase
from services.capture_service import CaptureService
import dataclasses

class RealtimeHandler(ModeHandlerBase):
    """リアルタイム解析モードのロジックを担当するクラス。"""
    def __init__(self, controller):
        super().__init__(controller)
        self.data_queue = self.controller.app.data_queue
        self.frame_queue = self.controller.app.frame_queue
        self.status_queue = self.controller.status_queue
        self.capture_service = None
        self.analysis_active = multiprocessing.Event()

    def on_mode_selected(self):
        print("INFO: リアルタイムモード選択。プレビューを開始します。")
        self.analysis_active.clear()

        rt_config_obj = self.controller.config_manager.config.realtime_settings
        rt_config_dict = dataclasses.asdict(rt_config_obj)

        self.capture_service = CaptureService(self.data_queue, self.frame_queue, self.status_queue, rt_config_dict, self.analysis_active)
        self.capture_service.start()

        self.app.load_csv_button.config(state="disabled")
        self.app.batch_button.config(state="disabled")
        self.app.start_button.config(state="normal")
        
        self.controller._start_preview_loop()

    def on_mode_deselected(self):
        print("INFO: 他のモードに切り替え。プレビューを停止します。")
        if self.capture_service:
            self.capture_service.stop()
            self.capture_service = None
        
        self.controller._stop_preview_loop()
        self.app.start_button.config(state="disabled")

    def _start_specifics(self):
        print("INFO: 解析開始ボタン押下。解析モードに移行します。")
        self.analysis_active.set()
        self.model.full_history = []
        self.model.active_ids = []

    def _stop_specifics(self):
        print("INFO: 解析停止ボタン押下。プレビューモードに戻ります。")
        self.analysis_active.clear()

    def _toggle_pause_specifics(self):
        if not self.is_running:
            return
        if self.is_paused:
            print("リアルタイム解析を一時停止しました。")
        else:
            print("リアルタイム解析を再開しました。")
            
    def get_next_data_packet(self):
        try:
            packet = self.data_queue.get_nowait()
            new_ids = [k for k in packet.keys() if k.startswith('ID_') and k not in self.model.active_ids]
            if new_ids:
                self.model.active_ids.extend(new_ids)
                self.model.active_ids.sort()
            return packet
        except queue.Empty:
            return None

    def get_latest_frame(self):
        frame = None
        if not self.frame_queue.empty():
            try:
                frame = self.frame_queue.get_nowait()
            except queue.Empty:
                pass
        return frame