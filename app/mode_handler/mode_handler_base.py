# app/mode_handler/mode_handler_base.py

from abc import ABC, abstractmethod

class ModeHandlerBase(ABC):
    """
    各解析モードの共通インターフェースを定義する抽象基底クラス。
    """
    def __init__(self, controller):
        self.controller = controller
        self.app = controller.app
        self.model = controller.model
        self.is_running = False
        self.is_paused = False

    def start(self):
        """解析を開始する"""
        if self.is_running:
            return
        if not self._before_start():
            return
        self.is_running = True
        self.is_paused = False
        self._start_specifics()
        self.controller.start_update_loop()

    def stop(self):
        """解析を停止する"""
        if not self.is_running:
            return
        self.is_running = False
        self._stop_specifics()
        self.controller.stop_update_loop()

    def toggle_pause(self):
        """一時停止/再開を切り替える"""
        if not self.is_running:
            return
        self.is_paused = not self.is_paused
        self._toggle_pause_specifics()
        if self.is_paused:
            self.controller.stop_update_loop()
        else:
            self.controller.start_update_loop()

    # --- 子クラスで実装するメソッド群 ---
    @abstractmethod
    def get_next_data_packet(self):
        """次のデータパケットを取得する"""
        pass

    def _before_start(self):
        """startが呼ばれた直後、ループ開始前に行うチェック処理"""
        return True # デフォルトでは常に成功

    @abstractmethod
    def _start_specifics(self):
        """モード固有の開始処理"""
        pass

    @abstractmethod
    def _stop_specifics(self):
        """モード固有の停止処理"""
        pass

    def _toggle_pause_specifics(self):
        """モード固有の一時停止/再開処理"""
        pass # 何も実装しなくても良い場合もある
    
    def on_mode_selected(self):
        """このモードが選択されたときに呼ばれる処理 (UIの有効/無効化など)"""
        pass

    def on_mode_deselected(self):
        """他のモードが選択されたときに呼ばれる処理"""
        pass