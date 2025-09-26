# app/mode_handler/csv_replay_handler.py

import pandas as pd
from .mode_handler_base import ModeHandlerBase
from constants import ALL_VARIABLES

class CsvReplayHandler(ModeHandlerBase):
    """CSV再生モードのロジックを担当するクラス。"""
    def __init__(self, controller):
        super().__init__(controller)
        self.csv_replay_index = 0

    def _before_start(self):
        """開始前のチェック処理"""
        if self.model.csv_replay_data is None:
            self.app.ui_manager.show_info("情報", "再生するCSVファイルが読み込まれていません。")
            return False
        return True

    def _start_specifics(self):
        """CSVモード固有の開始処理"""
        self.model.full_history = []
        self.csv_replay_index = 0
        print(f"CSV再生を開始します。対象ID: {self.model.active_ids}")

    def _stop_specifics(self):
        """CSVモード固有の停止処理"""
        print("CSV再生を停止しました。")

    def _toggle_pause_specifics(self):
        """CSVモード固有の一時停止/再開処理"""
        if self.is_paused:
            print("CSV再生を一時停止しました。")
        else:
            print("CSV再生を再開しました。")
    def get_next_data_packet(self):
        if self.model.csv_replay_data is None or self.csv_replay_index >= len(self.model.csv_replay_data):
            return None

        current_row = self.model.csv_replay_data.iloc[self.csv_replay_index]
        new_data = {'timestamp': current_row.name}
        for id_name in self.model.active_ids:
            id_data = {}
            for var in ALL_VARIABLES:
                col_name = f"{id_name}_{var}"
                if col_name in current_row and not pd.isna(current_row[col_name]):
                    id_data[var] = current_row[col_name]
            if id_data:
                new_data[id_name] = id_data

        self.csv_replay_index += 1
        return new_data

    def on_mode_selected(self):
        self.app.load_csv_button.config(state="normal")
        if self.model.csv_replay_data is not None and not self.model.csv_replay_data.empty:
            self.app.batch_button.config(state="normal")
        else:
            self.app.batch_button.config(state="disabled")

    def on_mode_deselected(self):
        self.app.load_csv_button.config(state="disabled")
        self.app.batch_button.config(state="disabled")


