# ファイル名: model.py (修正後)

from . import data_loader

class AnalysisModel:
    def __init__(self):
        """
        アプリケーション全体で共有するデータを保持するクラス。
        計算ロジックは持たない。
        """
        # --- データ管理 ---
        self.full_history = []
        self.active_ids = []
        self.time_series_df = None
        self.csv_replay_data = None
        
        # 計算結果を保持するプロパティ
        self.last_power_spectrums = {}
        self.last_slope_dfs = {}

    def load_csv_data(self, filepaths):
        """CSVを読み込み、自身のデータとして保持する"""
        df, ids = data_loader.load_csvs(filepaths)
        if df is not None:
            self.active_ids = ids
            self.time_series_df = df
            self.csv_replay_data = df
            return True, ids
        else:
            self.active_ids = []
            self.time_series_df = None
            self.csv_replay_data = None
            return False, []