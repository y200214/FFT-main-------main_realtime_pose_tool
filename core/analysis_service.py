# ファイル名: core/analysis_service.py (新規作成)

import pandas as pd

class AnalysisService:
    """
    データ処理と解析の実行を専門に担当するサービスクラス。
    Controllerからビジネスロジックを分離する。
    """
    def __init__(self, model, data_processor):
        self.model = model
        self.data_processor = data_processor

    def process_and_store_features(self, full_slice, sliding_slice=None):
        """
        特徴量を計算し、結果をモデルに格納する。
        (Controllerからロジックを移動)
        """
        # --- 全区間データの計算 ---
        df_full = self.data_processor.convert_history_to_df(full_slice, self.model.active_ids)
        df_full_features, ps_full = self.data_processor.get_features_from_df(df_full, self.model.active_ids)

        # --- スライディング窓データの計算 ---
        df_sliding_features, ps_sliding = pd.DataFrame(), {}
        if sliding_slice:
            df_sliding = self.data_processor.convert_history_to_df(sliding_slice, self.model.active_ids)
            df_sliding_features, ps_sliding = self.data_processor.get_features_from_df(df_sliding, self.model.active_ids)
            
        # --- 計算結果をModelに保存 ---
        self.model.last_slope_dfs = {'sliding': df_sliding_features, 'full': df_full_features}
        self.model.last_power_spectrums = {'sliding': ps_sliding, 'full': ps_full}
        
        # --- 一括解析用に計算結果を返す ---
        return df_full_features, ps_full

    def perform_batch_analysis(self, all_data_history):
        """
        一括解析の重い計算処理を実行する。
        (Controllerの別スレッド処理からロジックを移動)
        """
        df_full = self.data_processor.convert_history_to_df(all_data_history, self.model.active_ids)
        df_full_features, ps_full = self.data_processor.get_features_from_df(df_full, self.model.active_ids)
        
        # --- Modelに再生用・保存用データを格納 ---
        self.model.full_history = all_data_history
        self.model.last_slope_dfs = {'full': df_full_features, 'sliding': pd.DataFrame()}
        self.model.last_power_spectrums = {'full': ps_full, 'sliding': {}}
        
        return df_full_features