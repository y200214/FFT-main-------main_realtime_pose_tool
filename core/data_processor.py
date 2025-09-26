# ファイル名: data_processor.py (新規作成)

import pandas as pd
import numpy as np
from constants import ALL_VARIABLES

class DataProcessor:
    """
    データフレームを受け取り、特徴量の計算を行う専門クラス。
    """
    def calculate_slope(self, values):
        """時系列データからFFTを行い、そのパワースペクトルの傾きを計算する"""
        n = len(values)
        if n < 4: return 0, None, None, None
        
        fft_result = np.fft.rfft(values)
        amplitude = np.abs(fft_result) / (n / 2)
        frequency = np.fft.rfftfreq(n, d=1.0)
        mask = (frequency > 0) & (amplitude > 0)
        
        if np.sum(mask) < 2: return 0, None, None, None
            
        log_freq = np.log10(frequency[mask])
        log_amp = np.log10(amplitude[mask])
        
        try:
            slope, intercept = np.polyfit(log_freq, log_amp, 1)
        except np.linalg.LinAlgError:
            return 0, None, None, None

        return slope, frequency[mask], amplitude[mask], intercept

    def get_features_from_df(self, df, active_ids):
        """【メインの計算ロジック】DataFrameから全IDの特徴量とスペクトルを計算する"""
        if df is None or df.empty or not active_ids:
            return pd.DataFrame(), {}

        feature_matrix = {id_name: {} for id_name in active_ids}
        power_spectrums = {id_name: {} for id_name in active_ids}

        for id_name in active_ids:
            slopes = {}
            for var in ALL_VARIABLES:
                column_name = f"{id_name}_{var}"
                if column_name in df.columns:
                    values = df[column_name].dropna().values
                    slope, freq, amp, intercept = self.calculate_slope(values)
                    slopes[var] = slope
                    if freq is not None:
                        power_spectrums[id_name][var] = (freq, amp, slope, intercept)
                else:
                    slopes[var] = 0
            feature_matrix[id_name] = slopes
        
        return pd.DataFrame(feature_matrix).T, power_spectrums

    def convert_history_to_df(self, history_slice, active_ids):
        """history形式のデータ(辞書のリスト)をDataFrameに変換する"""
        if not history_slice or not active_ids:
            return pd.DataFrame()

        records = []
        for dp in history_slice:
            record = {'timestamp': dp['timestamp']}
            for id_name in active_ids:
                if id_name in dp:
                    for var, value in dp[id_name].items():
                        record[f"{id_name}_{var}"] = value
            records.append(record)
        
        return pd.DataFrame(records).set_index('timestamp')