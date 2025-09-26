# ファイル名: data_loader.py (新しく作成)

import pandas as pd
import time
from constants import ALL_VARIABLES

def load_csvs(filepaths):
    """複数CSVファイルを読み込み、横に結合して一つのデータフレームにする"""
    all_dataframes = {}
    loaded_ids = []
    
    for filepath in filepaths:
        try:
            import re
            match = re.search(r'ID_(\d+)', filepath, re.IGNORECASE)
            if not match:
                print(f"警告: ファイル名'{filepath}'からIDを推定できませんでした。スキップします。")
                continue
            
            target_id = f"ID_{match.group(1)}"
            if target_id in loaded_ids:
                print(f"警告: ID'{target_id}'が重複しています。スキップします。")
                continue

            loaded_ids.append(target_id)
            
            df = pd.read_csv(filepath)
            
            # あいまい検索で列名をマッピング
            column_mapping = {}
            temp_df = pd.DataFrame()
            for var in ALL_VARIABLES:
                for col in df.columns:
                    if var in col.lower():
                        temp_df[var] = df[col]
                        break
            
            all_dataframes[target_id] = temp_df

        except Exception as e:
            print(f"'{filepath}'の処理中にエラーが発生しました: {e}")
            
    if not all_dataframes:
        return None, []
        
    # 各IDのデータフレームの列名の先頭にIDを付与
    prepared_dfs = []
    for target_id, df in all_dataframes.items():
        df.columns = [f"{target_id}_{col}" for col in df.columns]
        prepared_dfs.append(df)
        
    # 全てのデータフレームを横に結合(axis=1)
    time_series_df = pd.concat(prepared_dfs, axis=1)
    # タイムスタンプとして行番号のインデックスを使用
    time_series_df.index.name = 'timestamp'
    
    return time_series_df, sorted(loaded_ids)















