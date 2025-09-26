# ファイル名: views/heatmap_view.py (新規作成)

import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import pandas as pd

class HeatmapView(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        # --- グラフ描画領域 ---
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(1, 1, 1)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def update_plot(self, df_full, df_sliding, full_duration_seconds, sliding_duration_seconds):
        """データを受け取り、ヒートマップを更新する"""
        self.ax.clear()

        # ヒートマップは情報量の多い全区間データのみを対象とする
        df_to_plot = df_full

        if df_to_plot is None or df_to_plot.empty:
            self.ax.text(0.5, 0.5, "データがありません", ha='center', va='center', fontsize=12, color='gray')
            self.ax.set_title("特徴量ヒートマップ (全区間)")
            self.canvas.draw()
            return

        try:
            # データを正規化 (0から1の範囲に) してからプロットすると見やすい
            df_normalized = (df_to_plot - df_to_plot.min()) / (df_to_plot.max() - df_to_plot.min())

            sns.heatmap(
                df_normalized.fillna(0),
                ax=self.ax,
                annot=True,       # 数値を表示
                fmt=".2f",        # 小数点以下2桁まで
                cmap='viridis',   # 色のテーマ
                linewidths=.5
            )
            self.ax.set_title("特徴量ヒートマップ (全区間)")
            self.ax.tick_params(axis='y', labelrotation=0)

        except Exception as e:
            error_msg = f'描画エラー:\n{e}'
            self.ax.text(0.5, 0.5, error_msg, ha='center', va='center', color='red')
            print(f"ERROR: ヒートマップの描画中にエラーが発生しました: {e}")

        self.fig.tight_layout()
        self.canvas.draw()

    def save_plot(self, output_folder, all_data, progress_callback, timestamp, cancel_check):
        """ヒートマップビューの保存処理（現時点では未実装）"""
        print("INFO: HeatmapViewの保存は現在実装されていません。スキップします。")
        # 将来的に実装する場合は、ここに処理を記述する
        # プログレスバーを進めるためにコールバックを呼ぶ
        if progress_callback:
            progress_callback()









