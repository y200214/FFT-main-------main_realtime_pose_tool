# ファイル名: views/kmeans_view.py (新規作成)

import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

class KmeansView(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        # --- 上部のコントロールフレーム ---
        ctrl_frame = ttk.Frame(self)
        ctrl_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(ctrl_frame, text="クラスタ数 (k):").pack(side=tk.LEFT, padx=(0, 5))

        self.k_value_var = tk.IntVar(value=3) # 初期値を3に設定
        self.k_slider = ttk.Scale(
            ctrl_frame,
            from_=2,
            to=10,
            orient=tk.HORIZONTAL,
            variable=self.k_value_var,
            command=self._on_k_slider_change
        )
        self.k_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.k_label_var = tk.StringVar(value="3")
        ttk.Label(ctrl_frame, textvariable=self.k_label_var, width=3).pack(side=tk.LEFT)

        # --- グラフ描画領域 ---
        self.fig = plt.figure(figsize=(16, 6))
        self.ax_sliding = self.fig.add_subplot(1, 2, 1)
        self.ax_full = self.fig.add_subplot(1, 2, 2)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _on_k_slider_change(self, value):
        """スライダーが動かされたときに呼ばれる"""
        k = int(float(value))
        self.k_value_var.set(k)
        self.k_label_var.set(str(k))
        # kの値が変わったら再描画をトリガーする
        self.controller._trigger_view_update()

    def update_plot(self, df_full, df_sliding, full_duration_seconds, sliding_duration_seconds):
        """データを受け取り、k-meansクラスタリングの結果をプロットする"""
        self.ax_full.clear()
        self.ax_sliding.clear()

        k = self.k_value_var.get()

        # 全区間データ
        title_full = f"全区間 k-means法 (N={full_duration_seconds:.0f}s, k={k})"
        self._perform_kmeans(self.ax_full, df_full, k, title_full)

        # スライディング窓データ
        title_sliding = f"直近{sliding_duration_seconds:.0f}秒 (N={sliding_duration_seconds:.0f}s, k={k})"
        self._perform_kmeans(self.ax_sliding, df_sliding, k, title_sliding)

        self.fig.tight_layout()
        self.canvas.draw()

    def _perform_kmeans(self, ax, df_features, k, title):
        """指定されたAxesにk-meansの結果を描画する"""
        if df_features is None or df_features.empty or len(df_features) < k:
            msg = f"クラスタリングには\nk={k}個以上のIDが必要です"
            ax.text(0.5, 0.5, msg, ha='center', va='center', fontsize=12, color='gray')
            ax.set_title(title)
            return

        try:
            df_processed = df_features.copy().fillna(0)
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(df_processed)

            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(scaled_features)

            # 主成分分析などで2次元に削減して可視化することも多いが、
            # ここではシンプルに特徴量の最初の2つで散布図を作成する
            ax.scatter(scaled_features[:, 0], scaled_features[:, 1], c=clusters, cmap='viridis', s=50, alpha=0.8)

            # 各点にIDラベルを付ける
            for i, label in enumerate(df_processed.index):
                ax.text(scaled_features[i, 0], scaled_features[i, 1], label, fontsize=8)

            ax.set_title(title)
            ax.set_xlabel("特徴量1 (標準化後)")
            ax.set_ylabel("特徴量2 (標準化後)")
            ax.grid(True)

        except Exception as e:
            error_msg = f'描画エラー:\n{e}'
            ax.text(0.5, 0.5, error_msg, ha='center', va='center', color='red')
            ax.set_title(title)
            print(f"ERROR: [{title}] の描画中にエラーが発生しました: {e}")

    def save_plot(self, output_folder, all_data, progress_callback, timestamp, cancel_check):
        """k-meansビューの保存処理（現時点では未実装）"""
        print("INFO: KmeansViewの保存は現在実装されていません。スキップします。")
        # 将来的に実装する場合は、ここに処理を記述する
        # プログレスバーを進めるためにコールバックを呼ぶ
        if progress_callback:
            progress_callback()







