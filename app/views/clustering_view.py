# ファイル名: views/clustering_view.py

import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import StandardScaler
import os

class ClusteringView(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        # --- 上部のコントロールフレーム ---
        ctrl_frame = ttk.Frame(self)
        ctrl_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(ctrl_frame, text="クラスタリング手法:").pack(side=tk.LEFT, padx=(0, 5))

        # 手法選択用のドロップダウンメニュー
        self.clustering_method_var = tk.StringVar(value='ward') # 初期値を 'ward' に設定
        method_options = ['ward', 'single', 'complete', 'average']
        self.method_combobox = ttk.Combobox(
            ctrl_frame,
            textvariable=self.clustering_method_var,
            values=method_options,
            state='readonly', # ユーザーによる自由入力を禁止
            width=15
        )
        self.method_combobox.pack(side=tk.LEFT)
        # ドロップダウンが選択されたら、コントローラーに通知する
        self.method_combobox.bind('<<ComboboxSelected>>', self._on_method_change)
        # ★★★ 追加ここまで ★★★

        # --- グラフ描画領域 ---
        self.fig = plt.figure(figsize=(16, 6))
        self.ax_sliding = self.fig.add_subplot(1, 2, 1)
        self.ax_full = self.fig.add_subplot(1, 2, 2)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect('button_press_event', self.on_click)

    def _on_method_change(self, event=None):
        """ドロップダウンメニューで手法が変更されたときに呼ばれる"""
        print(f"INFO: クラスタリング手法を '{self.clustering_method_var.get()}' に変更しました。")
        self.controller._trigger_view_update() # 再描画をトリガー

    def on_click(self, event):
        """デンドログラムのラベルクリックを検知してコントローラーに通知する"""
        if event.inaxes not in [self.ax_sliding, self.ax_full]:
            return
        
        ax = event.inaxes
        for label in ax.get_xticklabels():
            if label.get_window_extent().contains(event.x, event.y):
                clicked_id = label.get_text()
                if hasattr(self.controller, 'set_focused_id'):
                    self.controller.set_focused_id(clicked_id)
                break

    def update_plot(self, df_full, df_sliding, full_duration_seconds, sliding_duration_seconds):
        """
        データフレームと期間を受け取り、デンドログラムを更新する。
        """
        self.ax_full.clear()
        self.ax_sliding.clear()

        # ★★★ ここから修正 ★★★
        # 現在選択されている手法を取得
        selected_method = self.clustering_method_var.get()

        # --- 1. 全区間データのクラスタリング (右側のグラフ) ---
        title_full = f"全区間 階層型クラスタリング (N={full_duration_seconds:.0f}s, method='{selected_method}')"
        self._perform_clustering(self.ax_full, df_full, title_full, method=selected_method)

        # --- 2. スライディング窓データのクラスタリング (左側のグラフ) ---
        title_sliding = f"直近{sliding_duration_seconds:.0f}秒 (N={sliding_duration_seconds:.0f}s, method='{selected_method}')"
        self._perform_clustering(self.ax_sliding, df_sliding, title_sliding, method=selected_method)
        # ★★★ 修正ここまで ★★★
        
        self.fig.tight_layout()
        self.canvas.draw()

    def _perform_clustering(self, ax, df_features, title, method='ward'):
        """
        指定されたAxesに、指定された手法でクラスタリング結果を描画する。
        """
        if df_features is None or df_features.empty or len(df_features) < 2:
            msg = "クラスタリングには\n2つ以上のIDが必要です"
            ax.text(0.5, 0.5, msg, ha='center', va='center', fontsize=12, color='gray')
            ax.set_title(title)
            return

        try:
            df_processed = df_features.copy()
            df_processed.fillna(0, inplace=True)
            id_labels = df_processed.index.tolist()
            
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(df_processed)
            
            # ★★★ 引数で受け取った手法を使用する ★★★
            linkage_result = linkage(scaled_features, method=method)

            dendrogram(
                linkage_result,
                labels=id_labels,
                orientation='top',
                ax=ax
            )
            ax.set_title(title)
            ax.set_ylabel("クラスタ間距離")

        except Exception as e:
            error_msg = f'描画エラー:\n{e}'
            ax.text(0.5, 0.5, error_msg, ha='center', va='center', color='red')
            ax.set_title(title)
            print(f"ERROR: [{title}] の描画中にエラーが発生しました: {e}")

        ax.tick_params(axis='x', labelrotation=90)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    def save_plot(self, output_folder, all_data, progress_callback, timestamp, cancel_check):
        """
        全IDのデータを受け取り、それを元に新しいクラスタリンググラフを描画して保存する。
        保存する手法は、UIで現在選択されているものを採用する。
        """
        print("INFO: 全IDでのクラスタリンググラフの保存を開始します。")
        try:
            df_full_all_ids = all_data.get('slope_dfs', {}).get('full')

            if df_full_all_ids is None or df_full_all_ids.empty or len(df_full_all_ids) < 2:
                print("WARN: 保存するクラスターデータが2件未満のため、スキップします。")
                return

            # UIで選択されている手法を取得
            selected_method = self.clustering_method_var.get()
            
            temp_fig, temp_ax = plt.subplots(figsize=(10, 7))
            
            duration = timestamp
            title = f"全区間 階層型クラスタリング (N={duration:.0f}s, method='{selected_method}')"
            
            # 選択されている手法でクラスタリングを描画
            self._perform_clustering(temp_ax, df_full_all_ids, title, method=selected_method)
            
            temp_fig.suptitle(f"Saved at: {timestamp:.1f} sec", fontsize=10, y=0.03, ha='right')

            temp_fig.tight_layout(rect=[0, 0.05, 1, 1])
            file_path = os.path.join(output_folder, f"クラスター_全ID_{selected_method}.png") # ファイル名に手法名を追加
            temp_fig.savefig(file_path, dpi=150)
            
            plt.close(temp_fig)

            print(f"INFO: 全IDクラスターグラフを保存しました: {file_path}")

        except Exception as e:
            print(f"ERROR: クラスタリンググラフの保存中にエラーが発生しました: {e}")
            if 'temp_fig' in locals():
                plt.close(temp_fig)

    def _save_single_ax(self, ax, filepath):
        """指定されたAxesオブジェクトのみをファイルに保存します。"""
        self.fig.tight_layout()
        bbox = ax.get_tightbbox(self.fig.canvas.get_renderer()).transformed(self.fig.dpi_scale_trans.inverted())
        self.fig.savefig(filepath, bbox_inches=bbox, dpi=150)