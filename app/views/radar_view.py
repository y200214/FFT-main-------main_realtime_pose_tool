# ファイル名: views/radar_view.py (最終版)

import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from constants import EMOTION_VARS, BEHAVIOR_VARS
import os


class RadarView(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.max_val = 0.6

        # --- UI要素の構築 ---
        ctrl_frame = ttk.Frame(self)
        ctrl_frame.pack(fill=tk.X, pady=5, padx=5)

        # --- レーダーチャートの表示オプション ---
        self.show_values_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctrl_frame, text="数値を表示", variable=self.show_values_var, command=self._trigger_controller_update).pack(side=tk.LEFT, padx=20)
        
        # --- 最大値の調整 ---
        max_val_frame = ttk.Frame(ctrl_frame)
        max_val_frame.pack(side=tk.RIGHT, padx=20)
        
        ttk.Button(max_val_frame, text="-", command=self._decrease_max_val, width=2).pack(side=tk.LEFT)
        
        self.max_val_entry_var = tk.StringVar(value=str(self.max_val))
        entry = ttk.Entry(max_val_frame, textvariable=self.max_val_entry_var, width=5)
        entry.pack(side=tk.LEFT)
        
        ttk.Button(max_val_frame, text="適用", command=self._apply_max_val).pack(side=tk.LEFT, padx=(0,5))
        
        ttk.Button(max_val_frame, text="+", command=self._increase_max_val, width=2).pack(side=tk.LEFT)
        
        # --- グラフ描画領域の作成 (正しい方法) ---
        self.fig = plt.figure(figsize=(14, 6))
        self.ax_emotion = self.fig.add_subplot(1, 2, 1, polar=True)
        self.ax_behavior = self.fig.add_subplot(1, 2, 2, polar=True)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        # グラフの余白(上下左右)と、グラフ間のスペースを手動で設定
        self.fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.5)

    def update_plot(self, slope_dfs=None):
        if slope_dfs is None:
            slope_dfs = self.controller.model.last_slope_dfs
        self.ax_emotion.clear()
        self.ax_behavior.clear()
        
        # 常にメインウィンドウ(app)の共通変数を参照します 
        time_range = self.controller.app.time_range_var.get()
        key = 'sliding' if time_range == "30秒窓" else 'full'
        
        if slope_dfs and key in slope_dfs:
            df = slope_dfs[key]
            if not df.empty:
                df_abs = df.abs()
                show_values_flag = self.show_values_var.get() 
                df_emotion = df_abs[EMOTION_VARS]
                self._plot_radar(df_emotion, self.ax_emotion, f"感情 (EMOTION) - {time_range}", show_values=show_values_flag)
                df_behavior = df_abs[BEHAVIOR_VARS]
                self._plot_radar(df_behavior, self.ax_behavior, f"行動 (BEHAVIOR) - {time_range}", show_values=show_values_flag)
                
        self.canvas.draw()

    def _plot_radar(self, df, ax, title, show_values=True):
        """レーダーチャートを1つ描画するヘルパー関数"""
        labels = df.columns
        num_vars = len(labels)
        
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles_closed = angles + angles[:1]
        
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        
        ax.set_xticks(angles)
        ax.set_xticklabels(labels, fontsize=8)

        ax.set_ylim(0, self.max_val)
        ax.set_title(title, pad=25)
        
        for i, row_name in enumerate(df.index):
            stats = df.iloc[i].values.tolist()
            stats_closed = stats + stats[:1]
            ax.plot(angles_closed, stats_closed, label=row_name)
            ax.fill(angles_closed, stats_closed, alpha=0.1)
            
            if show_values:
                for angle, value in zip(angles, stats):
                    ax.text(angle, value + 0.05, f"{value:.2f}",
                            ha='center', va='center', fontsize=7, color='black')
                
        if not df.empty:
            ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.15), fontsize='small')

    def _apply_max_val(self):
        """入力ボックスの値をグラフの最大値に適用する"""
        try:
            new_max = float(self.max_val_entry_var.get())
            if new_max > 0:
                self.max_val = new_max
                self._trigger_controller_update() # ★変更
            else:
                messagebox.showwarning("入力エラー", "最大値は0より大きい数値を入力してください。")
        except ValueError:
            messagebox.showerror("入力エラー", "数値を入力してください。")
            self.max_val_entry_var.set(str(self.max_val))

    def _increase_max_val(self):
        self.max_val += 0.1
        self.max_val_entry_var.set(f"{self.max_val:.1f}")
        self._trigger_controller_update() # ★変更

    def _decrease_max_val(self):
        self.max_val = max(0.1, self.max_val - 0.1)
        self.max_val_entry_var.set(f"{self.max_val:.1f}")
        self._trigger_controller_update() # ★変更

    def _trigger_controller_update(self):
        """UI操作があったことをControllerに通知し、再描画を依頼する"""
        self.controller._trigger_view_update()

# views/radar_view.py

    def save_plot(self, output_folder, all_data, progress_callback, timestamp, cancel_check):
        """
        データを受け取り、IDごとに単一のレーダーチャート画像をファイルに保存する。
        """
        print("INFO: レーダーチャートの一括保存を開始します。")
        slope_df = all_data.get('slope_dfs', {}).get('full')

        if slope_df is None or slope_df.empty:
            if progress_callback:
                num_ids = len(self.controller.model.active_ids)
                for _ in range(num_ids):
                    progress_callback()
            return

        for id_name in slope_df.index:
            # ★追加: ループの先頭でキャンセルされたかチェックします
            if cancel_check():
                print("INFO: レーダーチャートの保存がキャンセルされました。")
                return

            temp_fig, (temp_ax1, temp_ax2) = plt.subplots(1, 2, figsize=(14, 6), subplot_kw=dict(polar=True))
            
            df_single_id = slope_df.loc[[id_name]]
            
            df_emotion = df_single_id[EMOTION_VARS].abs()
            self._plot_radar(df_emotion, temp_ax1, f"感情 (EMOTION) - {id_name}", show_values=True)
            
            df_behavior = df_single_id[BEHAVIOR_VARS].abs()
            self._plot_radar(df_behavior, temp_ax2, f"行動 (BEHAVIOR) - {id_name}", show_values=True)
            
            id_folder = os.path.join(output_folder, id_name)
            os.makedirs(id_folder, exist_ok=True)
            
            file_path = os.path.join(id_folder, f"レーダーチャート_{id_name}.png")

            temp_fig.suptitle(f"Saved at: {timestamp:.1f} sec", fontsize=10, y=0.02, ha='right')
            
            temp_fig.savefig(file_path, dpi=150)
            
            plt.close(temp_fig)
            if progress_callback:
                progress_callback()




