# spectrum_view.py

import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from constants import ALL_VARIABLES, EMOTION_VARS, BEHAVIOR_VARS
import os

class SpectrumView(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.spectrum_window = None # 別ウィンドウ管理用

        # 【修正】属性アクセスで設定値を取得
        fft_config = self.controller.config_manager.config.fft_initial_view
        initial_group = fft_config.variable_group
        initial_show_fit = fft_config.show_fit_line

        # --- 上部の常に表示されるコントロールフレーム ---
        top_controls_frame = ttk.Frame(self)
        top_controls_frame.pack(fill=tk.X, padx=5, pady=(5,0))

        # --- 表示変数グループ (常に表示) ---
        variable_group_frame = ttk.LabelFrame(top_controls_frame, text="表示変数グループ")
        variable_group_frame.pack(side=tk.LEFT, padx=(5, 10), pady=2)
        # 【修正】初期値を設定から読み込んだ値にする
        self.variable_group_var = tk.StringVar(value=initial_group)
        ttk.Radiobutton(variable_group_frame, text="全て", variable=self.variable_group_var, value="all", command=self._on_variable_group_change).pack(anchor='w')
        ttk.Radiobutton(variable_group_frame, text="感情のみ", variable=self.variable_group_var, value="emotion", command=self._on_variable_group_change).pack(anchor='w')
        ttk.Radiobutton(variable_group_frame, text="行動のみ", variable=self.variable_group_var, value="behavior", command=self._on_variable_group_change).pack(anchor='w')

        # --- 表示オプション (常に表示) ---
        # 【修正】初期値を設定から読み込んだ値にする
        self.show_fit_var = tk.BooleanVar(value=initial_show_fit)
        fit_check = ttk.Checkbutton(top_controls_frame, text="近似直線を表示", variable=self.show_fit_var, command=self._trigger_update)
        fit_check.pack(side=tk.LEFT, padx=20, pady=2, anchor='s')

        # --- 別ウィンドウ表示ボタン (追加) ---
        self.open_window_button = ttk.Button(top_controls_frame, text="別ウィンドウで表示", command=self._open_spectrum_window)
        self.open_window_button.pack(side=tk.LEFT, padx=20, pady=2, anchor='s')

        # --- 表示/非表示ボタン (常に表示) ---
        self.toggle_button = ttk.Button(top_controls_frame, text="◀ コントロールを隠す", command=self.toggle_controls)
        self.toggle_button.pack(side=tk.RIGHT)

        # --- 表示/非表示が切り替わるコントロールフレーム ---
        self.ctrl_frame = ttk.Frame(self)
        self.ctrl_frame.pack(fill=tk.X, pady=5, padx=5)

        # --- 変数個別選択 ---
        self.var_select_frame = ttk.LabelFrame(self.ctrl_frame, text="個別変数選択")
        self.var_select_frame.pack(side=tk.LEFT, padx=5, pady=2, fill='y')
        self.param_vars = {}
        for param_name in ALL_VARIABLES:
            var = tk.BooleanVar(value=True)
            ttk.Checkbutton(self.var_select_frame, text=param_name, variable=var, command=self._trigger_update).pack(anchor='w')
            self.param_vars[param_name] = var

        # --- グラフ描画領域 ---
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _open_spectrum_window(self):
        """スペクトルグラフを別ウィンドウで開く"""
        if self.spectrum_window and self.spectrum_window.winfo_exists():
            self.spectrum_window.lift() # 既に開いていれば最前面に表示
            return

        self.spectrum_window = tk.Toplevel(self)
        self.spectrum_window.title("パワースペクトル詳細")
        self.spectrum_window.geometry("800x600")

        # 新しいウィンドウ用のFigureとCanvasを作成
        self.fig_new = plt.figure()
        self.ax_new = self.fig_new.add_subplot(1, 1, 1)
        self.canvas_new = FigureCanvasTkAgg(self.fig_new, master=self.spectrum_window)
        self.canvas_new.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # ウィンドウが閉じられたときの処理
        self.spectrum_window.protocol("WM_DELETE_WINDOW", self._on_spectrum_window_close)

        # 現在のデータで即座に描画
        self.update_plot()

    def _on_spectrum_window_close(self):
        """別ウィンドウが閉じられたときのクリーンアップ処理"""
        if self.spectrum_window:
            self.spectrum_window.destroy()
            self.spectrum_window = None
            # MatplotlibのFigureリソースも解放
            plt.close(self.fig_new)

    def _draw_spectrum_on_ax(self, ax, power_spectrums):
        """指定されたAxesオブジェクトにグラフを描画する共通ヘルパー"""
        ax.clear()

        selected_params = [name for name, var in self.param_vars.items() if var.get()]
        time_range_key = 'sliding' if self.controller.app.time_range_var.get() == "30秒窓" else 'full'

        if power_spectrums:
            spectrum_data = power_spectrums.get(time_range_key, {})
            equations = []
            ids_to_plot = spectrum_data.keys()

            for id_name in ids_to_plot:
                for param_name in selected_params:
                    if id_name in spectrum_data and param_name in spectrum_data.get(id_name, {}):
                        freq, amp, slope, intercept = spectrum_data[id_name][param_name]
                        if freq is None or len(freq) == 0: continue

                        line, = ax.loglog(freq, amp, label=f"{id_name}_{param_name}")

                        if self.show_fit_var.get() and slope is not None and intercept is not None:
                            line_color = line.get_color()
                            log_freq = np.log10(freq)
                            fit_line = 10**(slope * log_freq + intercept)
                            ax.loglog(freq, fit_line, '--', color=line_color)
                            equations.append(f"{id_name}_{param_name}: y={slope:.2f}x+{intercept:.2f}")

            if ax.has_data():
                ax.legend(fontsize='small')
                if self.show_fit_var.get() and equations:
                    ax.text(0.98, 0.02, "\n".join(equations), transform=ax.transAxes, fontsize=8, verticalalignment='bottom', horizontalalignment='right', bbox={'boxstyle': 'round', 'facecolor': 'wheat', 'alpha': 0.5})
            ax.grid(True, which="both", ls="--")
        else:
            ax.text(0.5, 0.5, "データがありません", ha='center', va='center')

        title = f"パワースペクトル ({self.controller.app.time_range_var.get()})"
        ax.set_title(title)
        ax.set_xlabel("Frequency (log)")
        ax.set_ylabel("Amplitude (log)")

    def update_plot(self, power_spectrums=None):
        if power_spectrums is None:
            power_spectrums = self.controller.model.last_power_spectrums

        # メインウィンドウのタブ内グラフを描画
        self._draw_spectrum_on_ax(self.ax, power_spectrums)
        self.fig.tight_layout()
        self.canvas.draw()

        # 別ウィンドウが開いていれば、そちらも更新
        if self.spectrum_window and self.spectrum_window.winfo_exists():
            self._draw_spectrum_on_ax(self.ax_new, power_spectrums)
            self.fig_new.tight_layout()
            self.canvas_new.draw()

    def _trigger_update(self):
        """UI操作をコントローラーに通知する"""
        self.controller._trigger_view_update()

    def _on_variable_group_change(self):
        """変数グループのラジオボタンが変更されたときの処理"""
        selection = self.variable_group_var.get()

        target_vars = []
        if selection == "all":
            target_vars = ALL_VARIABLES
        elif selection == "emotion":
            target_vars = EMOTION_VARS
        elif selection == "behavior":
            target_vars = BEHAVIOR_VARS

        for param_name, var in self.param_vars.items():
            var.set(param_name in target_vars)

        self._trigger_update()

    def set_all_variable_checkboxes(self, state=True):
        """Controllerからの指示で、変数のチェックボックスをすべてON/OFFする"""
        if state:
            self.variable_group_var.set("all")
        self._on_variable_group_change()

    def toggle_controls(self):
        """コントロールパネルの表示/非表示を切り替える"""
        if self.ctrl_frame.winfo_viewable():
            self.ctrl_frame.pack_forget()
            self.toggle_button.config(text="▶ コントロールを表示")
        else:
            self.ctrl_frame.pack(fill=tk.X, pady=5, padx=5, before=self.canvas.get_tk_widget())
            self.toggle_button.config(text="◀ コントロールを隠す")

    def save_plot(self, output_folder, all_data, progress_callback, timestamp, cancel_check):
        """
        データを受け取り、IDと変数ごとに個別のスペクトルグラフをファイルに保存する。
        """
        print("INFO: スペクトルグラフの一括保存を開始します。")
        power_spectrums_data = all_data.get('power_spectrums')

        if not power_spectrums_data:
            num_ids = len(self.controller.model.active_ids)
            if progress_callback and num_ids > 0:
                num_spectrum_vars = len(ALL_VARIABLES)
                for _ in range(num_ids * num_spectrum_vars):
                    progress_callback()
            return

        spectrum_data = power_spectrums_data.get('full', {})
        all_ids = self.controller.model.active_ids
        all_vars = ALL_VARIABLES

        for id_name in all_ids:
            if cancel_check():
                print("INFO: スペクトルグラフの保存がキャンセルされました。")
                return

            id_folder = os.path.join(output_folder, id_name, "FFT")
            os.makedirs(id_folder, exist_ok=True)

            for param_name in all_vars:
                if cancel_check():
                    print("INFO: スペクトルグラフの保存がキャンセルされました。")
                    return

                if id_name in spectrum_data and param_name in spectrum_data[id_name]:
                    data = spectrum_data[id_name][param_name]
                    freq, amp, slope, intercept = data
                    if freq is None or len(freq) == 0:
                        if progress_callback: progress_callback()
                        continue

                    temp_fig, temp_ax = plt.subplots(figsize=(8, 6))
                    temp_ax.loglog(freq, amp, label=f"{id_name}_{param_name}")

                    if slope is not None and intercept is not None:
                        log_freq = np.log10(freq)
                        fit_line = 10**(slope * log_freq + intercept)
                        temp_ax.loglog(freq, fit_line, '--')
                        equation = f"y={slope:.2f}x+{intercept:.2f}"
                        temp_ax.set_title(f"パワースペクトル - {id_name}_{param_name}\n({equation})")
                    else:
                        temp_ax.set_title(f"パワースペクトル - {id_name}_{param_name}")

                    temp_ax.set_xlabel("Frequency (log)")
                    temp_ax.set_ylabel("Amplitude (log)")
                    temp_ax.grid(True, which="both", ls="--")
                    temp_ax.legend()

                    file_path = os.path.join(id_folder, f"スペクトル_{id_name}_{param_name}.png")

                    temp_fig.suptitle(f"Saved at: {timestamp:.1f} sec", fontsize=10, y=0.02, ha='right')

                    temp_fig.savefig(file_path, dpi=150)
                    plt.close(temp_fig)

                if progress_callback:
                    progress_callback()