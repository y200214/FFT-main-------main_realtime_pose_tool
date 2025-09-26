# ファイル名: app/views/components/control_panel.py

import tkinter as tk
from tkinter import ttk

class ControlPanel(ttk.Frame):
    """モード選択や実行ボタンなどをまとめたコントロールパネルのUIコンポーネント"""
    def __init__(self, parent, controller, app):
        super().__init__(parent)
        self.controller = controller
        self.app = app

        control_panel = ttk.LabelFrame(self, text="コントロールパネル")
        control_panel.pack(side=tk.LEFT)
        
        # --- モード選択 ---
        mode_frame = ttk.LabelFrame(control_panel, text="モード")
        mode_frame.pack(side=tk.LEFT, padx=5, pady=2, fill='y')

        ttk.Radiobutton(mode_frame, text="リアルタイム", variable=self.app.mode, value="realtime", command=self.controller._on_mode_change).pack(anchor='w')
        ttk.Radiobutton(mode_frame, text="CSV", variable=self.app.mode, value="csv", command=self.controller._on_mode_change).pack(anchor='w')


        # --- CSVコントロール ---
        csv_frame = ttk.LabelFrame(control_panel, text="CSVコントロール")
        csv_frame.pack(side=tk.LEFT, padx=5, pady=2, fill='y')
        self.app.load_csv_button = ttk.Button(csv_frame, text="CSV読込...", command=self.controller.load_csvs)
        self.app.load_csv_button.pack(side=tk.TOP, padx=5, pady=2)
        self.app.batch_button = ttk.Button(csv_frame, text="一括解析", state="disabled", command=self.controller._run_batch_analysis)
        self.app.batch_button.pack(side=tk.TOP, padx=5, pady=2)
        self.app.reset_button = ttk.Button(csv_frame, text="データリセット", command=self.controller.reset_all_data)
        self.app.reset_button.pack(side=tk.TOP, padx=5, pady=2)

        # --- 時間範囲選択 ---
        time_frame = ttk.LabelFrame(control_panel, text="時間範囲")
        time_frame.pack(side=tk.LEFT, padx=5, pady=2, fill='y')
        ttk.Radiobutton(time_frame, text="30秒窓", variable=self.app.time_range_var, value="30秒窓", command=self.controller._trigger_view_update).pack(anchor='w')
        ttk.Radiobutton(time_frame, text="全区間", variable=self.app.time_range_var, value="全区間", command=self.controller._trigger_view_update).pack(anchor='w')

        # --- 実行コントロール ---
        exec_frame = ttk.LabelFrame(control_panel, text="実行コントロール")
        exec_frame.pack(side=tk.LEFT, padx=5, pady=2, fill='y')
        self.app.start_button = ttk.Button(exec_frame, text="解析開始", command=self.controller.start_analysis)
        self.app.start_button.pack(side=tk.LEFT, padx=5)
        self.app.pause_button = ttk.Button(exec_frame, text="一時停止", state="disabled", command=self.controller.toggle_pause)
        self.app.pause_button.pack(side=tk.LEFT, padx=5)
        self.app.stop_button = ttk.Button(exec_frame, text="停止", state="disabled", command=self.controller.stop_analysis)
        self.app.stop_button.pack(side=tk.LEFT, padx=5)
        self.app.save_button = ttk.Button(exec_frame, text="グラフ保存", command=self.controller.save_plots)
        self.app.save_button.pack(side=tk.LEFT, padx=5)
        ttk.Button(exec_frame, text="特徴量をCSV保存", command=self.controller.save_features_to_csv).pack(side=tk.LEFT, padx=5)

        # --- その他フレーム ---
        other_frame = ttk.LabelFrame(control_panel, text="その他")
        other_frame.pack(side=tk.LEFT, padx=5, pady=2, fill='y')
        self.app.settings_button = ttk.Button(other_frame, text="設定...", command=self.controller.open_settings_dialog)
        self.app.settings_button.pack(side=tk.LEFT, padx=5, pady=13)