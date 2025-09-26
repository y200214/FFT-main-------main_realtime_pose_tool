# ファイル名: app/views/components/playback_panel.py

import tkinter as tk
from tkinter import ttk

class PlaybackPanel(ttk.Frame):
    """再生スライダーや時間表示のUIコンポーネント"""
    def __init__(self, parent, controller, app):
        super().__init__(parent)
        self.controller = controller
        self.app = app

        self.app.progress_bar = ttk.Progressbar(self, variable=self.app.progress_var)

        self.app.slider = ttk.Scale(self, from_=0, to=100, orient=tk.HORIZONTAL)
        self.app.slider.pack(fill=tk.X, expand=True, pady=2)
        self.app.slider.bind("<ButtonRelease-1>", self.controller._on_slider_change)
        
        playback_info_frame = ttk.Frame(self)
        playback_info_frame.pack(fill=tk.X, expand=True)
        
        ttk.Label(playback_info_frame, textvariable=self.app.elapsed_time_var).pack(side=tk.LEFT)
        
        playback_input_frame = ttk.Frame(playback_info_frame)
        playback_input_frame.pack(side=tk.RIGHT)

        time_entry = ttk.Entry(playback_input_frame, textvariable=self.app.time_input_var, width=6, justify='right')
        time_entry.pack(side=tk.LEFT)
        time_entry.bind("<Return>", self.controller.on_time_input_enter)
        
        ttk.Label(playback_input_frame, textvariable=self.app.total_time_var).pack(side=tk.LEFT)

        # --- ここから修正 ---
        self.app.rt_button = ttk.Button(self, text="リアルタイム表示に戻る", state="disabled", command=self.controller._return_to_realtime)
        self.app.rt_button.pack(pady=2)
        # --- 修正ここまで ---