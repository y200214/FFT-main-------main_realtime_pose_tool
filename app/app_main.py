# ファイル名: app/app_main.py

import platform
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
import multiprocessing

from .views.clustering_view import ClusteringView
from .views.spectrum_view import SpectrumView
from .views.radar_view import RadarView
from .views.kmeans_view import KmeansView
from .views.heatmap_view import HeatmapView
from .views.video_view import VideoView
from .controller import AppController
from .ui_manager import UIManager
from .views.components.focus_panel import FocusPanel
from .views.components.control_panel import ControlPanel
from .views.components.playback_panel import PlaybackPanel

class AppMainWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("リアルタイム解析ダッシュボード")
        self.geometry("1400x900")
        
        self.data_queue = multiprocessing.Queue()
        self.frame_queue = multiprocessing.Queue(maxsize=2)
        self.status_queue = multiprocessing.Queue()
        
        self.config_dialog = None 
        self.protocol("WM_DELETE_WINDOW", self.on_app_closing)

        # UIで直接使う変数を定義
        self.mode = tk.StringVar(value="csv")
        self.elapsed_time_var = tk.StringVar(value="経過時間: 0.0s")
        self.progress_var = tk.DoubleVar()
        self.time_range_var = tk.StringVar(value="30秒窓")
        self.time_input_var = tk.StringVar()
        self.total_time_var = tk.StringVar()
        
        # --- 初期化順序を修正 ---
        # 1. Controllerインスタンスを作成
        self.controller = AppController(self, status_queue=self.status_queue)
        
        # 2. UIの構築（ここでself.viewsが作成される）
        self._setup_ui()

        # 3. UIManagerのインスタンスを作成し、Controllerに参照を渡す
        self.ui_manager = UIManager(self)
        self.controller.set_ui_manager(self.ui_manager)

        # 4. UIの初期状態を設定
        self.controller._on_mode_change()
        
        # 5. 全ての準備が整った後、カメラ監視を開始
        if not self.controller.camera_monitor.is_alive():
            self.controller.camera_monitor.start()
        # --- 修正ここまで ---

    def _setup_ui(self):
        focus_panel_container = FocusPanel(self, self.controller, self)
        focus_panel_container.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(5,0))

        top_panel = ttk.Frame(self)
        top_panel.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        ControlPanel(top_panel, self.controller, self).pack(side=tk.LEFT)

        center_panel = PlaybackPanel(self, self.controller, self)
        center_panel.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(0, 5))

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.views = {
            "video": VideoView(self.notebook, self.controller), 
            "clustering": ClusteringView(self.notebook, self.controller),
            "spectrum": SpectrumView(self.notebook, self.controller),
            "radar": RadarView(self.notebook, self.controller),
            "kmeans": KmeansView(self.notebook, self.controller),
            "heatmap": HeatmapView(self.notebook, self.controller)
        }
        self.notebook.add(self.views["video"], text="リアルタイム映像")
        self.notebook.add(self.views["clustering"], text="階層クラスタリング") 
        self.notebook.add(self.views["spectrum"], text="パワースペクトル")
        self.notebook.add(self.views["radar"], text="レーダーチャート")
        self.notebook.add(self.views["kmeans"], text="k-means法")
        self.notebook.add(self.views["heatmap"], text="ヒートマップ")

    def on_app_closing(self):
        if self.controller:
            self.controller.on_app_closing()
        self.destroy()

    def toggle_focus_panel(self):
        if self.focus_panel.winfo_viewable():
            self.focus_panel.pack_forget()
            self.focus_toggle_button.config(text="▶ フォーカスパネルを表示")
        else:
            self.focus_panel.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(5,0), before=self.nametowidget('!frame2'))

    def set_focused_id(self, id_name):
        all_items = list(self.focus_id_listbox.get(0, tk.END))
        target_idx = -1
        for i, item in enumerate(all_items):
            if f"({id_name})" in item:
                target_idx = i
                break
        if target_idx == -1:
            print(f"WARN: 要求されたID '{id_name}' はリストボックスに存在しません。")
            return
        print(f"INFO: ID '{id_name}' にフォーカスします。")
        self.focus_id_listbox.selection_clear(0, tk.END)
        self.focus_id_listbox.selection_set(target_idx)
        self.focus_id_listbox.event_generate("<<ListboxSelect>>")
        self.notebook.select(self.views["spectrum"])

    def clear_all_graphs(self):
        if hasattr(self, 'ui_manager'):
            self.ui_manager.clear_all_views()