# ファイル名: app/views/components/focus_panel.py (新規作成)

import tkinter as tk
from tkinter import ttk

class FocusPanel(ttk.Frame):
    """フォーカスID選択パネルのUIコンポーネント"""
    def __init__(self, parent, controller, app):
        super().__init__(parent)
        self.controller = controller
        self.app = app

        # このフレーム自体を focus_panel として app に登録
        self.app.focus_panel = self

        # 表示/非表示を切り替えるボタン
        self.app.focus_toggle_button = ttk.Button(self, text="◀ フォーカスパネルを隠す", command=self.app.toggle_focus_panel)
        self.app.focus_toggle_button.pack(side=tk.RIGHT, anchor='n', padx=5, pady=5)

        # フォーカス選択パネル
        focus_panel_frame = ttk.LabelFrame(self, text="フォーカス対象ID（複数選択可）")
        focus_panel_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

        ttk.Button(focus_panel_frame, text="全員を選択", command=self.controller.focus_on_all_ids).pack(side=tk.RIGHT, padx=5, pady=5)

        list_frame = ttk.Frame(focus_panel_frame)
        list_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)
        
        # Listboxをappインスタンスの属性として作成
        self.app.focus_id_listbox = tk.Listbox(list_frame, selectmode=tk.MULTIPLE, height=4)
        self.app.focus_id_listbox.bind("<<ListboxSelect>>", self.controller.on_focus_id_change)
        
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.app.focus_id_listbox.yview)
        self.app.focus_id_listbox.config(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.app.focus_id_listbox.pack(side=tk.LEFT, fill=tk.X, expand=True)