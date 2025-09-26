# ファイル名: app/views/save_selection_dialog.py (新規作成)

import tkinter as tk
from tkinter import ttk

class SaveSelectionDialog(tk.Toplevel):
    """保存する項目を選択するためのダイアログ"""
    def __init__(self, parent):
        super().__init__(parent)
        self.transient(parent)
        self.grab_set()
        self.title("保存項目の選択")

        self.result = None

        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- チェックボックスの作成 ---
        self.selection_vars = {
            "features_csv": tk.BooleanVar(value=True),
            "slopes_csv": tk.BooleanVar(value=True),
            "clustering": tk.BooleanVar(value=True),
            "spectrum": tk.BooleanVar(value=True),
            "radar": tk.BooleanVar(value=True),
            "kmeans": tk.BooleanVar(value=True),
            "heatmap": tk.BooleanVar(value=True),
        }

        labels = {
            "features_csv": "特徴量 CSV",
            "slopes_csv": "傾きと切片 CSV",
            "clustering": "階層クラスタリング",
            "spectrum": "パワースペクトル",
            "radar": "レーダーチャート",
            "kmeans": "k-means法",
            "heatmap": "ヒートマップ"
        }

        for key, var in self.selection_vars.items():
            ttk.Checkbutton(main_frame, text=labels.get(key, key), variable=var).pack(anchor='w', padx=10)

        # --- ボタンフレーム ---
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(20, 0))

        ttk.Button(button_frame, text="キャンセル", command=self._on_cancel).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="保存実行", command=self._on_ok).pack(side=tk.RIGHT)

    def _on_ok(self):
        self.result = {key: var.get() for key, var in self.selection_vars.items()}
        self.destroy()

    def _on_cancel(self):
        self.result = None
        self.destroy()