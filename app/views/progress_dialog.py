import tkinter as tk
from tkinter import ttk

class ProgressDialog(tk.Toplevel):
    def __init__(self, parent, title="処理中...", cancel_callback=None):
        super().__init__(parent)
        self.title(title)
        
        self.geometry("400x150") # 少し縦幅を広げる
        self.resizable(False, False)
        self.protocol("WM_DELETE_WINDOW", lambda: None)
        
        self.transient(parent)
        self.grab_set()
        
        self.label = ttk.Label(self, text="処理を開始します...", padding=(20, 10))
        self.label.pack(pady=5)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self, variable=self.progress_var, length=360)
        self.progress_bar.pack(pady=10)
        
        # --- ここから追加 ---
        # cancel_callbackが指定されている場合のみボタンを表示する
        if cancel_callback:
            self.cancel_button = ttk.Button(self, text="キャンセル", command=cancel_callback)
            self.cancel_button.pack(pady=5)
        # --- 追加ここまで ---

        self.update_idletasks()

    def update_progress(self, value, text):
        """進捗バーの値とメッセージを更新する"""
        self.progress_var.set(value)
        self.label.config(text=text)
        self.update_idletasks() # 画面を即座に更新

    def close(self):
        """ウィンドウを閉じる"""
        self.grab_release()
        self.destroy()