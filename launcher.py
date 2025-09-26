# ファイル名: launcher.py

import multiprocessing
import matplotlib.pyplot as plt 
import sys
import os

# --- ここから追加 ---
# プロジェクトのルートディレクトリをPythonの検索パスに追加
# これにより、別プロセスからでもservicesなどのモジュールを正しく見つけられるようになる
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- 追加ここまで ---

from app.app_main import AppMainWindow
import logging
from utils.logger_config import setup_logging

# TensorFlowのINFOログを抑制
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

# 日本語フォント設定
import platform
system_name = platform.system()
if system_name == "Windows":
    plt.rcParams['font.family'] = 'Meiryo'
elif system_name == "Darwin": # Mac
    plt.rcParams['font.family'] = 'Hirino Sans'
else: # Linux
    plt.rcParams['font.family'] = 'IPAexGothic'
plt.rcParams['axes.unicode_minus'] = False


if __name__ == '__main__':
    multiprocessing.freeze_support()
    # アプリケーション起動の最初にロギングを設定
    setup_logging()
    app = AppMainWindow()
    app.mainloop()