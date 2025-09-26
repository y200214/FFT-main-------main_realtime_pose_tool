# utils/logger_config.py

import logging
import os
from logging.handlers import RotatingFileHandler

def setup_logging():
    """
    アプリケーション全体のロギングを設定する。
    """
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    log_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # --- ファイルハンドラ: ログをファイルに保存 ---
    log_file = os.path.join(log_dir, "app.log")
    # 5MBごとにファイルをローテーションし、バックアップは3つまで保持
    file_handler = RotatingFileHandler(
        log_file, maxBytes=5*1024*1024, backupCount=3, encoding='utf-8'
    )
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.INFO) # INFOレベル以上のログを記録

    # --- ストリームハンドラ: ログをコンソールに出力 ---
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel(logging.DEBUG) # DEBUGレベル以上のログを表示

    # --- ルートロガーにハンドラを追加 ---
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG) # アプリ全体の最低ログレベル
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    logging.info("ロギング設定が完了しました。")