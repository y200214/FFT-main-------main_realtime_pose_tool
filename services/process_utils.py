# ファイル名: services/process_utils.py

from enum import Enum, auto

class Status(Enum):
    """プロセス間の状態通知用Enum"""
    SUCCESS = auto()
    ERROR = auto()
    INFO = auto()
    WARNING = auto()
    COMPLETED = auto()

class StatusMessage:
    """プロセス間通信で送受信するメッセージクラス"""
    def __init__(self, status, message, data=None):
        self.status = status
        self.message = message
        self.data = data

    def __repr__(self):
        return f"StatusMessage(status={self.status}, message='{self.message}')"