# ファイル名: constants.py

EMOTION_VARS = ['happy', 'surprise', 'neutral', 'sad', 'contempt', 'disgust', 'fear', 'anger']
BEHAVIOR_VARS = ['lips', 'left_eye', 'right_eye', 'head', 'left_shoulder', 'right_shoulder', 'left_hand', 'right_hand', 'roll', 'pitch', 'yaw']
ALL_VARIABLES = EMOTION_VARS + BEHAVIOR_VARS
REALTIME_ID_PREFIX = "ID_" # リアルタイム処理で人を追跡する際のIDの接頭辞
UPDATE_INTERVAL_MS = 1000  # 更新間隔 (ミリ秒) = 1秒
SLIDING_WINDOW_SECONDS = 30 # スライディングウィンドウの秒数