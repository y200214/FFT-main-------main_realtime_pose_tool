# services/pipelines/pose_to_fft.py として保存

from packages.realtime_pose.api import RealtimePose
from core.fft_api import analyze_joint_fft
import collections
import numpy as np

FFT_WINDOW_SIZE = 128 

def pose_to_fft_pipeline(video_source, target_joint_index, fps=30):
    """
    ポーズ推定からFFT解析までの一連の処理を実行するパイプライン
    """
    pose_estimator = RealtimePose()
    history = collections.defaultdict(lambda: collections.deque(maxlen=FFT_WINDOW_SIZE))

    # track_videoから「描画済みフレーム」と「キーポイント辞書」を受け取る
    for annotated_frame, keypoints_dict in pose_estimator.track_video(video_source):
        
        fft_results_this_frame = {} # このフレームで得られたFFT結果を格納

        for person_id, keypoints in keypoints_dict.items():
            # keypointsは正規化座標なので、そのまま使用
            joint_xy = keypoints[target_joint_index][:2]
            history[person_id].append(joint_xy)

            if len(history[person_id]) == FFT_WINDOW_SIZE:
                joint_timeseries = np.array(history[person_id])
                fft_result = analyze_joint_fft(joint_timeseries, fps)
                fft_results_this_frame[person_id] = fft_result
        
        # 描画済みフレームと、このフレームで計算されたFFT結果を返す
        yield annotated_frame, fft_results_this_frame