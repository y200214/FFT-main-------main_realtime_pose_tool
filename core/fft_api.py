import numpy as np
from scipy.fft import rfft, rfftfreq

def analyze_joint_fft(joint_timeseries: np.ndarray, fps: int):
    """
    単一の関節の時系列データからFFTピークを計算する
    
    Args:
        joint_timeseries (np.ndarray): (フレーム数, 座標) の形状を持つデータ
        fps (int): 動画のフレームレート

    Returns:
        dict: ピーク周波数や振幅など
    """
    if joint_timeseries.ndim != 2 or joint_timeseries.shape[1] < 2:
        raise ValueError("joint_timeseries must be a 2D array with at least 2 columns (x, y)")

    # 座標の平均値を基準とした動きの大きさを計算
    magnitudes = np.linalg.norm(joint_timeseries - np.mean(joint_timeseries, axis=0), axis=1)
    
    # FFT実行
    yf = rfft(magnitudes)
    xf = rfftfreq(len(magnitudes), 1 / fps)
    
    # ピークを検出（0Hzの直流成分は除く）
    if len(xf) > 1:
        peak_idx = np.argmax(np.abs(yf[1:])) + 1
        peak_frequency = xf[peak_idx]
        peak_amplitude = np.abs(yf[peak_idx])
    else:
        peak_frequency = 0
        peak_amplitude = 0
    
    return {"freq": peak_frequency, "amp": peak_amplitude}