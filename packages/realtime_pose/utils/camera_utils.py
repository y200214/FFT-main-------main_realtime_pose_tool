# utils/camera_utils.py

import cv2
import logging
import platform
import subprocess

import numpy as np

def get_available_cameras(max_cameras_to_check=10):
    """
    利用可能なカメラデバイスのインデックスと名前のリストを取得します。
    OSに応じて最適な方法でカメラ名を取得します。

    Args:
        max_cameras_to_check (int): フォールバック時にチェックするカメラの最大数。

    Returns:
        list of tuple: (インデックス, 名前) のリスト。
    """
    os_type = platform.system()
    if os_type == "Windows":
        try:
            return _get_cameras_windows()
        except ImportError:
            logging.warning("pygrabberがインストールされていません。Windowsでカメラ名を取得するには 'pip install pygrabber' を実行してください。")
            return _get_cameras_fallback(max_cameras_to_check)
    elif os_type == "Linux":
        return _get_cameras_linux()
    else:
        logging.info(f"{os_type}では、フォールバックのカメラ検出方法を使用します。")
        return _get_cameras_fallback(max_cameras_to_check)

def _get_cameras_windows():
    """ Windows用のカメラ名取得処理 """
    from pygrabber.dshow_graph import FilterGraph
    
    graph = FilterGraph()
    devices = graph.get_input_devices()
    available_cameras = []
    for i, device_name in enumerate(devices):
        available_cameras.append((i, device_name))
    
    if not available_cameras:
        logging.warning("利用可能なカメラが見つかりませんでした。")
        
    return available_cameras

def _get_cameras_linux():
    """ Linux用のカメラ名取得処理 """
    try:
        # v4l2-ctlコマンドでデバイスをリストアップ
        result = subprocess.run(['v4l2-ctl', '--list-devices'], capture_output=True, text=True, check=True)
        output = result.stdout
        
        available_cameras = []
        devices = output.strip().split('\n\n')
        
        for device_block in devices:
            lines = device_block.strip().split('\n')
            device_name = lines[0].strip()
            device_path = lines[1].strip()
            
            # /dev/videoX からインデックスを抽出
            if '/dev/video' in device_path:
                try:
                    index = int(device_path.replace('/dev/video', ''))
                    # 念のためOpenCVで開けるか確認
                    cap = cv2.VideoCapture(index)
                    if cap.isOpened():
                        available_cameras.append((index, device_name))
                        cap.release()
                except (ValueError, IndexError):
                    continue
        
        # インデックスでソート
        available_cameras.sort(key=lambda x: x[0])
        return available_cameras
        
    except (subprocess.CalledProcessError, FileNotFoundError):
        logging.warning("v4l2-ctlコマンドの実行に失敗しました。フォールバックします。")
        return _get_cameras_fallback()

def _get_cameras_fallback(max_cameras_to_check=10):
    """
    OSに依存しないフォールバック用のカメラ検出処理。
    """
    available_cameras = []
    for i in range(max_cameras_to_check):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            device_name = f"Camera {i}"
            available_cameras.append((i, device_name))
            cap.release()
        else:
            break
    
    if not available_cameras:
        logging.warning("利用可能なカメラが見つかりませんでした。")
        
    return available_cameras

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    cameras = get_available_cameras()
    if cameras:
        logging.info("利用可能なカメラ:")
        for idx, name in cameras:
            logging.info(f"  - Index: {idx}, Name: {name}")
    else:
        logging.info("カメラが見つかりません。")

def get_camera_max_resolution(camera_index):
    """
    指定されたカメラの最大解像度を取得します。
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        return None, None
    
    # 非常に大きな解像度を設定しようと試みる
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 9999)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 9999)
    
    # カメラが実際に設定した最大値を取得
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    cap.release()
    logging.info(f"カメラ {camera_index} の最大解像度: {width}x{height}")
    return width, height

def create_fisheye_to_equirectangular_map(frame_shape):
    """
    魚眼映像を正距円筒図法に変換するためのマッピングを作成します。
    """
    H_in, W_in = frame_shape
    W_out = W_in 
    H_out = W_in // 2 # 正距円筒図法は通常、幅:高さ=2:1

    # 出力画像の各ピクセルの座標グリッドを作成
    j, i = np.meshgrid(np.arange(H_out), np.arange(W_out), indexing='ij')

    # 2D座標を3D球面座標に変換
    phi = (i / (W_out - 1) - 0.5) * 2 * np.pi
    theta = -(j / (H_out - 1) - 0.5) * np.pi

    # 3D球面座標を3D直交座標に変換
    x_3d = np.cos(theta) * np.sin(phi)
    y_3d = np.sin(theta)
    z_3d = np.cos(theta) * np.cos(phi)

    # 3D直交座標を魚眼レンズの2D座標に投影
    r = 2 * np.arctan2(np.sqrt(x_3d**2 + y_3d**2), z_3d) / np.pi
    
    u = r * x_3d / np.sqrt(x_3d**2 + y_3d**2)
    v = r * y_3d / np.sqrt(x_3d**2 + y_3d**2)

    # NaNを0で置き換える
    u = np.nan_to_num(u)
    v = np.nan_to_num(v)

    # 入力画像の中心を基準とした座標に変換
    map_x = (u * W_in / 2 + W_in / 2).astype(np.float32)
    map_y = (v * H_in / 2 + H_in / 2).astype(np.float32)

    return map_x, map_y

import numpy as np # ファイルの先頭にnumpyのインポートを追加するのを忘れないでください

def create_ptz_maps(frame_shape, center, fov, tilt, pan, zoom):
    """
    魚眼映像から特定の部分を切り出すためのリマップ用マップを作成します。
    (PTZ: Pan-Tilt-Zoom)
    """
    H_in, W_in = frame_shape
    W_out, H_out = W_in // 2, H_in // 2 # 出力サイズは入力の半分に

    # 出力画像の座標グリッド
    x_out, y_out = np.meshgrid(np.arange(W_out), np.arange(H_out))

    # 中心からの相対座標に変換
    x_norm = (x_out - W_out / 2) / (W_out / 2)
    y_norm = (y_out - H_out / 2) / (H_out / 2)

    # ズーム適用
    x_norm /= zoom
    y_norm /= zoom

    # 球面座標への変換
    r = np.sqrt(x_norm**2 + y_norm**2)
    theta = np.arctan2(y_norm, x_norm)
    
    f = W_in / fov # 簡易的な焦点距離
    phi = r / f

    # パン（水平回転）とチルト（垂直回転）の適用
    pan_rad = np.deg2rad(pan)
    tilt_rad = np.deg2rad(tilt)

    x_sphere = np.sin(phi) * np.cos(theta)
    y_sphere = np.sin(phi) * np.sin(theta)
    z_sphere = np.cos(phi)

    # チルト回転
    x_tilt = x_sphere
    y_tilt = y_sphere * np.cos(tilt_rad) - z_sphere * np.sin(tilt_rad)
    z_tilt = y_sphere * np.sin(tilt_rad) + z_sphere * np.cos(tilt_rad)

    # パン回転
    x_pan = x_tilt * np.cos(pan_rad) - z_tilt * np.sin(pan_rad)
    y_pan = y_tilt
    z_pan = x_tilt * np.sin(pan_rad) + z_tilt * np.cos(pan_rad)

    # 3D座標から魚眼画像の2D座標へ再投影
    theta_final = np.arctan2(y_pan, x_pan)
    phi_final = np.arccos(z_pan)
    r_final = phi_final * f

    map_x = (center[0] + r_final * np.cos(theta_final)).astype(np.float32)
    map_y = (center[1] + r_final * np.sin(theta_final)).astype(np.float32)

    return map_x, map_y