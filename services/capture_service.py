# ファイル名: services/capture_service.py

from multiprocessing import Queue, Event
import multiprocessing
import time
import logging
import queue
import traceback

from services.feature_extractor import FeatureExtractor
from services.process_utils import Status, StatusMessage
from services.video_source import VideoSource

logger = logging.getLogger(__name__)

class CaptureService:
    """
    映像取得と解析処理をバックグラウンドプロセスで実行するサービス。
    """
    def __init__(self, data_queue: Queue, frame_queue: Queue, status_queue: Queue, config: dict, analysis_active: Event):
        self.data_queue = data_queue
        self.frame_queue = frame_queue
        self.status_queue = status_queue
        self.config = config
        self.analysis_active = analysis_active
        self._process = None
        self.running = multiprocessing.Event()

    def start(self):
        if self._process and self._process.is_alive():
            return

        self.running.set()
        self._process = multiprocessing.Process(
            target=self._run_capture_loop,
            args=(self.data_queue, self.frame_queue, self.status_queue, self.running, self.config, self.analysis_active),
            daemon=True
        )
        self._process.start()
        logger.info("CaptureServiceを開始しました。")

    def stop(self):
        self.running.clear()
        if self._process:
            self._process.join(timeout=2)
            if self._process.is_alive():
                logger.warning("CaptureServiceが時間内に終了せず、強制終了します。")
                self._process.terminate()
            logger.info("CaptureServiceを停止しました。")
        self._process = None

    @staticmethod
    def _run_capture_loop(data_queue, frame_queue, status_queue, running_event, config, analysis_active_event):
        """【別プロセス】映像処理ループ"""
        logger.info("(別プロセス) 映像処理プロセスを開始します。")
        
        video_source = None
        feature_extractor = None
        try:
            video_source = VideoSource(config['video_source'])
            feature_extractor = FeatureExtractor(config)
        except Exception as e:
            logger.error(f"(別プロセス) 初期化に失敗: {e}")
            status_queue.put(StatusMessage(Status.ERROR, f"初期化に失敗しました:\n{e}"))
            return

        while running_event.is_set():
            try:
                ret, frame = video_source.get_frame()
                
                # --- ここからロジックを修正 ---
                if not ret:
                    logger.warning("(別プロセス) カメラからフレームを取得できませんでした。ループを終了します。")
                    status_queue.put(StatusMessage(Status.COMPLETED, "映像の再生が完了、またはカメラが停止しました。"))
                    break
                
                annotated_frame = frame
                
                # 「解析開始」が押されている場合のみ、重い解析処理を実行
                if analysis_active_event.is_set():
                    annotated_frame, features = feature_extractor.process_frame(frame)
                    if features:
                        packet = {'timestamp': time.time()}
                        for p in features:
                            packet[p['id']] = p['features']
                        # タイムアウトを短くして、UIが待たされすぎないようにする
                        data_queue.put(packet, timeout=0.5)

                # プレビュー映像をUIに送る (常に実行)
                # キューが満杯なら古いフレームを捨てて新しいものを入れる
                if not frame_queue.full():
                    frame_queue.put(annotated_frame, timeout=0.5)
                # --- 修正ここまで ---

            except queue.Full:
                logger.warning("(別プロセス) UI側の処理が追いついていないため、フレームまたはデータをスキップしました。")
                continue
            except Exception as e:
                logger.error(f"(別プロセス) フレーム処理中に予期せぬエラー: {e}\n{traceback.format_exc()}")
                status_queue.put(StatusMessage(Status.ERROR, f"フレーム処理中にエラーが発生しました:\n{e}"))
                time.sleep(1)
            
            time.sleep(0.001) # CPU負荷を少し下げる

        if video_source:
            video_source.release()
        if feature_extractor:
            feature_extractor.close()
        logger.info("(別プロセス) 映像処理プロセスが正常に終了しました。")