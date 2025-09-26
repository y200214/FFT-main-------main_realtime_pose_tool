# ファイル名: app/controller.py

import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd


from services.pipelines.pose_to_fft import pose_to_fft_pipeline
from threading import Thread
from constants import ALL_VARIABLES
from core.config_manager import ConfigManager
from core.data_processor import DataProcessor
from core.model import AnalysisModel
from core.analysis_service import AnalysisService 
from core.save_manager import SaveManager
from app.views.config_dialog import ConfigDialog
from .mode_handler.csv_replay_handler import CsvReplayHandler
from .mode_handler.realtime_handler import RealtimeHandler
from services.process_utils import Status
from utils.camera_utils import CameraMonitor

class AppController:
    def __init__(self, app, status_queue):
        self.app = app
        self.status_queue = status_queue
        self.model = AnalysisModel()
        self.data_processor = DataProcessor()
        self.analysis_service = AnalysisService(self.model, self.data_processor)
        self.config_manager = ConfigManager()
        self.save_manager = SaveManager(self)
        self.ui_manager = None

        self.analysis_params = self.config_manager.config.analysis_parameters
        self.update_interval = self.analysis_params.UPDATE_INTERVAL_MS
        self.sliding_window = self.analysis_params.SLIDING_WINDOW_SECONDS

        self.mode_handlers = {
            "csv": CsvReplayHandler(self),
            "realtime": RealtimeHandler(self)
        }
        self.current_mode_handler = self.mode_handlers["csv"]

        self.after_id = None
        self.status_check_after_id = None
        self.preview_after_id = None        
        self.is_realtime_mode = False
        self.is_display_paused = False
        self.focused_ids = []

        self.is_saving_cancelled = False
        self.batch_analysis_complete = False
        self.batch_result_df = None
        self.save_plots_complete = False
        self.save_plots_error = None
        self.progress_dialog = None
        self.save_progress = 0
        self.save_total_steps = 0
        
        self.camera_monitor = CameraMonitor(self._on_camera_list_changed)

    def set_ui_manager(self, ui_manager):
        self.ui_manager = ui_manager

    def _on_camera_list_changed(self, cameras):
        if self.app and self.ui_manager:
            self.app.after(0, self.ui_manager.update_camera_dropdown, cameras)

    def on_app_closing(self):
        print("INFO: アプリケーションを終了します。")
        self.camera_monitor.stop()
        if self.current_mode_handler.is_running:
            self.stop_analysis()

    def _on_mode_change(self):
        self._stop_preview_loop()
        self.stop_update_loop()
        if self.current_mode_handler:
            self.current_mode_handler.on_mode_deselected()
        new_mode = self.app.mode.get()
        self.current_mode_handler = self.mode_handlers[new_mode]
        self.is_realtime_mode = isinstance(self.current_mode_handler, RealtimeHandler)
        self.current_mode_handler.on_mode_selected()

    def load_csvs(self):
        filepaths = filedialog.askopenfilenames(
            title="再生するCSVファイルを選択（複数選択可）",
            filetypes=[("CSV files", "*.csv")]
        )
        if not filepaths: return
        success, ids = self.model.load_csv_data(filepaths)
        if not success:
            self.ui_manager.show_error("エラー", "ファイルの読み込みまたはIDの推定に失敗しました。")
        else:
            self.ui_manager.update_focus_listbox(ids)
            self.focus_on_all_ids()
        self.current_mode_handler.on_mode_selected()

    def start_analysis(self):
        self._stop_preview_loop()
        self.current_mode_handler.start()
        self.ui_manager.update_control_buttons_state()
        if self.is_realtime_mode:
            self._check_status_queue()

    def stop_analysis(self):
        if self.status_check_after_id:
            self.app.after_cancel(self.status_check_after_id)
            self.status_check_after_id = None
        self.current_mode_handler.stop()
        self.ui_manager.update_control_buttons_state()
        if self.is_realtime_mode:
            self._start_preview_loop()

    def toggle_pause(self):
        self.current_mode_handler.toggle_pause()
        self.ui_manager.update_control_buttons_state()

    def start_update_loop(self):
        self.stop_update_loop()
        self.process_data_and_update_views()

    def stop_update_loop(self):
        if self.after_id:
            self.app.after_cancel(self.after_id)
            self.after_id = None

    def _start_preview_loop(self):
        self._stop_preview_loop()
        def loop():
            if self.app.mode.get() == "realtime" and not self.current_mode_handler.is_running:
                latest_frame = self.current_mode_handler.get_latest_frame()
                if latest_frame is not None:
                    self.app.views["video"].update_frame(latest_frame)
                self.preview_after_id = self.app.after(33, loop)
        loop()

    def _stop_preview_loop(self):
        if self.preview_after_id:
            self.app.after_cancel(self.preview_after_id)
            self.preview_after_id = None

    def save_plots(self):
        self.is_saving_cancelled = False
        self.save_plots_complete = False
        self.save_plots_error = None
        self.save_progress = 0
        save_index = int(self.app.slider.get())
        if not self.model.full_history or save_index >= len(self.model.full_history):
            self.ui_manager.show_error("保存エラー", "保存できる有効なデータがありません。")
            return
        timestamp_to_save = self.model.full_history[save_index]['timestamp']
        self.save_manager.save_all_plots(timestamp_to_save)

    def save_features_to_csv(self):
        print("INFO: 特徴量のCSV保存処理を開始します。")
        df_to_save = self.model.last_slope_dfs.get('full')
        if df_to_save is None or df_to_save.empty:
            self.ui_manager.show_warning("保存エラー", "保存できる特徴量のデータがありません。\n先に「一括解析」などを実行してください。")
            return
        try:
            filepath = filedialog.asksaveasfilename(
                title="特徴量ファイルを保存",
                defaultextension=".csv",
                filetypes=[("CSVファイル", "*.csv"), ("すべてのファイル", "*.*")],
                initialfile="features.csv"
            )
            if filepath:
                df_to_save.to_csv(filepath, encoding='utf-8-sig')
                self.ui_manager.show_info("成功", f"特徴量ファイルが正常に保存されました。\n場所: {filepath}")
                print(f"INFO: 特徴量ファイルを保存しました: {filepath}")
            else:
                print("INFO: 特徴量の保存がキャンセルされました。")
        except Exception as e:
            print(f"ERROR: 特徴量の保存中にエラーが発生しました: {e}")
            self.ui_manager.show_error("保存エラー", f"ファイルの保存中にエラーが発生しました:\n{e}")

    def _on_slider_change(self, event):
        """スライダーが操作されたときの処理"""
        self.stop_update_loop()
        # --- ここから修正 ---
        if self.is_realtime_mode:
            self.is_realtime_mode = False 
            self.ui_manager.set_rt_button_state('normal')
        # --- 修正ここまで ---
        
        timestamp_index = int(self.app.slider.get())
        self.process_data_and_update_views(history_index=timestamp_index)

    def _return_to_realtime(self):
        """リアルタイム表示に復帰する"""
        self.is_display_paused = False
        self.is_realtime_mode = True
        
        self.ui_manager.set_rt_button_state('disabled')
        self.ui_manager.set_pause_button_state("一時停止", self.toggle_pause)
        self.ui_manager.clear_time_inputs()
        
        self.start_update_loop()

    def reset_all_data(self):
        if self.current_mode_handler.is_running:
            self.stop_analysis()
        if not self.ui_manager.ask_yes_no("確認", "本当にすべてのデータをリセットしますか？\nこの操作は元に戻せません。"):
            return
        print("INFO: 全てのデータをリセットします。")
        self.model.full_history = []
        self.model.active_ids = []
        self.model.time_series_df = None
        self.model.csv_replay_data = None
        self.model.last_power_spectrums = {}
        self.model.last_slope_dfs = {}
        self.focused_ids = []
        self.batch_result_df = None
        self.ui_manager.reset_ui_state()
        self.ui_manager.clear_all_views()
        self.ui_manager.show_info("完了", "すべてのデータをリセットしました。")

    def on_time_input_enter(self, event):
        if not self.model.full_history: return
        try:
            target_time = float(self.app.time_input_var.get())
        except (ValueError, TypeError):
            self.ui_manager.show_warning("入力エラー", "数値を入力してください。")
            return
        timestamps = [dp['timestamp'] for dp in self.model.full_history]
        if not (timestamps[0] <= target_time <= timestamps[-1]):
            self.ui_manager.show_warning("入力エラー", f"時間は {timestamps[0]:.1f} から {timestamps[-1]:.1f} の間で入力してください。")
            return
        time_diffs = [abs(ts - target_time) for ts in timestamps]
        closest_index = time_diffs.index(min(time_diffs))
        self.app.slider.set(closest_index)
        self._on_slider_change(None)

    def _run_batch_analysis(self):
        if self.model.csv_replay_data is None or self.model.csv_replay_data.empty:
            self.ui_manager.show_warning("警告", "先にCSVファイルを読み込んでください。")
            return
        if self.current_mode_handler.is_running:
            self.stop_analysis()
        self.app.batch_button.config(state="disabled")
        self.app.progress_bar.pack(fill=tk.X, expand=True, before=self.app.slider)
        self.app.progress_var.set(0)
        print("INFO: 一括解析のバックグラウンド処理を開始します。")
        self.batch_analysis_complete = False
        analysis_thread = threading.Thread(target=self._perform_batch_analysis_thread, daemon=True)
        analysis_thread.start()
        self._check_batch_analysis_status()

    def _perform_batch_analysis_thread(self):
        # (このメソッドは変更なし)
        try:
            print("INFO: (別スレッド) 一括解析の計算処理を開始します。")
            all_data_history = []
            for timestamp, row in self.model.csv_replay_data.iterrows():
                packet = {'timestamp': timestamp}
                for id_name in self.model.active_ids:
                    id_data = {}
                    for var in ALL_VARIABLES:
                        col_name = f"{id_name}_{var}"
                        if col_name in row and not pd.isna(row[col_name]):
                            id_data[var] = row[col_name]
                    if id_data:
                        packet[id_name] = id_data
                all_data_history.append(packet)
            df_full_features = self.analysis_service.perform_batch_analysis(all_data_history)
            self.batch_result_df = df_full_features
        except Exception as e:
            print(f"ERROR: (別スレッド) 一括解析の計算中にエラーが発生しました: {e}")
            self.batch_result_df = e
        finally:
            self.batch_analysis_complete = True
            print("INFO: (別スレッド) 計算処理が完了しました。")


    def _check_batch_analysis_status(self):
        if self.batch_analysis_complete:
            self.app.progress_bar.pack_forget()
            self.app.batch_button.config(state="normal")
            if isinstance(self.batch_result_df, Exception):
                self.ui_manager.show_error("エラー", f"一括解析中にエラーが発生しました:\n{self.batch_result_df}")
                return
            if self.batch_result_df is None or self.batch_result_df.empty:
                self.ui_manager.show_warning("警告", "特徴量の計算結果が空でした。")
                return
            print("INFO: 計算完了を検知。UIを更新します。")
            df_full = self.batch_result_df
            duration = self.model.csv_replay_data.index.max() if self.model.csv_replay_data is not None else 0
            self.ui_manager.views["clustering"].update_plot(df_full, pd.DataFrame(), duration, 0)
            self.ui_manager.views["radar"].update_plot()
            self.ui_manager.views["spectrum"].update_plot()
            self.ui_manager.views["kmeans"].update_plot(df_full, pd.DataFrame(), duration, 0)
            self.ui_manager.views["heatmap"].update_plot(df_full, pd.DataFrame(), duration, 0)
            if self.model.full_history:
                last_index = len(self.model.full_history) - 1
                self.app.slider.config(to=last_index)
                self.app.slider.set(last_index)
                last_timestamp = self.model.full_history[-1]['timestamp']
                self.app.elapsed_time_var.set(f"経過時間: {last_timestamp:.1f}s")
            self.app.update_idletasks()
            if messagebox.askyesno("完了", "一括解析が完了しました。\n結果をファイルに保存しますか？"):
                self.save_plots()
        else:
            self.after_id = self.app.after(100, self._check_batch_analysis_status)

    def _check_status_queue(self):
        try:
            while not self.status_queue.empty():
                msg = self.status_queue.get_nowait()
                if msg.status == Status.ERROR:
                    self.ui_manager.show_error("リアルタイムエラー", msg.message)
                    self.stop_analysis()
                elif msg.status == Status.WARNING:
                    self.ui_manager.show_warning("警告", msg.message)
                elif msg.status == Status.INFO:
                    self.ui_manager.show_info("情報", msg.message)
                elif msg.status == Status.COMPLETED:
                    self.ui_manager.show_info("完了", msg.message)
                    self.stop_analysis()
        except queue.Empty:
            pass
        finally:
            if self.current_mode_handler.is_running:
                self.status_check_after_id = self.app.after(200, self._check_status_queue)

    def open_settings_dialog(self):
        dialog = ConfigDialog(self.app, self.config_manager, self)
        self.app.wait_window(dialog)
        self.config_manager.load_config()
        self.analysis_params = self.config_manager.config.analysis_parameters
        self.update_interval = self.analysis_params.UPDATE_INTERVAL_MS
        self.sliding_window = self.analysis_params.SLIDING_WINDOW_SECONDS
        if self.app.mode.get() == "realtime":
            self._on_mode_change()
        print("INFO: 設定ダイアログが閉じられ、設定がリロードされました。")

    def update_preview_source(self, new_source):
        if self.app.mode.get() != "realtime":
            return
        print(f"INFO: プレビューソースを {new_source} に変更します。")
        self.config_manager.config.realtime_settings.video_source = new_source
        self.current_mode_handler.on_mode_deselected()
        self.current_mode_handler.on_mode_selected()

    def process_data_and_update_views(self, history_index=None):
        try:
            is_running = (history_index is None and self.current_mode_handler.is_running)
            if is_running:
                new_data = self.current_mode_handler.get_next_data_packet()
                if new_data:
                    self.model.full_history.append(new_data)
                else:
                    if self.is_realtime_mode:
                        pass # リアルタイムモードでは再生完了メッセージは不要
                    else:
                        self.stop_analysis()
                        self.ui_manager.show_info("完了", "再生が完了しました。")
                    return

            if not self.model.full_history:
                return

            target_index = history_index if history_index is not None else len(self.model.full_history) - 1
            full_slice_data = self.model.full_history[:target_index + 1]
            sliding_slice_data = self.model.full_history[max(0, target_index - self.sliding_window + 1) : target_index + 1]
            self.analysis_service.process_and_store_features(full_slice=full_slice_data, sliding_slice=sliding_slice_data)
            
            if not self.is_display_paused:
                self.ui_manager.update_active_view(self.model)
                self.ui_manager.update_slider_and_time(self.model, target_index)

        except (queue.Empty, IndexError):
            pass

        if self.current_mode_handler.is_running and history_index is None and not self.is_display_paused:
            self.after_id = self.app.after(self.update_interval, self.process_data_and_update_views)

    def on_focus_id_change(self, event):
        selected_indices = self.app.focus_id_listbox.curselection()
        raw_ids = [self.app.focus_id_listbox.get(i) for i in selected_indices]
        parsed_ids = []
        for text in raw_ids:
            try:
                parsed_ids.append(text.split('(')[1][:-1])
            except IndexError:
                print(f"WARN: ID名のパースに失敗しました: {text}")
        self.focused_ids = parsed_ids
        print(f"INFO: フォーカス対象を {self.focused_ids} に変更しました。")
        self._refresh_views()

    def focus_on_all_ids(self):
        self.focused_ids = []
        self.ui_manager.clear_focus_listbox_selection()
        print("INFO: フォーカス対象を全員に変更しました。")
        self._set_all_spectrum_vars(True)
        self._refresh_views()
            
    def _set_all_spectrum_vars(self, state=True):
        if "spectrum" in self.app.views:
            self.app.views["spectrum"].set_all_variable_checkboxes(state)

    def _trigger_view_update(self):
        self._refresh_views()

    def _refresh_views(self):
        if self.model.full_history:
            current_index = int(self.app.slider.get())
            self.process_data_and_update_views(history_index=current_index)

    def start_realtime_analysis(self):
        """ リアルタイム解析を開始する (UIのボタンから呼び出される) """
        
        # 映像ソースをUIから取得 (例: カメラ0番)
        video_source = 0 
        
        def run_pipeline():
            # パイプラインを実行し、結果をループで受け取る
            for frame, fft_results in pose_to_fft_pipeline(video_source, target_joint_index=5):
                # 結果をUIに渡して画面を更新
                self.ui_manager.update_video_frame(frame)
                if fft_results:
                    self.ui_manager.update_fft_graph(fft_results)

        # UIが固まらないように別スレッドで実行
        thread = Thread(target=run_pipeline, daemon=True)
        thread.start()