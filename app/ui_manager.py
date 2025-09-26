# ファイル名: app/ui_manager.py

import tkinter as tk
import pandas as pd
from tkinter import messagebox

class UIManager:
    def __init__(self, app_instance):
        self.app = app_instance
        self.controller = self.app.controller
        self.views = self.app.views
        self.analysis_params = self.controller.config_manager.config.analysis_parameters
        self.sliding_window = self.analysis_params.SLIDING_WINDOW_SECONDS
        self.setup_control_panel() # ボタンなどを配置するメソッドを呼び出す

    def show_info(self, title, message):
        messagebox.showinfo(title, message)

    def show_warning(self, title, message):
        messagebox.showwarning(title, message)

    def show_error(self, title, message):
        messagebox.showerror(title, message)

    def ask_yes_no(self, title, message):
        return messagebox.askyesno(title, message)

    def update_camera_dropdown(self, cameras):
        """開いている設定ダイアログのカメラドロップダウンを更新する"""
        if hasattr(self.app, 'config_dialog') and self.app.config_dialog:
            if self.app.config_dialog.winfo_exists():
                self.app.config_dialog.update_camera_dropdown(cameras)

    def update_focus_listbox(self, ids):
        self.app.focus_id_listbox.delete(0, tk.END)
        for i, id_name in enumerate(ids):
            display_text = f"{i + 1} ({id_name})"
            self.app.focus_id_listbox.insert(tk.END, display_text)
        self.app.focus_id_listbox.selection_set(0, tk.END)

    def update_active_view(self, model_data):
        if not model_data.full_history:
            return

        active_view_key = None
        try:
            selected_tab_id = self.app.notebook.select()
            if selected_tab_id:
                active_widget = self.app.notebook.nametowidget(selected_tab_id)
                for key, view_widget in self.views.items():
                    if view_widget == active_widget:
                        active_view_key = key
                        break
        except tk.TclError:
            return

        if not active_view_key:
            return
            
        df_full_filtered, df_sliding_filtered, ps_filtered = self._get_filtered_data(model_data)

        current_timestamp = model_data.full_history[-1].get('timestamp', 0)
        full_duration = current_timestamp
        sliding_duration = self.sliding_window

        if 'clustering' in active_view_key.lower():
            self.views["clustering"].update_plot(df_full_filtered, df_sliding_filtered, full_duration, sliding_duration)
        elif 'spectrum' in active_view_key.lower():
            self.views["spectrum"].update_plot(ps_filtered)
        elif 'radar' in active_view_key.lower():
            radar_dfs = {'sliding': df_sliding_filtered, 'full': df_full_filtered}
            self.views["radar"].update_plot(radar_dfs)
        elif 'kmeans' in active_view_key.lower():
            self.views["kmeans"].update_plot(df_full_filtered, df_sliding_filtered, full_duration, sliding_duration)
        elif 'heatmap' in active_view_key.lower():
            self.views["heatmap"].update_plot(df_full_filtered, df_sliding_filtered, full_duration, sliding_duration)

    def _get_filtered_data(self, model_data):
        df_full = model_data.last_slope_dfs.get('full', pd.DataFrame())
        df_sliding = model_data.last_slope_dfs.get('sliding', pd.DataFrame())
        ps_data = model_data.last_power_spectrums

        focused_ids = self.controller.focused_ids
        if focused_ids:
            df_full_filtered = df_full[df_full.index.isin(focused_ids)]
            df_sliding_filtered = df_sliding[df_sliding.index.isin(focused_ids)]
            ps_sliding_filtered = {id_name: data for id_name, data in ps_data.get('sliding', {}).items() if id_name in focused_ids}
            ps_full_filtered = {id_name: data for id_name, data in ps_data.get('full', {}).items() if id_name in focused_ids}
            ps_filtered = {'sliding': ps_sliding_filtered, 'full': ps_full_filtered}
        else:
            df_full_filtered, df_sliding_filtered = df_full, df_sliding
            ps_filtered = ps_data

        return df_full_filtered, df_sliding_filtered, ps_filtered

    def update_slider_and_time(self, model_data, history_index):
        """スライダーと時間表示を更新する"""
        if not model_data.full_history:
            return

        current_max_index = len(model_data.full_history) - 1
        if current_max_index > 0:
            self.app.slider.config(to=current_max_index)
        
        # 渡された現在の再生位置(history_index)にスライダーをセットする
        self.app.slider.set(history_index)
        
        try:
            playback_time = model_data.full_history[history_index]['timestamp']
            self.app.elapsed_time_var.set(f"経過時間: {playback_time:.1f}s")
            
            total_time = model_data.full_history[-1]['timestamp']
            self.app.time_input_var.set(f"{playback_time:.1f}")
            self.app.total_time_var.set(f"s / {total_time:.1f}s")
        except (IndexError, KeyError):
            pass # データが準備できていない場合のエラーを無視

    def update_control_buttons_state(self):
        handler = self.controller.current_mode_handler
        is_running = handler.is_running
        is_paused = handler.is_paused

        if not is_running:
            self.app.start_button.config(state="normal")
            self.app.stop_button.config(state="disabled")
            self.app.pause_button.config(state="disabled", text="一時停止")
        else:
            self.app.start_button.config(state="disabled")
            self.app.stop_button.config(state="normal")
            self.app.pause_button.config(state="normal")
            if is_paused:
                self.app.pause_button.config(text="再開")
            else:
                self.app.pause_button.config(text="一時停止")

    def clear_all_views(self):
        empty_df = pd.DataFrame()
        empty_ps = {}
        self.views["clustering"].update_plot(empty_df, empty_df, 0, 0)
        self.views["kmeans"].update_plot(empty_df, empty_df, 0, 0)
        self.views["heatmap"].update_plot(empty_df, empty_df, 0, 0)
        self.views["radar"].update_plot({'sliding': empty_df, 'full': empty_df})
        self.views["spectrum"].update_plot(empty_ps)

    def clear_focus_listbox_selection(self):
        self.app.focus_id_listbox.selection_clear(0, tk.END)

    def set_rt_button_state(self, state):
        """リアルタイム復帰ボタンの状態を設定する"""
        self.app.rt_button.config(state=state)

    def set_pause_button_state(self, text, command):
        """一時停止ボタンのテキストとコマンドを更新する"""
        self.app.pause_button.config(text=text, command=command)

    def clear_time_inputs(self):
        """時間入力ボックスをクリアする"""
        self.app.time_input_var.set("")
        self.app.total_time_var.set("")

    def reset_ui_state(self):
        self.app.slider.set(0)
        self.app.slider.config(to=100)
        self.app.progress_var.set(0)
        self.app.elapsed_time_var.set("経過時間: 0.0s")
        self.app.time_input_var.set("")
        self.app.total_time_var.set("")
        self.app.focus_id_listbox.delete(0, tk.END)
        self.app.batch_button.config(state="disabled")

    def update_video_frame(self, frame):
        """
        Controllerから渡された映像フレームをVideoViewに表示する
        """
        if self.video_view:
            self.video_view.update_frame(frame)

    def set_controller(self, controller):
        self.controller = controller
        # control_panel内の「リアルタイム解析」ボタンにcontrollerのメソッドを割り当てる
        # ボタンの変数名(例: self.control_panel.realtime_button)は実際のコードに合わせてください
        self.control_panel.realtime_button.config(command=self.controller.start_realtime_analysis)

