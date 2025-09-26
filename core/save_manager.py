# ファイル名: core/save_manager.py

from tkinter import filedialog
import threading
import os
from datetime import datetime
import pandas as pd
import matplotlib

from app.views.progress_dialog import ProgressDialog
from app.views.save_selection_dialog import SaveSelectionDialog
from constants import ALL_VARIABLES

class SaveManager:
    def __init__(self, controller):
        """
        ファイル保存に関する全ての処理を担当するクラス。
        """
        self.controller = controller
        self.app = controller.app
        self.model = controller.model

    def save_all_plots(self, save_timestamp):
        """【司令塔】指定されたタイムスタンプのスナップショットを保存する"""
        save_index = int(self.app.slider.get())
        if save_index >= len(self.model.full_history):
            self.app.ui_manager.show_error("保存エラー", "無効なデータ点を指しています。")
            return

        # --- 1. 保存対象となるデータのスナップショットを作成 ---
        history_slice_to_save = self.model.full_history[:save_index + 1]
        sliding_slice_to_save = self.model.full_history[max(0, save_index - self.controller.sliding_window + 1): save_index + 1]
        save_timestamp = history_slice_to_save[-1]['timestamp']
        
        data_processor = self.controller.data_processor
        active_ids = self.model.active_ids

        df_full_history = data_processor.convert_history_to_df(history_slice_to_save, active_ids)
        df_full, ps_full = data_processor.get_features_from_df(df_full_history, active_ids)

        df_sliding_history = data_processor.convert_history_to_df(sliding_slice_to_save, active_ids)
        df_sliding, ps_sliding = data_processor.get_features_from_df(df_sliding_history, active_ids)
        
        all_data_to_save = {
            'slope_dfs': {'full': df_full, 'sliding': df_sliding},
            'power_spectrums': {'full': ps_full, 'sliding': ps_sliding}
        }

        # --- 2. ユーザーに保存項目を選択させる ---
        selection_dialog = SaveSelectionDialog(self.app)
        self.app.wait_window(selection_dialog)
        save_selection = selection_dialog.result

        if not save_selection:
            print("INFO: 保存がキャンセルされました。")
            return

        # --- 3. 保存先フォルダを選択させる ---
        base_path = filedialog.askdirectory(title="保存先の親フォルダを選択してください")
        if not base_path:
            print("INFO: グラフの保存がキャンセルされました。")
            return

        timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_folder = os.path.join(base_path, f"解析結果_{timestamp_str}")
        os.makedirs(output_folder, exist_ok=True)
        print(f"INFO: 選択された項目を '{output_folder}' に保存します。")

        # --- 4. プログレスバーの準備 ---
        num_ids = len(self.model.active_ids)
        num_spectrum_vars = len(ALL_VARIABLES)
        total_steps = 0
        if save_selection.get("features_csv"): total_steps += 1
        if save_selection.get("slopes_csv"): total_steps += 1
        if save_selection.get("clustering"): total_steps += 1
        if save_selection.get("kmeans"): total_steps += 1
        if save_selection.get("heatmap"): total_steps += 1
        if save_selection.get("spectrum"): total_steps += (num_ids * num_spectrum_vars)
        if save_selection.get("radar"): total_steps += num_ids
        
        self.progress_dialog = ProgressDialog(self.app, title="ファイル保存中", cancel_callback=self._cancel_save)
        self.progress_dialog.progress_bar.config(maximum=total_steps)
        self.controller.save_total_steps = total_steps

        # --- 5. 別スレッドで保存処理を開始 ---
        save_thread = threading.Thread(
            target=self._perform_save_thread,
            args=(output_folder, save_timestamp, all_data_to_save, save_selection),
            daemon=True
        )
        save_thread.start()
        self._check_save_status()

    def _perform_save_thread(self, output_folder, timestamp, all_data, save_selection):
        """【作業員】受け取ったスナップショットデータを元にファイル保存を実行する"""
        matplotlib.use('Agg')
        
        def progress_callback():
            self.controller.save_progress += 1
        
        try:
            cancel_check = lambda: self.controller.is_saving_cancelled
            
            # --- CSVファイルの保存 ---
            if cancel_check(): return
            if save_selection.get("features_csv"):
                df_to_save = all_data.get('slope_dfs', {}).get('full')
                if df_to_save is not None and not df_to_save.empty:
                    filepath = os.path.join(output_folder, "features.csv")
                    df_to_save.to_csv(filepath, encoding='utf-8-sig')
                progress_callback()

            if cancel_check(): return
            if save_selection.get("slopes_csv"):
                results_list = []
                spectrum_data = all_data.get('power_spectrums', {}).get('full', {})
                for id_name, var_data in spectrum_data.items():
                    for var_name, spec_tuple in var_data.items():
                        _freq, _amp, slope, intercept = spec_tuple
                        if slope is not None and intercept is not None:
                            results_list.append({'ID': id_name, 'Variable': var_name, 'Slope': slope, 'Intercept': intercept})
                if results_list:
                    pd.DataFrame(results_list).to_csv(os.path.join(output_folder, "slopes_and_intercepts.csv"), index=False, encoding='utf-8-sig')
                progress_callback()

            # --- 各Viewのグラフ保存 ---
            views = self.app.views
            for view_name, view_instance in views.items():
                if cancel_check(): break
                
                # 選択されており、かつ保存機能を持つビューのみ実行
                if save_selection.get(view_name) and hasattr(view_instance, 'save_plot'):
                    if view_name == 'video': continue
                    
                    print(f"INFO: {view_name} のグラフを保存します。")
                    view_instance.save_plot(
                        output_folder,
                        all_data,
                        progress_callback,
                        timestamp,
                        cancel_check
                    )

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.controller.save_plots_error = e
        finally:
            self.controller.save_plots_complete = True

    def _check_save_status(self):
        """【監視員】保存処理の進捗と完了/キャンセルを監視し、UIに反映する"""
        if self.controller.save_plots_complete or self.controller.is_saving_cancelled:
            if self.progress_dialog:
                self.progress_dialog.close()

            if self.controller.is_saving_cancelled:
                self.app.ui_manager.show_info("キャンセル", "保存処理がキャンセルされました。")
            elif self.controller.save_plots_error:
                self.app.ui_manager.show_error("エラー", f"グラフの保存中にエラーが発生しました:\n{self.controller.save_plots_error}")
            else:
                self.app.ui_manager.show_info("成功", "グラフが正常に保存されました。")
        else:
            progress_text = f"処理中... ({self.controller.save_progress} / {self.controller.save_total_steps})"
            if self.progress_dialog:
                self.progress_dialog.update_progress(self.controller.save_progress, progress_text)
            self.app.after(100, self._check_save_status)

    def _cancel_save(self):
        """保存処理のキャンセルを要求する (ProgressDialogから呼ばれる)"""
        self.controller.is_saving_cancelled = True
        if self.progress_dialog:
            self.progress_dialog.label.config(text="キャンセルしています...")
            self.progress_dialog.cancel_button.config(state="disabled")