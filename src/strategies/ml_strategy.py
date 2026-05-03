import json
import os
import re
import tempfile

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

MODEL_IMPORT_ERRORS = {}

def _compact_exception(exc):
    for line in str(exc).splitlines():
        if line.strip():
            return line.strip()
    return repr(exc)

try:
    import xgboost as xgb
except Exception as exc:
    xgb = None
    MODEL_IMPORT_ERRORS['XGBoost'] = _compact_exception(exc)

try:
    import lightgbm as lgb
except Exception as exc:
    lgb = None
    MODEL_IMPORT_ERRORS['LightGBM'] = _compact_exception(exc)

try:
    import mlflow
    import mlflow.sklearn
    from mlflow.data import from_pandas as mlflow_from_pandas
except ImportError:
    mlflow = None
    mlflow_from_pandas = None

class EnsembleMLStrategy:
    @staticmethod
    def get_supported_model_names():
        names = ['Random Forest', 'Gradient Boosting']
        if xgb is not None:
            names.append('XGBoost')
        if lgb is not None:
            names.append('LightGBM')
        return names

    @staticmethod
    def get_model_availability():
        all_models = ['Random Forest', 'Gradient Boosting', 'XGBoost', 'LightGBM']
        availability = {}
        for model_name in all_models:
            if model_name in ('Random Forest', 'Gradient Boosting'):
                availability[model_name] = {'available': True, 'reason': ''}
                continue
            if model_name == 'XGBoost':
                is_available = xgb is not None
            else:
                is_available = lgb is not None
            availability[model_name] = {
                'available': is_available,
                'reason': '' if is_available else MODEL_IMPORT_ERRORS.get(model_name, 'Import failed'),
            }
        return availability

    def __init__(
        self,
        df,
        features,
        target='y_return',
        train_window_quarters=12,
        enable_mlflow=True,
        mlflow_experiment_name='QuantVN WalkForward',
        mlflow_tracking_uri=None,
        mlflow_run_name=None,
        extra_run_params=None,
        dataset_source_path=None,
        enable_model_registry=False,
        model_registry_name='QuantVN-WalkForward-BestModel',
    ):
        """
        - df: Dataframe đã qua bước DataProcess (đã làm sạch)
        - features: Danh sách cấc cột đầu vào X
        - target: Cột nhãn mục tiêu y
        - train_window_quarters: Cửa sổ lăn (Số lượng quý dùng để Train trước khi Test Quý kế tiếp)
        - enable_mlflow: Bật/tắt tracking bằng MLflow
        - mlflow_experiment_name: Tên experiment để so sánh các lần chạy
        - mlflow_tracking_uri: Nơi MLflow lưu run, mặc định là thư mục mlruns của project
        - mlflow_run_name: Tên run, nếu không truyền sẽ tự tạo theo train_window
        - extra_run_params: Tham số bổ sung từ dashboard/backtest để log vào MLflow
        """
        self.df = df.copy()
        self.features = features
        self.target = target
        self.train_window = train_window_quarters
        self.enable_mlflow = enable_mlflow
        self.mlflow_experiment_name = mlflow_experiment_name
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.mlflow_run_name = mlflow_run_name
        self.extra_run_params = extra_run_params or {}
        self.mlflow_run_id = None
        self.mlflow_artifact_uri = None
        self.dataset_source_path = dataset_source_path
        self.enable_model_registry = enable_model_registry
        self.model_registry_name = model_registry_name
        self.model_nested_run_ids = {}
        self.registered_model_version = None
        
        self.models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        if xgb:
            self.models['XGBoost'] = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42)
        elif 'XGBoost' in MODEL_IMPORT_ERRORS:
            print(f"Bỏ qua XGBoost vì không import được: {MODEL_IMPORT_ERRORS['XGBoost']}")

        if lgb:
            self.models['LightGBM'] = lgb.LGBMRegressor(
                n_estimators=100, 
                learning_rate=0.05, 
                random_state=42, 
                min_child_samples=2,  # Rất quan trọng khi Data nhỏ
                verbose=-1            # Tắt báo cáo Warning rác
            )
        elif 'LightGBM' in MODEL_IMPORT_ERRORS:
            print(f"Bỏ qua LightGBM vì không import được: {MODEL_IMPORT_ERRORS['LightGBM']}")

    def _safe_mlflow_name(self, value):
        cleaned = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(value)).strip('_')
        return cleaned.lower() or 'value'

    def _format_quarter(self, value):
        try:
            return str(pd.to_datetime(value).to_period('Q'))
        except Exception:
            return str(value)

    def _mlflow_available(self):
        if not self.enable_mlflow:
            return False
        if mlflow is None:
            print("MLflow chưa được cài đặt. Chạy `pip install mlflow` hoặc `pip install -r requirements.txt` để bật tracking.")
            return False
        return True

    def _log_dataset_to_mlflow(self):
        if not self._mlflow_available() or not self.mlflow_run_id:
            return
        if mlflow_from_pandas is None:
            return

        dataset_df = self.df.copy()
        if self.target in dataset_df.columns:
            dataset_df = dataset_df.dropna(subset=[self.target])
        digest_source = dataset_df[['ticker', 'Quarter_Time', self.target]].copy() if {'ticker', 'Quarter_Time', self.target}.issubset(dataset_df.columns) else dataset_df

        dataset = mlflow_from_pandas(
            digest_source,
            source=self.dataset_source_path or 'raw_fundamental_data.csv',
            name='quantvn_raw_dataset',
        )
        mlflow.log_input(dataset, context='training')
        mlflow.set_tag('dataset_rows', str(len(dataset_df)))
        if self.dataset_source_path:
            mlflow.set_tag('dataset_source_path', self.dataset_source_path)

    def _start_mlflow_run(self, df_clean, quarters, total_steps):
        if not self._mlflow_available():
            return False

        try:
            if self.mlflow_tracking_uri:
                mlflow.set_tracking_uri(self.mlflow_tracking_uri)

            mlflow.set_experiment(self.mlflow_experiment_name)
            run_name = self.mlflow_run_name or f'walk_forward_tw{self.train_window}_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}'
            nested = mlflow.active_run() is not None
            run = mlflow.start_run(run_name=run_name, nested=nested)
            self.mlflow_run_id = run.info.run_id
            self.mlflow_artifact_uri = run.info.artifact_uri

            params = {
                'target': self.target,
                'train_window_quarters': self.train_window,
                'test_window_quarters': 1,
                'n_features': len(self.features),
                'n_rows_after_clean': len(df_clean),
                'n_quarters': len(quarters),
                'total_walk_forward_steps': total_steps,
                'start_quarter': self._format_quarter(quarters[0]),
                'end_quarter': self._format_quarter(quarters[-1]),
                'models': ', '.join(self.models.keys()),
            }
            params.update(self.extra_run_params)
            mlflow.log_params(params)
            mlflow.set_tags({
                'project': 'QuantVN-WalkForward',
                'validation': 'walk_forward_rolling_window',
                'target': self.target,
            })
            self._log_dataset_to_mlflow()

            return True
        except Exception as exc:
            print(f"Không khởi tạo được MLflow run, tiếp tục huấn luyện không tracking: {exc}")
            if mlflow.active_run() and self.mlflow_run_id == mlflow.active_run().info.run_id:
                mlflow.end_run()
            self.mlflow_run_id = None
            self.mlflow_artifact_uri = None
            return False

    def _log_mlflow_results(self, leaderboard, performance_log, step_metrics, final_scaler, input_example):
        if not self._mlflow_available() or not self.mlflow_run_id:
            return

        best_model = next(iter(leaderboard))
        mlflow.set_tag('best_model', best_model)
        mlflow.log_metric('best_avg_mse', float(leaderboard[best_model]))

        for model_name, scores in performance_log.items():
            metric_prefix = self._safe_mlflow_name(model_name)
            mlflow.log_metrics({
                f'{metric_prefix}_avg_mse': float(np.mean(scores['mse'])),
                f'{metric_prefix}_avg_mae': float(np.mean(scores['mae'])),
                f'{metric_prefix}_avg_rmse': float(np.mean(scores['rmse'])),
            })

        with tempfile.TemporaryDirectory() as tmp_dir:
            leaderboard_path = os.path.join(tmp_dir, 'leaderboard.csv')
            predictions_path = os.path.join(tmp_dir, 'walk_forward_predictions.csv')
            step_metrics_path = os.path.join(tmp_dir, 'walk_forward_step_metrics.csv')
            features_path = os.path.join(tmp_dir, 'features.json')

            leaderboard_df = pd.DataFrame(
                [{'model': name, 'avg_mse': mse} for name, mse in leaderboard.items()]
            )
            leaderboard_df.to_csv(leaderboard_path, index=False)
            self.predictions_df.to_csv(predictions_path, index=False)
            pd.DataFrame(step_metrics).to_csv(step_metrics_path, index=False)
            with open(features_path, 'w', encoding='utf-8') as f:
                json.dump({'features': self.features, 'target': self.target}, f, ensure_ascii=False, indent=2)

            mlflow.log_artifacts(tmp_dir, artifact_path='walk_forward')

        for model_name, model in self.models.items():
            scores = performance_log[model_name]
            with mlflow.start_run(run_name=f'model_{self._safe_mlflow_name(model_name)}', nested=True):
                mlflow.set_tag('model_name', model_name)
                mlflow.log_params({
                    'model_name': model_name,
                    'train_window_quarters': self.train_window,
                    'target': self.target,
                })
                mlflow.log_params({f'model_param.{k}': v for k, v in model.get_params().items()})
                mlflow.log_metrics({
                    'avg_mse': float(np.mean(scores['mse'])),
                    'avg_mae': float(np.mean(scores['mae'])),
                    'avg_rmse': float(np.mean(scores['rmse'])),
                })

                inference_pipeline = Pipeline([
                    ('scaler', final_scaler),
                    ('model', model),
                ])
                mlflow.sklearn.log_model(
                    sk_model=inference_pipeline,
                    artifact_path='model',
                    input_example=input_example,
                )
                self.model_nested_run_ids[model_name] = mlflow.active_run().info.run_id

        if self.enable_model_registry:
            self._register_best_model(best_model)

    def _register_best_model(self, best_model_name):
        if not self._mlflow_available():
            return
        best_run_id = self.model_nested_run_ids.get(best_model_name)
        if not best_run_id:
            return
        model_uri = f"runs:/{best_run_id}/model"
        safe_name = re.sub(r'[^A-Za-z0-9_.-]+', '_', self.model_registry_name).strip('_')
        registered = mlflow.register_model(model_uri=model_uri, name=safe_name)
        self.registered_model_version = f"{registered.name} v{registered.version}"
        mlflow.set_tag('registered_model_name', registered.name)
        mlflow.set_tag('registered_model_version', str(registered.version))

    def log_portfolio_artifacts(self, weights_df=None, kpi_text=None):
        """
        Ghi thêm artifact của phần portfolio/backtest vào parent MLflow run đã tạo ở bước train.
        """
        if not self._mlflow_available() or not self.mlflow_run_id:
            return None

        client = mlflow.tracking.MlflowClient()
        if weights_df is not None:
            client.log_metric(self.mlflow_run_id, 'portfolio_rebalance_count', float(len(weights_df)))

        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_paths = []
            if weights_df is not None:
                weights_path = os.path.join(tmp_dir, 'weights_matrix.csv')
                weights_df.to_csv(weights_path)
                artifact_paths.append(weights_path)
            if kpi_text:
                kpi_path = os.path.join(tmp_dir, 'backtest_kpi_report.txt')
                with open(kpi_path, 'w', encoding='utf-8') as f:
                    f.write(kpi_text)
                artifact_paths.append(kpi_path)

            for artifact_path in artifact_paths:
                client.log_artifact(self.mlflow_run_id, artifact_path, artifact_path='portfolio')

        return self.mlflow_run_id

    def prepare_data(self):
        """
        Chuẩn bị dữ liệu: Xếp theo thời gian tiến dần, loại bỏ quý cuối bị khuyết nhãn y_return.
        """
        print("Đang tiền xử lý Dữ liệu Chuỗi Thời Gian (Time-Series)...")
        # Xóa dòng không có y_return (Quý gần nhất do không có giá trong tương lai)
        df_clean = self.df.dropna(subset=[self.target]).copy()
        
        # SẮP XẾP cực kỳ quan trọng: Luôn luôn đi từ Quá khứ -> Tương lai để chống rò rỉ dữ liệu (Data Leakage)
        df_clean = df_clean.sort_values(by='Quarter_Time').reset_index(drop=True)
        
        return df_clean

    def walk_forward_competition(self):
        """
        Thực thi Rolling Window (Cuốn chiều):
        Dùng [Train Window] quý ở quá khứ -> Test [1] Quý ở tương lai -> Di chuyển Window tới 1 quý -> Lặp lại.
        """
        df_clean = self.prepare_data()
        
        # Lấy danh sách các mốc thời gian quý (Đã được sort tăng dần)
        quarters = df_clean['Quarter_Time'].sort_values().unique()
        
        if len(quarters) <= self.train_window:
            raise ValueError(f"Dữ liệu chỉ có {len(quarters)} quý, không đủ để tạo cửa sổ Train ({self.train_window} quý). Lấy thêm dữ liệu hoặc giảm train_window!")
        
        print(f"\n Test MODEL (Walk-Forward Rolling Window: {self.train_window} Quý Train -> 1 Quý Test)")
        print("-" * 60)

        # Bắt đầu trượt (Rolling)
        total_steps = len(quarters) - self.train_window
        mlflow_started = self._start_mlflow_run(df_clean, quarters, total_steps)

        performance_log = {name: {'mse': [], 'mae': [], 'rmse': []} for name in self.models.keys()}
        step_metrics = []
        scaler = StandardScaler()
        
        # Biến dành riêng cho Vẽ Biểu Đồ
        self.timeline_quarters = []
        self.actual_history = []
        self.model_predictions_history = {name: [] for name in self.models.keys()}
        
        # Bảng Dataframe So sánh chi tiết từng mã
        self.detailed_predictions_list = []

        try:
            for step in range(total_steps):
                # 1. Cắt cửa sổ Window
                train_start = quarters[step]
                train_end = quarters[step + self.train_window - 1]
                test_target = quarters[step + self.train_window]

                # 2. Lọc dữ liệu Train (Trong cửa sổ) và Test (Quý tương lai)
                train_data = df_clean[(df_clean['Quarter_Time'] >= train_start) & (df_clean['Quarter_Time'] <= train_end)]
                test_data = df_clean[df_clean['Quarter_Time'] == test_target]

                X_train, y_train = train_data[self.features], train_data[self.target]
                X_test,  y_test  = test_data[self.features], test_data[self.target]

                # Khởi tạo bảng Lưu Dấu vết Chi tiết cho Quý Test
                step_df = test_data[['ticker', 'Quarter_Time', self.target]].copy()
                step_df.rename(columns={self.target: 'y_true'}, inplace=True)
                step_df['y_true'] = step_df['y_true'].round(4)

                # Chuẩn hoá (Scale) - Fit trên Train và Transform trên Test
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                # Khôi phục tên Cột (Feature names) cho Numpy Array để tắt Warning của LightGBM
                X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
                X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

                # Lưu lại Mốc Thời gian và Lợi suất thực tế (Trung bình) của Quý này
                quarter_label = str(pd.to_datetime(test_target).to_period('Q'))
                self.timeline_quarters.append(quarter_label)
                self.actual_history.append(np.mean(y_test))

                # 3. Huấn luyện Model
                print(f"BƯỚC {step+1}/{total_steps} | Huấn luyện Quý Test: {quarter_label} | Size: Train({len(X_train)}), Test({len(X_test)})")

                for name, model in self.models.items():
                    # Train Model
                    model.fit(X_train_scaled, y_train)

                    # Dự đoán Quý tương lai
                    predictions = model.predict(X_test_scaled)

                    # Lưu Dấu vết Chi tiết
                    step_df[f'pred_{name}'] = np.round(predictions, 4)

                    # Lưu Dự báo trung bình của Model để vẽ Chart sau này
                    self.model_predictions_history[name].append(np.mean(predictions))

                    # Tính lỗi (MSE càng thấp => Càng tốt)
                    mse_score = mean_squared_error(y_test, predictions)
                    mae_score = mean_absolute_error(y_test, predictions)
                    rmse_score = np.sqrt(mse_score)
                    performance_log[name]['mse'].append(mse_score)
                    performance_log[name]['mae'].append(mae_score)
                    performance_log[name]['rmse'].append(rmse_score)
                    step_metrics.append({
                        'step': step + 1,
                        'model': name,
                        'train_start_quarter': self._format_quarter(train_start),
                        'train_end_quarter': self._format_quarter(train_end),
                        'test_quarter': quarter_label,
                        'train_rows': len(X_train),
                        'test_rows': len(X_test),
                        'mse': mse_score,
                        'mae': mae_score,
                        'rmse': rmse_score,
                    })

                # Đóng gói Quý Test
                self.detailed_predictions_list.append(step_df)

            # Gộp toàn bộ lịch sử soi Cổ phiếu vào Dataframe
            self.predictions_df = pd.concat(self.detailed_predictions_list, ignore_index=True)

            # --- TỔNG KẾT KẾT QUẢ ---
            print("\n🏆 KẾT QUẢ KIỂM THỬ TỪNG MÔ HÌNH (Trung bình lỗi MSE trên tất cả các Quý Test)")
            print("-" * 60)

            leaderboard = {}
            for name, scores in performance_log.items():
                avg_mse = np.mean(scores['mse'])
                leaderboard[name] = avg_mse

            # Sắp xếp từ lỗi thấp nhất đến cao nhất
            sorted_leaderboard = dict(sorted(leaderboard.items(), key=lambda item: item[1]))

            for name, score in sorted_leaderboard.items():
                print(f"Mô hình: {name:<20} | Lỗi MSE trung bình: {score:.6f}")

            self.latest_scaler = scaler
            input_example = X_train.head(min(5, len(X_train)))
            if mlflow_started:
                try:
                    self._log_mlflow_results(sorted_leaderboard, performance_log, step_metrics, scaler, input_example)
                except Exception as exc:
                    print(f"MLflow logging gặp lỗi, nhưng kết quả huấn luyện vẫn hợp lệ: {exc}")
                    mlflow.set_tag('mlflow_logging_error', str(exc)[:500])

            return sorted_leaderboard, self.models
        except Exception:
            if mlflow_started:
                mlflow.set_tag('run_status', 'failed')
            raise
        finally:
            if mlflow_started and mlflow.active_run() and mlflow.active_run().info.run_id == self.mlflow_run_id:
                mlflow.end_run()

    def plot_model_comparison(self, leaderboard):
        """
        Vẽ đồ thị Bar Chart vinh danh Model Top 1 và Biểu đồ Line Chart độ bám sát Tương lai
        """ 
        # 1. Vẽ Biểu đồ so sánh Lỗi MSE
        plt.figure(figsize=(10, 5))
        
        # Chọn màu sắc khác nhau (Model tốt nhất - MSE nhỏ nhất sẽ nằm đầu tiên)
        model_names = list(leaderboard.keys())
        scores = list(leaderboard.values())
        
        sns.barplot(x=model_names, y=scores, palette='coolwarm')
        plt.title('Model Error Leaderboard (Lower is Better)', fontsize=14, fontweight='bold')
        plt.ylabel('Mean Squared Error (MSE)')
        
        for i, v in enumerate(scores):
            plt.text(i, v + (max(scores)*0.01), f"{v:.5f}", color='black', ha='center', va='bottom')
            
        plt.show()
        
        # 2. Vẽ Biểu đồ Dòng thời gian: So sánh Dự Báo vs Thực Tế
        plt.figure(figsize=(12, 6))
        
        # Đường lợi suất thực tế của Thị trường (Đen, in đậm)
        plt.plot(self.timeline_quarters, self.actual_history, 
                 label='ACTUAL (Avg Market Return)', color='black', linewidth=4, marker='o')
        
        # Các đường dự báo của từng Mô hình (Đứt nét)
        colors = ['red', 'blue', 'green', 'orange']
        for (name, preds), color in zip(self.model_predictions_history.items(), colors):
            plt.plot(self.timeline_quarters, preds, label=f'Predicted: {name}', linestyle='--', marker='o')
            
        plt.title('Time Sensitivity: Actual vs Predicted Return Across Walk-Forward Quarters', fontsize=14, fontweight='bold')
        plt.xlabel('Test Quarters', fontsize=12)
        plt.ylabel('Average Log Return (y_return)', fontsize=12)
        plt.axhline(0, color='gray', linestyle='-', alpha=0.5) # Đường Zero cắt ngang
        plt.legend(loc='upper right', bbox_to_anchor=(1.05, 1.15))
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.show()

    def analyze_ticker(self, ticker):
        """
        Chiết xuất và hiển thị bảng so sánh (Dataframe) + Biểu đồ Thực tế vs Dự báo của riêng một Cổ phiếu cụ thể.
        """
        if not hasattr(self, 'predictions_df'):
            print("Chưa có dữ liệu dự báo. Vui lòng chạy walk_forward_competition() trước!")
            return None
            
        ticker_df = self.predictions_df[self.predictions_df['ticker'] == ticker].copy()
        
        if ticker_df.empty:
            print(f"Không tìm thấy dữ liệu cho mã {ticker}!")
            return None
            
        # Vẽ biểu đồ So sánh Cụ thể mã này
        plt.figure(figsize=(10, 5))
        
        # Format thời gian lại cho đẹp trên đồ thị
        x_labels = ticker_df['Quarter_Time'].apply(lambda x: str(pd.to_datetime(x).to_period('Q')))
        
        plt.plot(x_labels, ticker_df['y_true'], label='ACTUAL (y_true)', color='black', linewidth=3, marker='o')
        
        colors = ['red', 'blue', 'green', 'orange']
        for name, color in zip(self.models.keys(), colors):
            plt.plot(x_labels, ticker_df[f'pred_{name}'], label=f'Predicted: {name}', linestyle='--', marker='x', color=color)
            
        plt.title(f'Future Return Tracking for {ticker} (Actual vs Predicted)', fontsize=14, fontweight='bold')
        plt.xlabel('Evaluation Quarter', fontsize=11)
        plt.ylabel('Expected Return (y_return)', fontsize=11)
        plt.axhline(0, color='gray', linestyle='-', alpha=0.5)
        plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1))
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.show()
        
        return ticker_df

    def generate_weights_matrix(self, top_k=5, chosen_model='XGBoost'):
        """
        Giai đoạn 1: Biến Đổi Lợi Nhuận Dự Báo -> Ma Trận Trọng Số Bố Trí Vốn
        Chỉ chọn mua TOP K cổ phiếu có dự báo y_return cao nhất trong mỗi mốc thời gian.
        """
        if not hasattr(self, 'predictions_df'):
            print("Chưa có predictions_df. Vui lòng chạy walk_forward_competition() trước!")
            return None
            
        pred_col = f'pred_{chosen_model}'
        if pred_col not in self.predictions_df.columns:
            print(f"Mô hình {chosen_model} chưa được huấn luyện!")
            return None
            
        df = self.predictions_df.copy()
        
        # Đổi Tên Cột Thời Gian thành định dạng Ngày để khớp chuẩn Thư viện bt
        df['Date'] = pd.to_datetime(df['Quarter_Time'])
        
        weights_list = []
        
        # Duyệt qua từng Quý để chốt danh sách mua
        for date, group in df.groupby('Date'):
            # Lọc bớt các mã dự báo âm (Giải pháp Đầu tư An toàn)
            positive_preds = group[group[pred_col] > 0]
            
            if positive_preds.empty:
                # Nếu thị trường quá xấu dự báo toàn âm, cắt ra tiền mặt (Trọng số = 0)
                continue
                
            # Xếp hạng Top K Danh Tướng
            top_stocks = positive_preds.nlargest(top_k, pred_col)
            
            # Chia đều vốn (Allocation 1/K)
            weight_per_stock = 1.0 / len(top_stocks)
            
            # Tạo dictionary phân bổ vốn
            step_weights = {'Date': date}
            for ticker in top_stocks['ticker']:
                step_weights[ticker] = weight_per_stock
                
            weights_list.append(step_weights)
            
        weights_df = pd.DataFrame(weights_list).set_index('Date')
        weights_df = weights_df.fillna(0.0) # Những mã không được gọi tên = 0% vốn
        
        return weights_df
