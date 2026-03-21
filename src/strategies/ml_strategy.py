import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns

class EnsembleMLStrategy:
    def __init__(self, df, features, target='y_return', train_window_quarters=12):
        """
        - df: Dataframe đã qua bước DataProcess (đã làm sạch)
        - features: Danh sách cấc cột đầu vào X
        - target: Cột nhãn mục tiêu y
        - train_window_quarters: Cửa sổ lăn (Số lượng quý dùng để Train trước khi Test Quý kế tiếp)
        """
        self.df = df.copy()
        self.features = features
        self.target = target
        self.train_window = train_window_quarters
        
        self.models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        if xgb:
            self.models['XGBoost'] = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42)
        if lgb:
            self.models['LightGBM'] = lgb.LGBMRegressor(
                n_estimators=100, 
                learning_rate=0.05, 
                random_state=42, 
                min_child_samples=2,  # Rất quan trọng khi Data nhỏ
                verbose=-1            # Tắt báo cáo Warning rác
            )

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
        
        performance_log = {name: [] for name in self.models.keys()}
        scaler = StandardScaler()
        
        # Biến dành riêng cho Vẽ Biểu Đồ
        self.timeline_quarters = []
        self.actual_history = []
        self.model_predictions_history = {name: [] for name in self.models.keys()}
        
        # Bảng Dataframe So sánh chi tiết từng mã
        self.detailed_predictions_list = []

        # Bắt đầu trượt (Rolling)
        total_steps = len(quarters) - self.train_window
        
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
                performance_log[name].append(mse_score)
                
            # Đóng gói Quý Test
            self.detailed_predictions_list.append(step_df)
            
        # Gộp toàn bộ lịch sử soi Cổ phiếu vào Dataframe
        self.predictions_df = pd.concat(self.detailed_predictions_list, ignore_index=True)
                
        # --- TỔNG KẾT KẾT QUẢ ---
        print("\n🏆 KẾT QUẢ KIỂM THỬ TỪNG MÔ HÌNH (Trung bình lỗi MSE trên tất cả các Quý Test)")
        print("-" * 60)
        
        leaderboard = {}
        for name, scores in performance_log.items():
            avg_mse = np.mean(scores)
            leaderboard[name] = avg_mse
            
        # Sắp xếp từ lỗi thấp nhất đến cao nhất
        sorted_leaderboard = dict(sorted(leaderboard.items(), key=lambda item: item[1]))
        
        for name, score in sorted_leaderboard.items():
            print(f"Mô hình: {name:<20} | Lỗi MSE trung bình: {score:.6f}")
        
        return sorted_leaderboard, self.models

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
        plt.title('Bảng Xếp Hạng Sai Số Mô Hình (Cột càng THẤP càng tốt)', fontsize=14, fontweight='bold')
        plt.ylabel('Mean Squared Error (MSE)')
        
        for i, v in enumerate(scores):
            plt.text(i, v + (max(scores)*0.01), f"{v:.5f}", color='black', ha='center', va='bottom')
            
        plt.show()
        
        # 2. Vẽ Biểu đồ Dòng thời gian: So sánh Dự Báo vs Thực Tế
        plt.figure(figsize=(12, 6))
        
        # Đường lợi suất thực tế của Thị trường (Đen, in đậm)
        plt.plot(self.timeline_quarters, self.actual_history, 
                 label='THỰC TẾ (Lợi suất TB Thị trường)', color='black', linewidth=4, marker='o')
        
        # Các đường dự báo của từng Mô hình (Đứt nét)
        colors = ['red', 'blue', 'green', 'orange']
        for (name, preds), color in zip(self.model_predictions_history.items(), colors):
            plt.plot(self.timeline_quarters, preds, label=f'Dự báo: {name}', linestyle='--', marker='o')
            
        plt.title('Độ Nhạy Theo Thời Gian: Lợi suất Thực tế vs Dự báo qua Các Mùa Giải Walk-Forward', fontsize=14, fontweight='bold')
        plt.xlabel('Quý Đánh Giá (Test Quarters)', fontsize=12)
        plt.ylabel('Trung bình Lợi suất Logarit (y_return)', fontsize=12)
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
        
        plt.plot(x_labels, ticker_df['y_true'], label='THỰC TẾ (y_true)', color='black', linewidth=3, marker='o')
        
        colors = ['red', 'blue', 'green', 'orange']
        for name, color in zip(self.models.keys(), colors):
            plt.plot(x_labels, ticker_df[f'pred_{name}'], label=f'Dự báo: {name}', linestyle='--', marker='x', color=color)
            
        plt.title(f'Theo Dấu Lợi Suất Tương Lai Của Mã {ticker} (Thực Tế vs Dự Báo)', fontsize=14, fontweight='bold')
        plt.xlabel('Quý Đánh Giá', fontsize=11)
        plt.ylabel('Lợi suất kỳ vọng (y_return)', fontsize=11)
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
