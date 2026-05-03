import bt
import pandas as pd
import matplotlib.pyplot as plt

class BacktestEngine:
    def __init__(self, weights_df, initial_capital=10000.0, prices_df=None):
        """
        Giai đoạn 2: Sàn Đấu Lịch Sử - Nhận Ma trận Trọng số đã qua xử lý từ AI đẻ Đóng gói vào Thư viện `bt`.
        """
        self.weights_df = weights_df.copy()
        self.initial_capital = initial_capital
        
        # Danh sách các mã cổ phiếu cần tải
        self.tickers = [col for col in self.weights_df.columns if col != 'Date']
        
        # Chuyển index thành datetime để khớp chuẩn bt
        self.weights_df.index = pd.to_datetime(self.weights_df.index)
        
        self.start_date = self.weights_df.index.min()
        # Mở rộng thêm 1 Quý để đo chặng cuối
        self.end_date = self.weights_df.index.max() + pd.DateOffset(months=3)

        if prices_df is not None:
            self.prices_df = self._prepare_prices(prices_df)

    @staticmethod
    def build_prices_from_fundamental_data(df, tickers=None, date_col='Quarter_Time', price_col='adj_close_q'):
        """
        Tạo ma trận giá local từ file raw_fundamental_data.csv đã lưu sẵn.
        Không gọi API. Giá dùng cột adj_close_q theo từng Quarter_Time.
        """
        required_cols = {'ticker', date_col, price_col}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f"Thiếu cột trong dữ liệu local để backtest: {sorted(missing_cols)}")

        price_data = df[['ticker', date_col, price_col]].copy()
        if tickers is not None:
            price_data = price_data[price_data['ticker'].isin(tickers)]

        price_data[date_col] = pd.to_datetime(price_data[date_col], errors='coerce')
        price_data[price_col] = pd.to_numeric(price_data[price_col], errors='coerce')
        price_data = price_data.dropna(subset=['ticker', date_col, price_col])

        prices_df = price_data.pivot_table(
            index=date_col,
            columns='ticker',
            values=price_col,
            aggfunc='last'
        )
        prices_df = prices_df.sort_index().ffill()
        prices_df.index.name = 'Date'
        return prices_df

    def _prepare_prices(self, prices_df):
        prices = prices_df.copy()
        prices.index = pd.to_datetime(prices.index)
        prices = prices.sort_index()

        missing_tickers = [ticker for ticker in self.tickers if ticker not in prices.columns]
        if missing_tickers:
            raise ValueError(f"Thiếu giá local cho các mã: {missing_tickers}")

        prices = prices[self.tickers].ffill().dropna(how='all')

        missing_dates = self.weights_df.index.difference(prices.index)
        if not missing_dates.empty:
            prices = prices.reindex(prices.index.union(self.weights_df.index)).sort_index().ffill()

        prices = prices.loc[self.start_date:self.end_date]
        prices = prices.dropna(how='all')

        if prices.empty:
            raise ValueError("Ma trận giá local rỗng. Kiểm tra cột Quarter_Time/adj_close_q trong CSV.")

        return prices

    def run_simulation(self):
        """
        Thiết lập Chiến lược, Gắn Ma Trận Trọng Số vào bt và Phóng Backtest!
        """
        if not hasattr(self, 'prices_df'):
            raise ValueError(
                "BacktestEngine cần prices_df local. Hãy tạo bằng "
                "BacktestEngine.build_prices_from_fundamental_data(raw_df, tickers=weights_df.columns)."
            )

        print("\nĐang khởi chiếu Sàn Đấu Backtest từ dữ liệu CSV local (không gọi API)...")

        # bt.algos.WeighTarget tự động nhìn vào bảng self.weights_df để tái phân bổ vốn.
        strategy = bt.Strategy('AI_Quantitative_Fund', [
            bt.algos.RunAfterDate(self.start_date),
            bt.algos.RunOnDate(*self.weights_df.index.tolist()), # Chỉ Trade vào những ngày chốt Quý
            bt.algos.WeighTarget(self.weights_df),
            bt.algos.Rebalance()
        ])

        # Kết hợp Giá Local và Chiến lược để bắt đầu đua
        self.backtest = bt.Backtest(strategy, self.prices_df, initial_capital=self.initial_capital)
        
        # Chạy giả lập
        self.res = bt.run(self.backtest)
        
        print("\n=== MÔ PHỎNG BACKTEST HOÀN TẤT ===")
        return self.res

    def report_kpis(self):
        """
        Trích xuất 4 Chỉ số Vàng của Quỹ (KPIs)
        """
        if not hasattr(self, 'res'):
            print("Chưa chạy run_simulation(). Vui lòng Backtest trước!")
            return
            
        # In Bảng Xếp Hạng Chuẩn Form Định Lượng
        self.res.display()
        
        # Vẽ Đường cong Vốn (Equity Curve) của tài khoản Initial Capital
        plt.figure(figsize=(12, 6))
        self.res.plot()
        plt.title('Biểu đồ Tăng trưởng Tài khoản Tích lũy (Equity Curve)', fontsize=14, fontweight='bold')
        plt.xlabel('Thời Gian')
        plt.ylabel(f'Giá trị Tài khoản ($)')
        plt.grid(True, linestyle=':', alpha=0.6)
        
        plt.tight_layout()
        plt.show()
