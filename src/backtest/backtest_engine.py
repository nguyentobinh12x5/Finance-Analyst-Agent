import bt
import pandas as pd
import time
import matplotlib.pyplot as plt
from vnstock import Quote

class BacktestEngine:
    def __init__(self, weights_df, initial_capital=10000.0):
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

    def fetch_daily_prices(self):
        
        str_start = self.start_date.strftime("%Y-%m-%d")
        str_end = self.end_date.strftime("%Y-%m-%d")
        
        print(f"Đang gọi API VNStock tải Giá Hàng Ngày từ {str_start} đến {str_end}...")
        price_dict = {}
        
        for ticker in self.tickers:
            time.sleep(0.2) # Chống ban IP
            try:
                # Gọi VNStock API với nguồn VCI theo hệ thống gốc
                quote = Quote(source="VCI", symbol=ticker)
                df_ticker = quote.history(start=str_start, end=str_end, interval="1D")
                
                if not df_ticker.empty and 'close' in df_ticker.columns:
                    df_ticker['time'] = pd.to_datetime(df_ticker['time'])
                    df_ticker.set_index('time', inplace=True)
                    price_dict[ticker] = df_ticker['close']
                else:
                    print(f"⚠️ {ticker}: Không có dữ liệu giá giai đoạn này.")
            except Exception as e:
                print(f"Lỗi tải {ticker}: {e}")
                
        # Gộp toàn bộ Giá của các con Cổ phiếu vào 1 Bảng chữ nhật
        self.prices_df = pd.DataFrame(price_dict)
        
        # Xử lý Ngày nghỉ lễ chứng khoán (Kéo dài giá của ngày hôm trước)
        self.prices_df = self.prices_df.ffill().dropna()
        
        print("✅ Tải Giá VNStock Thành Công!")
        return self.prices_df

    def run_simulation(self):
        """
        Thiết lập Chiến lược, Gắn Ma Trận Trọng Số vào bt và Phóng Backtest!
        """
        if not hasattr(self, 'prices_df'):
            self.fetch_daily_prices()

        print("\nĐang khởi chiếu Sàn Đấu Backtest (Simulation)...")

        # bt.algos.WeighTarget tự động nhìn vào bảng self.weights_df để tái phân bổ vốn.
        strategy = bt.Strategy('AI_Quantitative_Fund', [
            bt.algos.RunAfterDate(self.start_date),
            bt.algos.RunOnDate(*self.weights_df.index.tolist()), # Chỉ Trade vào những ngày chốt Quý
            bt.algos.WeighTarget(self.weights_df),
            bt.algos.Rebalance()
        ])

        # Kết hợp Giá Hàng Ngày và Chiến lược để bắt đầu đua
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
