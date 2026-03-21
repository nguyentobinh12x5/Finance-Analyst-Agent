import pandas as pd
import numpy as np
import time
# Import các module của vnstock
from vnstock import Listing, Quote, Company, Finance, Trading, Screener

class DataFetcher:
    def __init__(self, source="VCI"):
        self.source = source
    # Liệt kê tất cả mã chứng khoán theo nhóm phân loại. Ví dụ HOSE, VN30, VNMidCap, VNSmallCap, VNAllShare, VN100, ETF, HNX, HNX30, HNXCon, HNXFin, HNXLCap, HNXMSCap, HNXMan, UPCOM, FU_INDEX (mã chỉ số hợp đồng tương lai), CW (chứng quyền)
    def get_group_ticker(self, group_ticker: str):
        # Không dùng từ khoá built-in `list` đặt làm tên biến
        ticker_list = Listing(source=self.source).symbols_by_group(group_ticker=group_ticker)
        return ticker_list
        
    def get_merged_financial_reports(self, ticker_list: list):
        """
        Hàm lấy 4 loại báo cáo tài chính của danh sách mã cổ phiếu, ghép lại thành 1 Dataframe duy nhất.
        Lấy tối đa 20 quý gần nhất cho mỗi mã.
        """
        all_tickers_df = []
        
        for symbol in ticker_list:
            # Khởi tạo Finance API
            finance = Finance(symbol=symbol, source=self.source)
            
            print(f"Đang tải báo cáo tài chính cho {symbol}...")
            
            # Cấu hình retry 3 lần cho mỗi mã cổ phiếu
            max_retries = 3
            success = False
            
            for attempt in range(max_retries):
                try:
                    # TỰ ĐỘNG GIẢM TỐC API: Delay 1.5s giữa TỪNG request 
                    # 1 mã gồm 4 request = 6s. 1 phút chạy định mức 10 mã (40 request) < 60 request/min 
                    # => Đảm bảo cực kỳ an toàn, không bao giờ chạm ngưỡng chặn của server!
                    
                    bs = finance.balance_sheet(period='quarter')
                    time.sleep(2)
                    
                    is_df = finance.income_statement(period='quarter')
                    time.sleep(2)
                    
                    cf = finance.cash_flow(period='quarter')
                    time.sleep(2)
                    
                    rt = finance.ratio(period='quarter')
                    time.sleep(2)
                    
                    # Cứu hộ "Rate Limit Ngầm" (API in ra chữ vàng nhưng không báo Exception, trả về rỗng)
                    if bs is not None and getattr(bs, "empty", False) and attempt < max_retries - 1:
                        print(f" => Lần thử {attempt + 1}: Trả về bảng rỗng (Khả năng kẹt API). Đợi 20 giây...")
                        time.sleep(20)
                        continue # Gọi vòng lặp thử lại
                        
                    success = True
                    break # Thoát khỏi vòng lặp thử lại nếu đã có dữ liệu
                    
                except Exception as e:
                    error_msg = str(e)
                    print(f" Lỗi Code/Mạng ở Lần {attempt + 1}: {error_msg[:60]}...")
                    time.sleep(10)
            
            if not success:
                print(f" Bỏ qua {symbol} sau {max_retries} lần bị sập.")
                continue
            
            # Danh sách các cột khoá chính để merge
            key_cols = ['ticker', 'yearReport', 'lengthReport']
            df_list = []
            
            # Duyệt qua từng dataframe để tiền xử lý trước khi ghép
            for df in [bs, is_df, cf, rt]:
                if not df.empty:
                    df = df.copy()
                    
                    # Xử lý MultiIndex cho bảng ratio (flatten cột)
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = [col[1] for col in df.columns]
                        
                    # Ép kiểu string cho các cột khoá để join không bị lệch kiểu dữ liệu
                    for col in key_cols:
                        if col in df.columns:
                            df[col] = df[col].astype(str)
                    
                    # Đặt index bằng các cột chính để ghép theo chiều dòng
                    df = df.set_index(key_cols)
                    df_list.append(df)
                    
            if not df_list:
                print(f" Không có dữ liệu báo cáo nào cho {symbol}!")
                continue

            # 1. Ghép toàn bộ báo cáo của mã này (axis=1)
            merged_df = pd.concat(df_list, axis=1)
            
            # 2. Xoá những cột bị trùng tên
            merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]
            merged_df = merged_df.reset_index()
            
            # 3. Tạo cột Quarter_Time
            quarter_mapping = {'1': '31/03/', '2': '30/06/', '3': '30/09/', '4': '31/12/'}
            raw_quarter_time = merged_df['lengthReport'].map(quarter_mapping) + merged_df['yearReport']
            merged_df['Quarter_Time'] = pd.to_datetime(raw_quarter_time, format='%d/%m/%Y', errors='coerce')
            
            # Loại bỏ dòng thiếu ngày tháng (nếu có dữ liệu lỗi từ API)
            merged_df = merged_df.dropna(subset=['Quarter_Time'])
            
            # 4. Chỉ lấy 20 QUÁ KHỨ GẦN NHẤT
            merged_df = merged_df.sort_values(by='Quarter_Time', ascending=False).head(20)
            
            all_tickers_df.append(merged_df)
        
        if not all_tickers_df:
            print("Toàn bộ danh sách không có dữ liệu!")
            return pd.DataFrame()

        # Ghép tất cả các mã chứng khoán lại với nhau
        final_df = pd.concat(all_tickers_df, ignore_index=True)
        # Sắp xếp lại lần cuối để các dòng tăng dần theo thời gian (chuẩn Machine Learning)
        final_df = final_df.sort_values(by=['ticker', 'Quarter_Time']).reset_index(drop=True)
        
        print(f"\n=> Merge toàn bộ thành công! Bảng: {final_df.shape[0]} dòng x {final_df.shape[1]} cột.")
        return final_df
    def align_and_fetch_price(self, df, align_quarter_dates=True):
        """
        Điều chỉnh Quarter_Time tịnh tiến để lấy giá (do độ trễ ra báo cáo) 
        và tự động chèn giá trị adj_close_q vào bảng.
        """
        # Clone DataFrame để khỏi ảnh hưởng bản gốc
        df_out = df.copy()
        
        # 1. Hàm tính toán Target Date (Ngày mục tiêu để lấy giá)
        def compute_target_date(row):
            # Nếu không cần dời lịch, dùng luôn giá trị tại Quarter_Time
            if not align_quarter_dates:
                return pd.to_datetime(row['Quarter_Time'])
            
            # Nếu True, tính toán dời ngày
            year = int(float(row['yearReport']))
            length = str(row['lengthReport']).strip()
            
            if length == '1':   # Q1
                return pd.Timestamp(year, 6, 1)
            elif length == '2': # Q2
                return pd.Timestamp(year, 9, 1)
            elif length == '3': # Q3
                return pd.Timestamp(year, 12, 1)
            elif length == '4': # Q4 đẩy sang năm tiếp theo
                return pd.Timestamp(year + 1, 4, 1)
            else:
                return pd.to_datetime(row['Quarter_Time'])

        # Áp dụng tính toán sinh ra cột target_date tạm thời
        df_out['target_date'] = df_out.apply(compute_target_date, axis=1)
        df_out['target_date'] = pd.to_datetime(df_out['target_date'])
        
        # Tạo cột adj_close_q rỗng
        df_out['adj_close_q'] = np.nan
        df_out['actual_price_date'] = pd.NaT # Ghi log lại xem thực tế đã lấy giá ngày nào
        
        # 2. Xử lý lấy giá bằng vnstock Quote theo từng Ticker (Siêu tốc độ vì lấy Batch 1 lần/mã)
        unique_tickers = df_out['ticker'].unique()
        
        for ticker in unique_tickers:
            mask = (df_out['ticker'] == ticker)
            subset = df_out[mask]
            
            # Lấy khoảng thời gian của ticker này để fetch API 1 lần
            min_date = subset['target_date'].min() - pd.Timedelta(days=5)
            max_date = subset['target_date'].max() + pd.Timedelta(days=15) # Dư dả 15 ngày cho cuối tuần
            
            try:
                # Ngủ ngắn 1.5s để làm mát luồng API (do phải lặp qua nhiều ticker)
                time.sleep(1.5)
                print(f"Đang fetch giá {ticker} từ {(min_date.strftime('%Y-%m-%d'))} đến {(max_date.strftime('%Y-%m-%d'))}...")
                q = Quote(symbol=ticker, source=self.source)
                history = q.history(start=min_date.strftime('%Y-%m-%d'), end=max_date.strftime('%Y-%m-%d'))
                
                if not history.empty:
                    # Đưa cột time thành chuẩn datetime
                    history['time'] = pd.to_datetime(history['time'])
                    # Đảm bảo lịch sử giá sort tăng dần
                    history = history.sort_values(by='time')
                    
                    # Móc giá cho từng dòng của mã ticker này
                    for idx, row in subset.iterrows():
                        target_dt = row['target_date']
                        
                        # Lấy các ngày giao dịch >= target_date 
                        # Nếu target_date rơi vào T7/CN, iloc[0] sẽ tự ăn vào ngày thứ 2 tiếp theo!
                        valid_prices = history[history['time'] >= target_dt]
                        
                        if not valid_prices.empty:
                            matched_row = valid_prices.iloc[0] # Ngày giao dịch đầu tiên ngay sau ngưởng trễ
                            df_out.at[idx, 'adj_close_q'] = matched_row['close']
                            df_out.at[idx, 'actual_price_date'] = matched_row['time']
            except Exception as e:
                print(f"Lỗi khi lấy dữ liệu giá của {ticker}: {e}")

        # Xóa cột temp
        df_out = df_out.drop(columns=['target_date'])
        return df_out

    def calculate_y_return(self, df):
        """
        Tính Log Return tương lai (y_return) cho mục tiêu huấn luyện model ML.
        """
        df_out = df.copy()
        
        # 1. Đảm bảo ép kiểu các cột trước khi sort để đảm bảo thứ tự Quý [1, 2, 3, 4]
        df_out['yearReport'] = df_out['yearReport'].astype(int)
        df_out['lengthReport'] = df_out['lengthReport'].astype(int)
        
        # 2. Sort chuẩn theo đúng yêu cầu
        df_out = df_out.sort_values(by=['ticker', 'yearReport', 'lengthReport']).reset_index(drop=True)
        
        # 3. Tính toán forward log return
        # Vì dùng shift(-1), Quý hiện tại sẽ nhìn thấy Tỷ suất Sinh Lợi dẫn đến Quý tới
        # Các quỹ định lượng ép máy tính dùng lợi suất Log (Log return) cho đối xứng và ổn định 
        df_out['y_return'] = df_out.groupby('ticker')['adj_close_q'].transform(lambda s: np.log(s.shift(-1) / s))
        
        # Lưu ý: Các dòng ở Quý cuối cùng của mỗi Ticker bị Shift(-1) sẽ trả về NaN giá trị do không có tương lai phía trước. Có thể dùng .dropna(subset=['y_return']) sau cùng trước khi đưa vào FinRL.
        
        return df_out
