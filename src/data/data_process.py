import pandas as pd
import numpy as np

class DataProcess: 
    def __init__(self, df):
        self.df = df.copy() # Tránh warning SettingWithCopy
        print("Đã khởi tạo DataProcess. Chuẩn bị Feature Engineering...")
    
    def extract_features(self):
        df = self.df
        
        # Xử lý các chỉ số chưa ép kiểu Float về đúng định dạng chuẩn
        numeric_convert_cols = [
            'EPS (VND)', 'EPS_basis', 'BVPS (VND)', 'Dividend yield (%)', 
            'Current Ratio', 'CURRENT ASSETS (Bn. VND)', 'Current liabilities (Bn. VND)',
            'Quick Ratio', 'Inventories, Net (Bn. VND)', 'Cash Ratio', 
            'Cash and cash equivalents (Bn. VND)', 'Days Sales Outstanding',
            'Revenue (Bn. VND)', 'Accounts receivable (Bn. VND)', 'LIABILITIES (Bn. VND)',
            'TOTAL ASSETS (Bn. VND)', 'Debt/Equity', "OWNER'S EQUITY(Bn.VND)",
            'P/E', 'P/S', 'P/B', 'ROE (%)', 'Net Profit Margin (%)', 'adj_close_q'
        ]
        
        # Ép kiểu an toàn (Bỏ qua lỗi nếu lỡ truyền Dataframe thiếu cột)
        for c in numeric_convert_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')

        # --- TÍNH TOÁN 14 CHỈ SỐ LỖI (FEATURES) --- #
        
        # 1. EPS: Ưu tiên lấy từ 'EPS (VND)', dự phòng bằng 'EPS_basis'
        df['EPS'] = df.get('EPS (VND)', np.nan).fillna(df.get('EPS_basis', np.nan))
        
        # 2. BPS:
        df['BPS'] = df.get('BVPS (VND)', np.nan)
        
        # 3. DPS (Tiền Cổ tức kiếm được): Cổ tức = Tỷ suất cổ tức * Giá cổ phiếu
        df['DPS'] = (df.get('Dividend yield (%)', 0).fillna(0) / 100) * df.get('adj_close_q', 0)
        
        # 4. cur_ratio:
        if 'Current Ratio' in df.columns:
            df['cur_ratio'] = df['Current Ratio']
        else:
            # Dự phòng công thức cơ sở
            _ca = df.get('CURRENT ASSETS (Bn. VND)', 0)
            _cl = df.get('Current liabilities (Bn. VND)', 1).replace(0, np.nan)
            df['cur_ratio'] = _ca / _cl
            
        # 5. quick_ratio:
        if 'Quick Ratio' in df.columns:
            df['quick_ratio'] = df['Quick Ratio']
        else:
            _ca = df.get('CURRENT ASSETS (Bn. VND)', 0)
            _inv = df.get('Inventories, Net (Bn. VND)', 0).fillna(0)
            _cl = df.get('Current liabilities (Bn. VND)', 1).replace(0, np.nan)
            df['quick_ratio'] = (_ca - _inv) / _cl
            
        # 6. cash_ratio:
        if 'Cash Ratio' in df.columns:
            df['cash_ratio'] = df['Cash Ratio']
        else:
            _cash = df.get('Cash and cash equivalents (Bn. VND)', 0)
            _cl = df.get('Current liabilities (Bn. VND)', 1).replace(0, np.nan)
            df['cash_ratio'] = _cash / _cl
            
        # 7. acc_rec_turnover: Vòng quay phải thu
        if 'Days Sales Outstanding' in df.columns:
            # 1 năm 360 ngày / Phải thu bao nhiêu ngày
            _dso = df['Days Sales Outstanding'].replace(0, np.nan)
            df['acc_rec_turnover'] = 360 / _dso
        else:
            _rev = df.get('Revenue (Bn. VND)', 0)
            _ar = df.get('Accounts receivable (Bn. VND)', 1).replace(0, np.nan)
            df['acc_rec_turnover'] = _rev / _ar
            
        # 8. debt_ratio: Tỷ lệ Nợ / Tổng tài sản
        _liab = df.get('LIABILITIES (Bn. VND)', 0)
        _assets = df.get('TOTAL ASSETS (Bn. VND)', 1).replace(0, np.nan)
        df['debt_ratio'] = _liab / _assets
        
        # 9. debt_to_equity:
        if 'Debt/Equity' in df.columns:
            df['debt_to_equity'] = df['Debt/Equity']
        else:
            _liab = df.get('LIABILITIES (Bn. VND)', 0)
            _equity = df.get("OWNER'S EQUITY(Bn.VND)", 1).replace(0, np.nan)
            df['debt_to_equity'] = _liab / _equity
            
        # 10. P/E
        df['pe'] = df.get('P/E', np.nan)
        
        # 11. P/S
        df['ps'] = df.get('P/S', np.nan)
        
        # 12. P/B
        df['pb'] = df.get('P/B', np.nan)
        
        # 13. ROE
        df['roe'] = df.get('ROE (%)', np.nan)
        
        # 14. Net Income Margin
        df['net_income_ratio'] = df.get('Net Profit Margin (%)', np.nan)
        
        # --- LƯU TRỮ FINAL --- #
        
        # Chỉ lấy 14 Biến + Biến định danh + Target Y
        keep_cols = [
            'ticker', 'Quarter_Time', 'yearReport', 'lengthReport',
            'EPS', 'BPS', 'DPS', 'cur_ratio', 'quick_ratio', 'cash_ratio',
            'acc_rec_turnover', 'debt_ratio', 'debt_to_equity',
            'pe', 'ps', 'pb', 'roe', 'net_income_ratio',
            'y_return' # Biến Target
        ]
        
        # Lấy riêng các cột có tồn tại trong Dataframe gốc (Tránh sinh lỗi nếu bảng Data thiếu cột)
        cols_to_keep = [c for c in keep_cols if c in df.columns]
        df_clean = df[cols_to_keep].copy()
        
        # === VỆ SINH CHUYÊN SÂU DÀNH CHO FINRL ===
        
        # 1. Chuyển các chỉ báo Vô Cực (Infinity rác trên API do chia cho số 0) thành Giá trị Lỗ (NaN)
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
        
        # 2. Xử lý Missing Value (NaN): Chiến lược Forward Fill cực mạnh
        # Chúng ta giả định nếu quý này API bị mất tỷ số P/E, ta sẽ tự mượn chỉ số P/E của Quý Liền Trước đắp qua.
        df_clean = df_clean.sort_values(by=['ticker', 'Quarter_Time'])
        
        # Cập nhật riêng cho nhóm Data Number
        feature_cols = [c for c in cols_to_keep if c not in ['ticker', 'Quarter_Time', 'yearReport', 'lengthReport', 'y_return']]
        df_clean[feature_cols] = df_clean.groupby('ticker')[feature_cols].ffill()
        
        # 3. Những NaN cứng đầu (như tại Q1 của mã đo, chưa có quý trước để mượn) sẽ được lấp bằng 0.
        df_clean[feature_cols] = df_clean[feature_cols].fillna(0)
        
        print(f"✅ Engineering 14 Features thành công! Dữ liệu đạt {df_clean.shape[0]} dòng và {df_clean.shape[1]} cột chuẩn bị châm vào Model.")
        return df_clean