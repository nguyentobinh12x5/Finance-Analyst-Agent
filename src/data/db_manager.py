from sqlalchemy import create_all, create_engine
import pandas as pd
from src.config.settings import settings

class DatabaseManager:
    def __init__(self):
        self.engine = create_engine(settings.DB_PATH)
        
    def save_data(self, df: pd.DataFrame, table_name: str):
        df.to_sql(table_name, self.engine, if_exists='replace', index=True)
        
    def load_data(self, table_name: str) -> pd.DataFrame:
        return pd.read_sql(table_name, self.engine)