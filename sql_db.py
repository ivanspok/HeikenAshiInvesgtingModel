from typing import List
from typing import Optional
from sqlalchemy import ForeignKey
from sqlalchemy import Integer, String, Column, DateTime, TIMESTAMP, Float
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy import create_engine
from sqlalchemy.orm import Session 
from sqlalchemy import select
from datetime import datetime
import pathlib
import os
import pytz
import pandas as pd
import pickle
import numpy as np
import time

from colog.colog import colog
c = colog()
warning = colog(TextColor='orange')
alarm = colog(TextColor='red')

class Base(DeclarativeBase):
    pass

class Orders(Base):
    __tablename__ = "orders"
    id = Column(Integer, primary_key=True)
    ticker = Column(String(20))
    buy_time = Column(DateTime) 
    buy_price = Column(Float)
    buy_sum = Column(Float)
    buy_commission = Column(Float)
    sell_time = Column(DateTime) 
    sell_price = Column(Float)
    sell_sum = Column(Float)
    sell_commission = Column(Float)
    stocks_number = Column(Integer)
    status = Column(String(30))
    gain_coef = Column(Float)
    trailing_LIT_gain_coef = Column(Float)
    trailing_ratio = Column(Float)
    lose_coef = Column(Float)
    profit = Column(Float)
    buy_order_id = Column(String(30))
    limit_if_touched_order_id = Column(String(30))
    stop_order_id = Column(String(30))
    trailing_LIT_order_id = Column(String(30))
    trailing_stop_limit_order_id = Column(String(30))
    timezone = Column(String(40))   

    def __repr__(self) -> str:
        return f"Order(id={self.id!r}, order_id={self.order_id!r})"

class DB_connection():
   
   def __init__(self, folder_path, db_name, df = None):
      self.folder_path = folder_path
      self.db_name = db_name
      self.db_path = pathlib.Path.joinpath(folder_path, db_name)
      self.engine_init()

      self.columns = [
       'ticker', 'id', 'buy_time', 'buy_price', 'buy_sum', 'buy_commission',
       'sell_time', 'sell_price', 'sell_sum', 'sell_commission', 'stocks_number', 'status',
       'gain_coef', 'lose_coef', 'trailing_LIT_gain_coef', 'trailing_ratio', 'profit',
       'buy_order_id', 'limit_if_touched_order_id', 'stop_order_id', 'trailing_LIT_order_id', 
       'trailing_stop_limit_order_id','timezone'
      ]

      self.timezone = str(datetime.now().astimezone().tzinfo)

      if not(pathlib.Path(self.db_path).is_file())\
        and df is not None:
         self.update_db_from_df(df)

   def engine_init(self):
      if not(os.path.isdir(self.folder_path)):
        os.mkdir(self.folder_path)
      self.engine = create_engine(f'sqlite:///{pathlib.Path.joinpath(self.folder_path, self.db_name)}', echo=False)
      
   def update_db_from_df(self, df):
    
      # timezone = datetime.tzname(df['buy_time'].iloc[0])
      # timezone = 'E. Australia Standard Time'
      # if timezone == 'E. Australia Standard Time':
      #    timezone = 'Australia/Melbourne'
      # df['timezone'] = timezone
      with self.engine.begin() as connection:
        df.to_sql(name ='orders',
                  con = connection,
                  if_exists='replace',
                  index=False)

   def add_record(self, order):
     
    with Session(self.engine) as session:

      if type(order) in [dict, pd.Series]:
         for column in self.columns:
            locals()[column] = order[column]

      if type(order) == pd.DataFrame:
         for column in self.columns:
            locals()[column] = order[column].values[0]

      order = Orders(
            id =  order['id'],
            ticker = locals()['ticker'],
            buy_time = locals()['buy_time'],
            buy_price = locals()['buy_price'],
            buy_sum = locals()['buy_sum'],
            buy_commission = locals()['buy_commission'],
            sell_time = locals()['sell_time'],
            sell_price = locals()['sell_price'],
            sell_sum = locals()['sell_sum'],
            sell_commission = locals()['sell_commission'],
            stocks_number = locals()['stocks_number'],
            status = locals()['status'],
            gain_coef = locals()['gain_coef'],
            lose_coef = locals()['lose_coef'],
            trailing_LIT_gain_coef = locals()['trailing_LIT_gain_coef'],
            trailing_ratio = locals()['trailing_ratio'],
            profit = locals()['profit'],
            buy_order_id = locals()['buy_order_id'],
            limit_if_touched_order_id = locals()['limit_if_touched_order_id'],
            stop_order_id = locals()['stop_order_id'],
            trailing_LIT_order_id = locals()['trailing_LIT_order_id'],
            trailing_stop_limit_order_id = locals()['trailing_stop_limit_order_id'],
            timezone = self.timezone  
      )

      # try:
      #   session.add(order)
      #   session.commit()
      # except Exception as e:
      #   warning.print(e)

   def update_record(self, order):
      
      try:
        if type(order) in [dict, pd.Series]:
          for column in self.columns:
              if column in ['buy_time', 'sell_time']:
                if type(order[column]) == str and '+' in order[column]: 
                  converted_time = datetime.strptime(order[column].split('+')[0], '%Y-%m-%d %H:%M:%S.%f')
                elif type(order[column]) == pd.Timestamp:
                  converted_time = order[column]
                else:
                  converted_time = datetime(1971,1,1,0,0)
                locals()[column] = converted_time
              else:
                locals()[column] = order[column]
      except Exception as e:
        print(e)
      
      try:
        if type(order) == pd.DataFrame:
          for column in self.columns:
            # SQL Lite supports only time in datetime type
            if column in ['buy_time', 'sell_time']:
              dt64 = order[column].values[0]
              if dt64 not in [np.isnan, '', None]:
                dt_pd = pd.to_datetime(dt64)
                dt = pd.Timestamp.to_pydatetime(dt_pd)
              else:
                dt = datetime(1,1,1,0,0)
              locals()[column] = dt
            else: 
              locals()[column] = order[column].values[0]
      except Exception as e:
        warning.print(e)

      with Session(self.engine) as session:

        try:
          sql_order = session.query(Orders).filter_by(id= int(locals()['id']), buy_time=locals()['buy_time']).first()
          if sql_order is not None:
            for param in self.columns:
              if param != 'id':
                setattr(sql_order, param, locals()[param])
                time.sleep(0.1)
                session.commit()
          else:
              self.add_record(order)
        except Exception as e:
          warning.print(e)
      
if __name__ == "__main__":
  parent_path = pathlib.Path(__file__).parent
  folder_path = pathlib.Path.joinpath(parent_path, 'sql')
  db = DB_connection(folder_path, 'trade.db')
  Base.metadata.create_all(db.engine)
  
 
  df_name = 'real_trade_db.pkl'
  file_path = pathlib.Path.joinpath(parent_path,'db', df_name)
  if pathlib.Path(file_path).is_file():
    with open(file_path, 'rb') as file:
      df = pickle.load(file)
  print(df)
  db.add_record(df.iloc[0])
  # df['gain_coef'].iloc[0] = 1.005
  db.update_record(df.iloc[0])
  # db.add_record(order)

  # add conditions to read files from df
  if False:
    df_name = 'real_trade_db.pkl'
    file_path = pathlib.Path.joinpath(parent_path,'db', df_name)
    if pathlib.Path(file_path).is_file():
      with open(file_path, 'rb') as file:
        df = pickle.load(file)
        db.update_db_from_df(df)


