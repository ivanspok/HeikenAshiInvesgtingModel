from typing import List
from typing import Optional
from sqlalchemy import ForeignKey
from sqlalchemy import Integer, String, Column, DateTime, TIMESTAMP, Float
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship

from sqlalchemy import create_engine
from sqlalchemy.orm import Session 
from sqlalchemy import select
from datetime import datetime
import pathlib
import os
import pytz
import pandas as pd
import pickle

class Base(DeclarativeBase):
    pass

class Orders(Base):
    __tablename__ = "orders"
    id = Column(Integer, primary_key=True)
    ticker = Column(String(20))
    order_id=  Column(String(50))
    buy_time = Column(DateTime) 
    buy_price = Column(Float)
    buy_sum = Column(Float)
    buy_commission = Column(Float)
    sell_time = Column(DateTime) 
    sell_price = Column(Float)
    sell_commission = Column(Float)
    stocks_number = Column(Integer)
    status = Column(String(30))
    gain_coef = Column(Float)
    lose_coef = Column(Float)
    sell_sum = Column(Float)
    profit = Column(Float)
    buy_order_id = Column(String(30))
    limit_if_touched_order_id = Column(String(30))
    stop_order_id = Column(String(30))
    timezone = Column(String(40))   
    # id = Column(Integer)

    def __repr__(self) -> str:
        return f"Order(id={self.id!r}, order_id={self.order_id!r})"

class DB_connection():
   
   def __init__(self, folder_path, db_name):
      self.folder_path = folder_path
      self.db_name = db_name
      self.engine_init()

   def engine_init(self):
      if not(os.path.isdir(self.folder_path)):
        os.mkdir(self.folder_path)
      self.engine = create_engine(f'sqlite:///{pathlib.Path.joinpath(self.folder_path, self.db_name)}', echo=True)
      with self.engine.connect() as conn:
         pass
      
   def update_db_from_df(self, df):
      
      # df add column timezone!!!
      timezone = datetime.tzname(df['buy_time'].iloc[0])
      if timezone == 'E. Australia Standard Time':
         timezone = 'Australia/Melbourne'
      df['timezone'] = timezone
      with self.engine.begin() as connection:
        df.to_sql(name ='orders',
                  con = connection,
                  if_exists='replace')

      # with Session(self.engine) as session:
        
      #   for index, row in df.iterrows():
      #      order = Orders(
      #         ticker 
      #  )
           
        # session.add_all([test_order])
        # session.commit()

   def add_record(self):
      pass

   def update_record(self):
      pass  


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
      db.update_db_from_df(df)

  # Write to db
  # with Session(db.engine) as session:
  #     tzinfo = pytz.timezone('Australia/Melbourne')
  #     test_order = Orders(
  #         ticker='ORM',
  #         order_id = '2342efsf3',
  #         buy_time = datetime.now().astimezone(tz=tzinfo),
  #         timezone = tzinfo.zone
  #     )
  #     session.add_all([test_order])
  #     session.commit()

  # Update db
  # with Session(db.engine) as session:
  #     order = session.query(Orders).filter_by(ticker = 'AAPL').first()
  #     print(order)
  #     setattr(order, 'order_id', '1234567')
  #     session.commit()


  

#     from sqlalchemy import select

# session = Session(engine)

# stmt = select(Order)

# for item in session.scalars(stmt):
#     print(item)