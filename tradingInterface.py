from datetime import datetime
import pandas as pd
import os, pathlib
import pickle
import moomoo as ft
from colog.colog import colog
import global_variables as gv
import pytz
from zoneinfo import ZoneInfo
import sql_db

from colog.colog import colog
c = colog()
warning = colog(TextColor='orange')
alarm = colog(TextColor='red')

float_columns = ['buy_price', 'buy_sum', 'buy_commission', 'sell_price', 'sell_sum', 'sell_commission', 'gain_coef', 'lose_coef', 'profit']

class TradeInterface():


  def __init__(self, platform, df_name=None, **kwargs):
    self.platform = platform
    self.commission = 0.002 # commission for test platform

    self.parent_path = pathlib.Path(__file__).parent
    if df_name:
      self.df_path = df_name
    self.folder_name = 'db'
    self.folder_path = pathlib.Path.joinpath(self.parent_path, self.folder_name)

    if 'moomoo_api' in kwargs:
      self.moomoo_api = kwargs['moomoo_api']
    else:
      self.moomoo_api = None

    # SQL INIT
    try:
      folder_path = pathlib.Path.joinpath(self.parent_path, 'sql')
      self.db = sql_db.DB_connection(folder_path, 'trade.db')
    except Exception as e:
      alarm.print(e)

  def buy_order(self, ticker, buy_price, buy_sum=0, stocks_number=0, **kwargs):
    response = 'None'
    order = {}
    if buy_sum != 0:
      stocks_number = int(buy_sum / buy_price)
    
    buy_sum = stocks_number * buy_price

    if buy_sum == 0 and stocks_number == 0:
      alarm.print("Buy sum or stocks numbrer should't be both zero to place the order!")
    if self.platform == 'test':
      gv.ORDERS_ID_test += 1
      id = gv.ORDERS_ID_test
    else:
      gv.ORDERS_ID += 1
      id = gv.ORDERS_ID
    order = {
      'ticker' : ticker,
      'buy_time' : datetime.now().astimezone(),
      'buy_price' : buy_price,
      'buy_sum' : buy_sum,
      'buy_commission': 0,
      'sell_time' : None,
      'sell_price' : None,
      'sell_commission': 0,
      'stocks_number' : stocks_number,
      'status' : 'created',
      'id' : id,  # Generate by global ID 
      'gain_coef': 0,
      'lose_coef' : 0,
      'sell_sum': 0, 
      'profit': 0,
      'buy_order_id': None,
      'limit_if_touched_order_id': None, 
      'stop_order_id' : None
    }

    if self.platform == 'test':
      order['buy_commission'] = buy_sum * self.commission
      response = 'success'

    if self.platform == 'moomoo':
      moomoo_order, order_id = self.moomoo_api.place_buy_limit_if_touched_order(ticker, buy_price, stocks_number)
      if not (order_id is None):
        if moomoo_order['order_status'].values[0] == ft.OrderStatus.SUBMITTING \
          or moomoo_order['order_status'].values[0] == ft.OrderStatus.SUBMITTED \
          or moomoo_order['order_status'].values[0] == ft.OrderStatus.WAITING_SUBMIT:
          response = 'success'
          order['buy_order_id'] = order_id
          order['buy_price'] = moomoo_order['price'].values[0]
          order['buy_sum'] = moomoo_order['price'].values[0] * moomoo_order['qty'].values[0]
          order['stocks_number'] = moomoo_order['qty'].values[0]
          order['buy_commission'] = 1.111 # defatul commision; real commision will be updated from historical order
      else:
        alarm.print(f'Problem with placing the order for the {ticker}')
        response = 'error'
    
    if response == 'success':    
      order['status'] = 'bought'

    return order

  def sell_order(self, order, sell_price, **kwargs):
    response = 'None'
    if order['status'] in ['bought', 'filled part']:
      if self.platform == 'test':
        order['sell_time'] = datetime.now().astimezone()
        order['sell_price'] = sell_price
        order['sell_sum'] = sell_price * order['stocks_number']
        order['sell_commission'] = order['sell_sum'] * self.commission
        response = 'success'

      if self.platform == 'moomoo':
        historical_order = kwargs['historical_order']
        tzinfo_ny = pytz.timezone('America/New_York')
        tzinfo = pytz.timezone('Australia/Melbourne')
        order['sell_time'] = datetime.strptime(historical_order['updated_time'].values[0], '%Y-%m-%d %H:%M:%S.%f').replace(tzinfo=tzinfo_ny) # New-York time
        order['sell_time'] = order['sell_time'].astimezone(tzinfo) # Melbourne Time
        order['sell_price'] = float(historical_order['dealt_avg_price'].values[0])
        order['sell_sum'] = float(historical_order['dealt_avg_price'].values[0] * historical_order['qty'].values[0])
        sell_commision = self.moomoo_api.get_order_commission(historical_order['order_id'].values[0])
        if sell_commision is None:
          sell_commision = 1.101
        order['sell_commission'] = sell_commision
        if historical_order['order_status'].values[0] == ft.OrderStatus.FILLED_ALL:
          response = 'success'
      
      if response == 'success': 
        order['profit'] = order['sell_sum'] - order['buy_sum'] - order['buy_commission'] - order['sell_commission']
        order['status'] = 'completed'
    return order
  
  def load_trade_history(self):
    if not(os.path.isdir(self.folder_path)):
        os.mkdir(self.folder_path)

    file_path = pathlib.Path.joinpath(self.folder_path, self.df_path + '.pkl')
    if pathlib.Path(file_path).is_file():
      with open(file_path, 'rb') as file:
        df = pickle.load(file)
        gv.ORDERS_ID  = df['id'].max()
    else:
      c.print('Trade history does not exist', color='yellow')
      gv.ORDERS_ID = 0
      df = pd.DataFrame(
        {
          'id' : pd.Series(dtype='int'),
          'ticker': pd.Series(dtype='str'),
          'buy_time' : pd.Series(dtype='datetime64[ns]'),
          'buy_price' : pd.Series(dtype='float'),
          'buy_sum' : pd.Series(dtype='float'),
          'buy_commission': pd.Series(dtype='float'),
          'sell_time' : pd.Series(dtype='datetime64[ns]'),
          'sell_price' : pd.Series(dtype='float'),
          'sell_sum': pd.Series(dtype='float'),
          'sell_commission': pd.Series(dtype='float'),
          'stocks_number' : pd.Series(dtype='int'),
          'status' : pd.Series(dtype='str'),
          'id' : pd.Series(dtype='int'),
          'gain_coef': pd.Series(dtype='float'),
          'lose_coef' : pd.Series(dtype='float'),
          'profit': pd.Series(dtype='float'),
          'limit_if_touched_order_id': pd.Series(dtype='int'),
          'stop_order_id' : pd.Series(dtype='int')          
        }
      )
    return df

  def __save_orders__(self, df):
    file_path = pathlib.Path.joinpath(self.folder_path, self.df_path + '.pkl')
    try:
      with open(file_path, 'wb') as file:
        pickle.dump(df, file)
        c.print('df file saved', color='green')
        try:
          csv_path = pathlib.Path.joinpath(self.folder_path, self.df_path + '.csv')
          df.to_csv(csv_path)
          c.print('cvs file saved', color='green')
        except Exception as e:
          alarm.print(f'{e}')
    except Exception as e:
        alarm.print(f'{e}')

  def __update_sql_db__(self):
    pass

  def update_order(self, df, order):
   
    # if not(os.path.isdir(self.folder_path)):
    #   os.mkdir(self.folder_path)

    file_path = pathlib.Path.joinpath(self.folder_path, self.df_path + '.pkl')
    # if pathlib.Path(file_path).is_file():
    #   with open(file_path, 'rb') as file:
    #     df = pickle.load(file)
    #     index = df.loc[(df['id'] == order['id']) & (df['buy_time'] == order['buy_time'])].index
    #     update_line = pd.DataFrame([order])
    #     update_line[float_columns] = update_line[float_columns].astype(float)
    #     df.iloc[index] = update_line
    # else:
    #   df = pd.DataFrame([order])
    #   df[float_columns] = df[float_columns].astype(float)
    
    index = df.loc[(df['id'] == order['id']) & (df['buy_time'] == order['buy_time'])].index
    update_line = pd.DataFrame([order])
    update_line[float_columns] = update_line[float_columns].astype(float)
    df.loc[index] = update_line

    self.__save_orders__(df)
    # Update order in the SQL:
    try:
      self.__update_sql_db__()
      self.db.update_record(update_line.iloc[0])
    except Exception as e:
      alarm.print(e)
    return df

  def record_order(self, df, order):

    # if not(os.path.isdir(self.folder_path)):
    #     os.mkdir(self.folder_path)

    file_path = pathlib.Path.joinpath(self.folder_path, self.df_path + '.pkl')
    # if pathlib.Path(file_path).is_file():
    #   with open(file_path, 'rb') as file:
    #     df = pickle.load(file)
    #     df2 = pd.DataFrame([order])
    #     df2[float_columns] = df2[float_columns].astype(float)
    #     df = pd.concat([df, df2], ignore_index=True)
    # else:
    #   df = pd.DataFrame([order])
    #   df[float_columns] = df[float_columns].astype(float)
    df2 = pd.DataFrame([order])
    df2[float_columns] = df2[float_columns].astype(float)
    df = pd.concat([df, df2], ignore_index=True)
    self.__save_orders__(df)
    # Add record to SQL:
    try:
      self.db.add_record(df2.iloc[0])
    except Exception as e:
      alarm.print(e)
    return df
  
  def stock_is_bought(self, ticker, df):
  
    df = df.loc[(df['ticker'] == ticker) & (df['status'] == 'bought')]
    return not(df.empty)

  def limit_if_touched_order_set(self, ticker, df):
    # check why useing iloc[0] !!!
    df = df[(df['ticker'] == ticker) & (df['status'] == 'bought') & (df['limit_if_touched_order_id'].iloc[0] != 'None')]
    return not(df.empty)
  
  def stop_order_set(self, ticker, df):
    # check why useing iloc[0] !!!
    df = df[(df['ticker'] == ticker) & (df['status'] == 'bought') & (df['stop_order_id'].iloc[0] != None)]

    return not(df.empty)

if __name__ == '__main__':
  ti = TradeInterface(platform='test', df_name='testing')
  df = ti.load_trade_history()
  # Tests:
  order = df[(df['status'] == 'bought')].iloc[0]
  order['stop_order_id'] = None
  order['limit_if_touched_order_id'] = 543
  df = ti.update_order(order)
  print(ti.stock_is_bought('AAPL', df))
  print(ti.limit_if_touched_order_set('AAPL', df))
  print(ti.stop_order_set('AAPL', df))

  # order = ti.buy_order(ticker='AAPL', buy_price=100, buy_sum=1000)
  # c.print(f'Place order: {order}', color='red')
  # df = ti.record_order(order)
  # order = ti.sell_order(order, sell_price=120)
  # c.print(f'Sell order: {order}', color='green')
  # df = ti.record_order(order)

