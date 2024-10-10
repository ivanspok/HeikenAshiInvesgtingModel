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
import numpy as np
from colog.colog import colog
c = colog()
warning = colog(TextColor='orange')
alarm = colog(TextColor='red')

float_columns = ['buy_price', 'buy_sum', 'buy_commission', 'sell_price', 'sell_sum', 'sell_commission',
                  'gain_coef', 'lose_coef', 'trailing_LIT_gain_coef', 'profit', 'trailing_ratio']

class TradeInterface():


  def __init__(self, platform, df_name=None, **kwargs):
    self.platform = platform
    self.test_commission = 0.002 # commission for test platform

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
    if platform != 'test':
      try:
        folder_path = pathlib.Path.joinpath(self.parent_path, 'sql')
        self.db = sql_db.DB_connection(folder_path, 'trade.db')
      except Exception as e:
        alarm.print(e)

  def buy_order(self, ticker, buy_price, buy_condition_type, default, buy_sum=0, stocks_number=0, **kwargs):
    response = None
    order = {}
    if buy_sum != 0:
      stocks_number = int(buy_sum / buy_price)
       
    buy_sum = stocks_number * buy_price
    current_timezone = datetime.now().astimezone().tzinfo
    if buy_sum == 0 and stocks_number == 0:
      alarm.print("Buy sum or stocks numbrer should't be both zero to place the order!")
    if self.platform == 'test':
      gv.ORDERS_ID_test += 1
      id = gv.ORDERS_ID_test
    else:
      gv.ORDERS_ID += 1
      id = gv.ORDERS_ID
    order = {
      'id' : int(id),  # Generate by global ID 
      'ticker' : ticker,
      'buy_time' : datetime.now(),
      'buy_price' : buy_price,
      'buy_sum' : buy_sum,
      'buy_commission': 0,
      'sell_time' : datetime(1971,1,1,0,0),
      'sell_price' : None,
      'sell_commission': 0,
      'stocks_number' : int(stocks_number),
      'status' : 'created',
      'gain_coef': 1.0001,
      'lose_coef' : 1.0001,
      'trailing_LIT_gain_coef' : 1.006,
      'trailing_ratio': default.trailing_ratio,
      'sell_sum': 0, 
      'profit': 0,
      'buy_order_id': None,
      'limit_if_touched_order_id': None, 
      'stop_order_id' : None,
      'stop_limit_sell_order_id': None,
      'trailing_LIT_order_id': None,
      'trailing_stop_limit_order_id': None,
      'buy_condition_type': buy_condition_type,
      'tech_indicators': {},
      'timezone' : str(current_timezone)
    }

    if self.platform == 'test':
      order['buy_commission'] = buy_sum * self.test_commission
      response = 'success'

    if self.platform == 'moomoo':
      # if buy_condition_type == '1230':
      #   moomoo_order, order_id = self.moomoo_api.place_buy_limit_if_touched_order(ticker, buy_price, stocks_number)
      # else:
      if buy_condition_type == 'opening':
        moomoo_order, order_id = self.moomoo_api.place_stop_limit_buy_order(ticker, buy_price, stocks_number)
      else:
        moomoo_order, order_id = self.moomoo_api.place_buy_limit_order(ticker, buy_price, stocks_number)
      if order_id is not None:
        if moomoo_order['order_status'].values[0] == ft.OrderStatus.SUBMITTING \
          or moomoo_order['order_status'].values[0] == ft.OrderStatus.SUBMITTED \
          or moomoo_order['order_status'].values[0] == ft.OrderStatus.WAITING_SUBMIT:
          response = 'success'
          order['buy_order_id'] = order_id
          order['buy_price'] = moomoo_order['price'].values[0]
          order['buy_sum'] = moomoo_order['price'].values[0] * moomoo_order['qty'].values[0]
          order['stocks_number'] = moomoo_order['qty'].values[0]
          order['buy_commission'] = 1.111 # defatul commision; real commision will be updated from historical order
          order['buy_condition_type'] = buy_condition_type
      else:
        alarm.print(f'Problem with placing the order for the {ticker}')
        response = 'error'
    
    if response == 'success':    
      order['status'] = 'placed'

    return order

  def sell_order(self, order, sell_price, **kwargs):
    response = 'None'
    current_timezone = datetime.now().astimezone().tzinfo
    if order['status'] in ['bought', 'filled part']:
      if self.platform == 'test':
        order['sell_time'] = datetime.now()
        order['sell_price'] = sell_price
        order['sell_sum'] = sell_price * order['stocks_number']
        order['sell_commission'] = order['sell_sum'] * self.test_commission
        order['timezone'] = str(current_timezone)
        response = 'success'

      if self.platform == 'moomoo':
        historical_order = kwargs['historical_order']
        tzinfo_ny = pytz.timezone('America/New_York')
        # tzinfo = pytz.timezone('Australia/Melbourne')
        try:
          if type(historical_order) == pd.Series:
            order['sell_time'] = datetime.strptime(historical_order['updated_time'], '%Y-%m-%d %H:%M:%S.%f').replace(tzinfo=tzinfo_ny)
            order['sell_time'] = order['sell_time'].astimezone(current_timezone) # Change time to current timezone time
            order['sell_time'] = order['sell_time'].replace(tzinfo=None) # remove timezone
            order['sell_price'] = float(historical_order['dealt_avg_price'])
            order['sell_sum'] = float(historical_order['dealt_avg_price'] * historical_order['qty'])
            sell_commision = self.moomoo_api.get_order_commission(historical_order['order_id'])
          else:
            order['sell_time'] = datetime.strptime(historical_order['updated_time'].values[0], '%Y-%m-%d %H:%M:%S.%f').replace(tzinfo=tzinfo_ny) # New-York time
            order['sell_time'] = order['sell_time'].astimezone(current_timezone) # Change time to current timezone time
            order['sell_time'] = order['sell_time'].replace(tzinfo=None) # remove timezone
            order['sell_price'] = float(historical_order['dealt_avg_price'].values[0])
            order['sell_sum'] = float(historical_order['dealt_avg_price'].values[0] * historical_order['qty'].values[0])
            sell_commision = self.moomoo_api.get_order_commission(historical_order['order_id'].values[0])
        except Exception as e:
          alarm.print(e)
        if sell_commision is None:
          sell_commision = 1.101
        order['sell_commission'] = sell_commision
        if type(historical_order) == pd.Series:
          if historical_order['order_status'] == ft.OrderStatus.FILLED_ALL:
            response = 'success'
        else:
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
    if gv.ORDERS_ID == np.nan:
      gv.ORDERS_ID = 1
        
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
          'gain_coef': pd.Series(dtype='float'),
          'lose_coef' : pd.Series(dtype='float'),
          'trailing_LIT_gain_coef' : pd.Series(dtype='float'),
          'trailing_ratio': pd.Series(dtype='float'),
          'profit': pd.Series(dtype='float'),
          'buy_order_id' : pd.Series(dtype='int'), 
          'limit_if_touched_order_id': pd.Series(dtype='int'),
          'stop_order_id' : pd.Series(dtype='int'),   
          'stop_limit_sell_order_id': pd.Series(dtype='int'),   
          'trailing_LIT_order_id' : pd.Series(dtype='int'),         
          'trailing_stop_limit_order_id' : pd.Series(dtype='int'),
          'timezone': pd.Series(dtype='str'),
          'buy_condition_type': pd.Series(dtype='str'),
          'tech_indicators': pd.Series(dtype='str')
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
          xlsx_path = pathlib.Path.joinpath(self.folder_path, self.df_path + '.xlsx')
          df.to_csv(csv_path)
          df.to_excel(xlsx_path)
          c.print('cvs file saved', color='green')
        except Exception as e:
          alarm.print(f'{e}')
    except Exception as e:
        alarm.print(f'{e}')

  def __update_sql_db__(self):
    pass

  def update_order(self, df, order, sim=False):
   
    # index = df.loc[(df['id'] == order['id']) & (df['buy_time'] == order['buy_time'])].index
    try:
      # index = df.loc[(df['buy_order_id'] == order['buy_order_id']) & (df['buy_time'] == order['buy_time'])].index
      index = df.loc[(df['buy_order_id'] == order['buy_order_id'])].index
      update_line = pd.DataFrame([order])
      update_line[float_columns] = update_line[float_columns].astype(float)
      df.loc[index] = update_line[df.columns].values
      self.__save_orders__(df)
    except Exception as e:
      alarm.print(e)

    # if not sim:
    #   # Update order in the SQL:
    #   try:
    #     self.__update_sql_db__()
    #     self.db.update_record(update_line)
    #   except Exception as e:
    #     alarm.print(e)
    return df

  def record_order(self, df, order, sim=False):

    df2 = pd.DataFrame([order])
    df2[float_columns] = df2[float_columns].astype(float)
    if df.empty:
      df = df2
    else:
      df = pd.concat([df, df2], ignore_index=True)
    self.__save_orders__(df)
    # if not sim:
    #   # Add record to SQL:
    #   try:
    #     self.db.add_record(df2.iloc[0])
    #   except Exception as e:
    #     alarm.print(e)
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

