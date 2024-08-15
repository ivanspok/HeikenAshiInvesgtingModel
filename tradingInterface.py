from datetime import datetime
import pandas as pd
import os, pathlib
import pickle
import moomoo as ft
from colog.colog import colog
c = colog()
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

  def buy_order(self, ticker, buy_price, buy_sum=0, stocks_number=0, **kwargs):
    order = {}
    if buy_sum != 0:
      stocks_number = int(buy_sum / buy_price)
    
    buy_sum = stocks_number * buy_price

    if buy_sum == 0 and stocks_number == 0:
      print("Buy sum or stocks numbrer should't be both zero to place the order!")

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
      'id' : 1,
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
      moomoo_order, order_id = self.moomoo_api.place_market_buy_order(ticker, buy_price, stocks_number)
      if not (order_id is None):
        if moomoo_order['order_status'] == ft.OrderStatus.SUBMITTING \
          or moomoo_order['order_status'] == ft.OrderStatus.SUBMITTED:
          response = 'success'
          order['buy_order_id'] = order_id
          order['buy_price'] = moomoo_order['price']
          order['buy_sum'] = moomoo_order['price'] * moomoo_order['qty'] 
          order['stocks_number'] = moomoo_order['qty'] 
          order['buy_commission'] = 1.111
      else:
        print(f'Problem with placing the order for the {ticker}')
        response = 'error'
    
    if response == 'success':    
      order['status'] = 'bought'
      order['id'] = 1   # need connect to SQL database

    return order

  def sell_order(self, order, sell_price, **kwargs):
    if order['status'] == 'bought':
      if self.platform == 'test':
        order['sell_time'] = datetime.now().astimezone()
        order['sell_price'] = sell_price
        order['sell_sum'] = sell_price * order['stocks_number']
        order['sell_commission'] = order['sell_sum'] * self.commission
        response = 'success'

      if self.platform == 'moomoo':
        historical_order = kwargs['historical_order']
        order['sell_time'] = historical_order['updated_time'].astimezone()
        order['sell_price'] = historical_order['dealt_avg_price']
        order['sell_sum'] = historical_order['dealt_avg_price'] * historical_order['qty']
        sell_commision = self.moomoo_api.order_fee_query(historical_order['order_id'])
        if sell_commision is None:
          sell_commision = 1.01
        order['sell_commission'] = sell_commision
        if historical_order['order_status'] == ft.OrderStatus.FILLED_ALL:
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
    else:
      c.print('Trade history do not exist', color='yellow')
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

  def __save_orders__(self, df, file_path):
    try:
      with open(file_path, 'wb') as file:
        pickle.dump(df, file)
        c.print('df file saved', color='green')
        try:
          csv_path = pathlib.Path.joinpath(self.folder_path, self.df_path + '.csv')
          df.to_csv(csv_path)
          c.print('cvs file saved', color='green')
        except Exception as e:
          c.print(f'{e}')
    except Exception as e:
        c.print(f'{e}')

  def __update_sql_db__():
    pass

  def update_order(self, order):
   
    if not(os.path.isdir(self.folder_path)):
      os.mkdir(self.folder_path)

    file_path = pathlib.Path.joinpath(self.folder_path, self.df_path + '.pkl')
    if pathlib.Path(file_path).is_file():
      with open(file_path, 'rb') as file:
        df = pickle.load(file)
        index = df.loc[(df['id'] == order['id']) & (df['buy_time'] == order['buy_time'])].index
        update_line = pd.DataFrame([order])
        update_line[float_columns] = update_line[float_columns].astype(float)
        df.iloc[index] = update_line
    else:
      df = pd.DataFrame([order])
      df[float_columns] = df[float_columns].astype(float)

    self.__save_orders__(df, file_path)
    self.__update_sql_db__()
    return df

  def record_order(self, order):

    if not(os.path.isdir(self.folder_path)):
        os.mkdir(self.folder_path)

    file_path = pathlib.Path.joinpath(self.folder_path, self.df_path + '.pkl')
    if pathlib.Path(file_path).is_file():
      with open(file_path, 'rb') as file:
        df = pickle.load(file)
        df2 = pd.DataFrame([order])
        df2[float_columns] = df2[float_columns].astype(float)
        df = pd.concat([df, df2], ignore_index=True)
    else:
      df = pd.DataFrame([order])
      df[float_columns] = df[float_columns].astype(float)
    self.__save_orders__(df, file_path)
    return df
  
  def stock_is_bought(self, ticker, df):
  
    df = df[(df['ticker'] == ticker) & (df['status'] == 'bought')]
    return not(df.empty)

  def limit_if_touched_order_set(self, ticker, df):
    df = df[(df['ticker'] == ticker) & (df['status'] == 'bought') & (df['limit_if_touched_order_id'].iloc[0] != None)]
    return not(df.empty)
  
  def stop_order_set(self, ticker, df):
    df = df[(df['ticker'] == ticker) & (df['status'] == 'bought') & (df['stop_order_id'].iloc[0] != None)]

    return not(df.empty)

  # if ticker in bought_stocks \
# and ticker in df and status is bought 
# and not ORDER LIMIT and STOP

if __name__ == '__main__':
  ti = TradeInterface(platform='test', df_name='testing')
  df = ti.load_trade_history()
  # Tests:
  order = df[(df['status'] == 'bought')].iloc[0]
  order['stop_order_id'] = 2343
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

