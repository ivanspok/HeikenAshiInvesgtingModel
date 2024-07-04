from datetime import datetime
import pandas as pd
import os, pathlib
import pickle
from colog.colog import colog
c = colog()

class TradeInterface():


  def __init__(self, platform, df_name=None):
    self.platform = platform
    self.commission = 0.002 # commission for test platform

    self.parent_path = pathlib.Path(__file__).parent
    if df_name:
      self.df_path = df_name

  def buy_order(self, ticker, buy_price, buy_sum=0, stocks_number=0):
    order = {}
    if buy_sum != 0:
      stocks_number = int(buy_sum / buy_price)
    elif stocks_number != 0:
      buy_sum = stocks_number * buy_price
    else:
      print("Buy sum or stocks numbrer should't be both zero to place the order!")

    order = {
      'ticker' : ticker,
      'buy_time' : datetime.now(),
      'buy_price' : buy_price,
      'buy_sum' : buy_sum,
      'buy_commision': 0,
      'sell_time' : None,
      'sell_price' : None,
      'sell_commission': 0,
      'stocks_number' : stocks_number,
      'status' : 'created',
      'id' : 1,
      'gain_coef': 0,
      'lose_coef' : 0
    }

    if self.platform == 'test':
      order['buy_commission'] = buy_sum * self.commission
      response = 'success'
    
    if response == 'success':    
      order['status'] = 'bought'
      order['id'] = 1   # need connect to SQL database

    return order


  def sell_order(self, order, sell_price):
    if order['status'] == 'bought':
      if self.platform == 'test':
        order['sell_time'] = datetime.now()
        order['sell_price'] = sell_price
        order['sell_sum'] = sell_price * order['stocks_number']
        order['sell_commission'] = order['sell_sum'] * self.commission
        response = 'success'
      
      if response == 'success': 
        order['profit'] = order['sell_price'] - order['buy_price'] - order['buy_commission'] - order['sell_commission']
        order['status'] = 'completed'
    return order
    
  def record_order(self, order):

    folder_name = 'db'
    folder_path = pathlib.Path.joinpath(self.parent_path, folder_name)
    if not(os.path.isdir(folder_path)):
        os.mkdir(folder_path)

    file_path = pathlib.Path.joinpath(folder_path, self.df_path + '.pkl')
    if pathlib.Path(file_path).is_file():
      with open(file_path, 'rb') as file:
        df = pickle.load(file)
        df = pd.concat([df, pd.DataFrame([order])], ignore_index=True)
    else:
      df = pd.DataFrame([order])
    
    try:
      with open(file_path, 'wb') as file:
        pickle.dump(df, file)
        c.print('df file saved', color='green')
        try:
          csv_path = pathlib.Path.joinpath(folder_path, self.df_path + '.csv')
          df.to_csv(csv_path)
          c.print('cvs file saved', color='green')
        except Exception as e:
          c.print(f'{e}')
    except Exception as e:
        c.print(f'{e}')
    
if __name__ == '__main__':
  ti = TradeInterface(platform='test', df_name='test')
  order = ti.buy_order( ticker='AAPL', buy_price=100, buy_sum=1000)
  c.print(f'Place order: {order}', color='red')
  ti.record_order(order)
  order = ti.sell_order(order, sell_price=120)
  c.print(f'Sell order: {order}', color='green')
  ti.record_order(order)
