#%%Imports:
import os
from numpy.core.fromnumeric import std
import tinvest as ti

from datetime import datetime, timedelta
import time
from pytz import timezone

import numpy as np
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt

import telebot
from tinvest.schemas import Candles
# if __name__ == '__main__': 

import settings


dirname = os.path.dirname(__file__)
#%% settings
bot = settings.bot
current_chat = settings.current_chat
col = settings.col

#%% Functions
def reper_function():
  return 0

def get_turnover(client):
  '''
    Return today turnover
  '''

  turnover = 0

  try:
      to_ = datetime(datetime.now().year, datetime.now().month, datetime.now().day, 0, 0, 1, 0, timezone('Europe/Moscow')) + timedelta(days = 1)
      from_ = datetime(datetime.now().year, datetime.now().month, datetime.now().day, 3, 0, 0, 0, timezone('Europe/Moscow'))
      operations = client.get_operations(from_, to_)

      glass = client.get_market_orderbook('BBG0013HGFT4', depth = 20)
      usd_price = float(glass.payload.last_price)

      for operation in operations.payload.operations:

          if operation.operation_type == 'Buy' or operation.operation_type == 'Sell':
            
            if operation.currency[:] == 'RUB':

              turnover += abs (float (operation.payment))

            else:

              turnover += abs (float (operation.payment)) * usd_price

  except Exception as e:

      print(f'Problem with getting turnover over the day: {e}')
  
  return turnover

def get_candle_list(client, figi, resolution):

    close_price_list = []
    volume_list = []
    time_list = []
    candles_list = [] 
   
    if resolution == ti.CandleResolution.hour:
      # to_ = datetime(datetime.now().year, datetime.now().month, datetime.now().day, 0, 0, 1, 0, timezone('Europe/Moscow')) # working this way, but wrong calculation hour candles
      to_ = datetime(datetime.now().year, datetime.now().month, datetime.now().day, 0, 0, 1, 0, timezone('Europe/Moscow')) + timedelta(days = 1)
      from_ = to_ - timedelta(hours = 168)   
    
      response = client.get_market_candles(figi, from_, to_, resolution)
      
      for candel in response.payload.candles:
          close_price_list.append( float(candel.c) )
          volume_list.append(int (candel.v))          
          time_list.append(candel.time + timedelta(hours = 3))
          candles_list.append(candel)
    
    if resolution == ti.CandleResolution.min1 or resolution == ti.CandleResolution.min5 \
       or resolution == ti.CandleResolution.min30:

      for i in range(5, -1, -1):
        from_ = datetime(datetime.now().year, datetime.now().month, datetime.now().day, 0, 0, 1, 0, timezone('Europe/Moscow')) - timedelta(days = i)
        to_ = datetime(datetime.now().year, datetime.now().month, datetime.now().day, 0, 0, 1, 0, timezone('Europe/Moscow')) - timedelta(days = i - 1)
        response = client.get_market_candles(figi, from_, to_, resolution)        
        for candel in response.payload.candles:
          close_price_list.append( float(candel.c) )
          volume_list.append(int (candel.v))
          time_list.append(candel.time + timedelta(hours = 3))
          candles_list.append(candel)    

    return time_list, close_price_list, volume_list, candles_list

def WMA(series,window = 14,EMA0=-1):
  alpha = 1/(window)
  EMA=[]

  for i in range(window-1):
    EMA.append(-1)

  if EMA0 == -1:    
    SMA = sum(series[:window]) / window   
    EMA.append(SMA)
  else: 
    EMA.append(EMA0)    

  for i in range(window, len(series)):    
    EMA.append(alpha * series[i] + (1-alpha) * EMA[i-1])
  return EMA

def get_RSI(close_price, window = 14, EMA0 =-1):
  U = []
  D = []
  RSI = []
  EMA_U=[EMA0]
  EMA_D=[EMA0]
  RS = []

  if EMA0 !=-1:
    close_yestarday = EMA0
  else:
    close_yestarday = close_price[0]

  for close in close_price:    
    if close > close_yestarday:
      U.append(close - close_yestarday)
      D.append(0)
    elif close < close_yestarday:
      U.append(0)
      D.append(close_yestarday-close)
    else: 
      U.append(0) 
      D.append(0)
    close_yestarday = close      
   
  EMA_U = WMA(U, window)
  EMA_D = WMA(D, window)

  for ema_u, ema_d in zip(EMA_U, EMA_D):
    if (ema_u == -1) or (ema_d == -1):
      RSI.append(-2)
    else:
      if ema_d == 0:
        RSI.append(100)
      else:
        RS.append(ema_u / ema_d)
        RSI.append(100 - 100 / (1 + RS[-1]))
  return RSI

def STOCH(close_price, window = 14):

  K=[]
  D = [] 
  for i in range(len(close_price)):    
    if i < window:
      K.append(-2)    
    else:
      min_ = min(close_price[i-window:i+1])
      max_ = max(close_price[i-window:i+1])
      if max_ != min_:  
        K.append(100 * (close_price[i] - min_) / (max_ - min_))
      else:
        K.append(100)
  return K

def SMA(f, window):
  SMA = []
  for i in range(window - 1):
    SMA.append(f[i])
  for i in range(window, len(f) + 1):
    SMA.append(sum(f[i - window : i]) / window)
  return SMA

def ROC(f, window):
  ROC = []
  for i in range(window):
    ROC.append(100 * (f[i] / f[0] - 1))
  for i in range(window, len(f)):
    ROC.append(100 * (f[i] / f[i-window] -1))
  return ROC

def get_spline_acceleration(time, f, window = 30, show = 0):

  x_list = []
  y_list = []
  acc_list = []

  num_windows = len(time) // window
  start_i = len(time) - num_windows * window


  # end_i = len(time) // window - 1
  for i in range(start_i, len(time), window):
    a = i
    b = i + window
    x = np.arange(1, len(time)+1)[a : b]
    y = f[a : b]
    spline = UnivariateSpline(x, y, k = 3)
    x_list += x.tolist()
    y_list += spline(x).tolist()

  # a = window * (end_i) 
  # b = - window
  # x = np.arange(1, len(time)+1)[a : b]
  # y = f[a : b]
  # spline = UnivariateSpline(x, y, k = 3)
  # x_list += x.tolist()
  # y_list += spline(x).tolist()

  # a = - window
  # b = - 1
  # x = np.arange(1, len(time)+1)[a :]
  # y = f[a :]
  # spline = UnivariateSpline(x, y, k = 3)
  # x_list += x.tolist()
  # y_list += spline(x).tolist()   

  acc_list = [y_list[i] - y_list[i-1] for i in range(len(y_list))]

  if show:
    plt.plot(np.arange(1, len(time) + 1), f, x_list, y_list, x_list, acc_list)
    plt.show()
  
  return time[start_i:], acc_list, y_list


def pct_var_fun(f, window = 30):

  pct_var_list = []

  for i in range(len(f)):
    if i < window:
      pct_var_list.append( - 1 )
    else:
      temp_f = np.array(f[-window + i : i + 1 ])
      pct_var = np.var(temp_f) * 100 / (max(temp_f) - min(temp_f)) / np.mean(temp_f)
      pct_var_list.append(pct_var)

  return pct_var_list

def is_bokovichok(f, window = 30, delta = 0.01, for_getting_functions = False):

  if not(for_getting_functions):
    pct_var_fun_ = pct_var_fun(f, window)
    delta = 12 / 100 * max(pct_var_fun_)

  f = np.array(f[-window:])

  # f = (f - min(f)) / (max(f) - min(f))
  var = np.var(f)
  # pct_var = var * 100 / np.mean(f)
  pct_var = np.var(f) * 100 / (max(f) - min(f)) / np.mean(f)

  if pct_var < delta:
    return 1, pct_var
  else:
    return 0, pct_var

def get_bokovichok_function(f, window = 30):

  pct_var_fun_ = pct_var_fun(f, window)
  delta = 12 / 100 * max(pct_var_fun_) 

  out_f = []
  pct_var_list = []

  for i in range(len(f)):
    if i < window:
      out_f.append( - 1 )
      pct_var_list.append( - 1 )
    else:
      out, pct_var = is_bokovichok(f[: i + 1], window, delta = delta, for_getting_functions = True)
      out_f.append(out)
      pct_var_list.append(pct_var)
  return out_f, pct_var_list

def grad(f, delta = 0.01):
    # delta = 0.01
    if len(f) < 2: return 0

    f = np.array(f)
    f = (f - min(f)) / (max(f) - min(f))

    i = -2    
    while abs(f[-1] - f[i]) < delta and abs(i) < len(f):
      i -= 1    

    if f[-1] > f[i] + delta: return 1

    if abs(f[-1] - f[i]) <= delta: return 0
    
    if f[-1] < f[i] - delta: return -1

def f_range(value, low, high):
  if value >= low and value<= high: return True
  else: return False

def write_log(folder_name, filename, text):

  folder_name = os.path.join(dirname, folder_name)
  if not os.path.exists(folder_name):
      os.makedirs(folder_name)

  filename = os.path.join(folder_name, filename)
  if not os.path.isfile(filename):
    f = open(filename, 'w')
  else:
    f = open(filename, 'a')
  f.write(text)
  f.close()
#%%
def get_value_precision(number):
    s = str(number)
    if '.' in s:
        return abs(s.find('.') - len(s)) - 1
    elif 'e-' in s:
        return abs(s.find('2e-0') - len(s))
    else:
        return 0
#%%
def get_order_price(min_price_increment, sell_price_short, fixation_profit_value, buy = True, show = 0):
  
  round_precision = get_value_precision(min_price_increment)

  if buy:

    fixation_price = sell_price_short * (1 - fixation_profit_value)
    order_price = round(fixation_price // min_price_increment * min_price_increment , round_precision)

  else:

    fixation_price = sell_price_short * (1 + fixation_profit_value)
    order_price = round(fixation_price // min_price_increment * min_price_increment , round_precision) + min_price_increment
  
  # while (order_price + min_price_increment < fixation_price):
  #   order_price += min_price_increment
  # order_price = round(order_price, round_precision)

  if show:
    print(f'fixation_price is {fixation_price}, order_price is {order_price}')

  return order_price

# if __name__ == '__main__':
#   min_price_increment = 0.00003
#   fixation_profit_value = 0.2 / 100
#   sell_price_short = 0.0064
#   order_price = get_order_price(min_price_increment, sell_price_short, fixation_profit_value, show = 1)
# %%
def operation_check(operation):
  try:
    if abs(float (operation.price) ) != 0 \
        and abs(float (operation.payment) ) !=0:
        # and operation.commission is not None:
      return True
    else:
      return False
  except Exception as e:
    send_and_log_exceptions('operation_check function', e)
    return False
#%%
def get_operation_by_order_id(order_id, operations):

  for operation in operations.payload.operations:

    if int(operation.id) == int(order_id):
      return operation
  return 'Operation not found'

#%%
def is_in_new_orders(order_id, orders):

  for order in orders.payload:

    if int(order.order_id) == int(order_id) \
      and (order.status == 'New' or order.status == 'PartiallyFill'):
      return True
  return False

#%%
def cancel_orders_at_life_time(client, broker_account_id, operations, orders, col, order_id, time_of_life, order_time):

  is_check = 0
  is_cancel = 0

  operation = get_operation_by_order_id(order_id, operations)
  current_time = datetime.now()
  time_diff = current_time - order_time
  
  if time_of_life != - 1 \
     and time_diff.seconds >= time_of_life:
    
    print(f'time_diff.seconds is {time_diff.seconds}')
  
    if operation != 'Operation not found':

        is_check = 1
  
        # if operation.status == 'Decline' \
        #   or not( operation.status == 'Progress' \
        #       or int(operation.quantity_executed) > 0 \
        #       or operation.status == 'Done'):
              
        #       is_check = 1
    
    if is_in_new_orders(order_id, orders):
      
      is_check = 1
    
    print(f'is_check is {is_check}')
    
    if is_check:

      try:
        log_text = f'{current_time}:'  f'try to cancel order {order_id}' + '\n'              
        write_log(os.path.join('torgach_logs','cancel_orders'), f'cancel_orders_log.txt', log_text)
        print(f'try to cancel:')
        client.post_orders_cancel(int(order_id), broker_account_id)
        is_cancel = 1
      except Exception as e:
        send_and_log_exceptions('cancel_orders_at_life_time function', e)    
        # col.log(f'Exception arised in cancel_orders_at_life_time, order_id is {order_id}:') 
        # print(str(e))

  return is_cancel

def restore_order_if_decline(client, broker_account_id, ins_list, num, operations, order_name, waiting_time = - 1):

  current_time = datetime.now()

  order_id = vars(ins_list[num])[f'{order_name}_id']
  order_time = vars(ins_list[num])[f'{order_name}_id_time']

  operation = get_operation_by_order_id(order_id, operations)
  time_diff = current_time - order_time

  if waiting_time != - 1 \
     and time_diff.seconds >= waiting_time:
  
    if operation != 'Operation not found' \
        and (operation.status == 'Decline' \
        or (operation.quantity != operation.quantity_executed and operation.status == 'Done')):

      try:
        
        body = ti.LimitOrderRequest ( 
              lots =   int ((operation.quantity - operation.quantity_executed) // ins_list[num].lot),
              operation = operation.operation_type,
              price = operation.price
        )

        response = client.post_orders_limit_order(ins_list[num].figi, body, broker_account_id)
          
        log_text = f'{current_time}:' + str(response) + f'order {order_id} is restored due to previos decline' + '\n'              
        write_log(os.path.join('torgach_logs','restored_orders'), f'torgach_{ins_list[num].stock_name}_log.txt', log_text)

        vars(ins_list[num])[f'{order_name}_id_time'] = current_time
        vars(ins_list[num])[f'{order_name}_id'] = int(response.payload.order_id)
        
      except Exception as e:

        if 'invalid literal for int()' not in e:
          send_and_log_exceptions('restore_order_if_decline function', e)    
     

#%%
def send_and_log_exceptions(place, e, send_to_bot = True, bot = bot, chat = current_chat):


  message = f'Exception arised in {place}: {str(e)} + \n'

  if send_to_bot:
    
    bot.send_message(chat, message)

  col.log(message)

  current_time = datetime.now()

  write_log(os.path.join('torgach_logs'), f'exceptions.txt', f'{current_time}: ' + message)

#%%
def three_candles_criteria(candles, time):

  candles = np.array(candles)

  min_pct_value = 0.1
  permission_to_long = 0
  signal_to_long = 0
  signal_buy  = 0

  time_short = []
  points_short = []
  time_long = []
  points_long = []
  signal_short = 0


  candles_pct_list = []

  current_candles_criteria = 0

  for i in range(1, len(candles)):

    if candles[i] > candles[i - 1]:
    
      candle_pct = (candles[i] / candles[i - 1] - 1) * 100

    else:

      candle_pct = - ( candles[i - 1] / candles[i] - 1) * 100

    if current_candles_criteria == 0:

      if candle_pct > min_pct_value:

        current_candles_criteria = 1

      if candle_pct < - min_pct_value:

        current_candles_criteria = - 1

      candles_pct_list = []
      candles_pct_list.append(candle_pct)

    elif current_candles_criteria > 0:

      if candle_pct > min_pct_value:

        current_candles_criteria += 1
        candles_pct_list.append(candle_pct)
      
      if candle_pct < - min_pct_value:

        current_candles_criteria = - 1
        candles_pct_list = []
        candles_pct_list.append(candle_pct)

    else:
      
      if candle_pct < - min_pct_value:
        
        current_candles_criteria -= 1
        candles_pct_list.append(candle_pct)

      if candle_pct > min_pct_value:

        current_candles_criteria = 1
        candles_pct_list = []
        candles_pct_list.append(candle_pct)

    mean = np.mean (candles[:i])

    if i > 10:

      rise_value  = float( (candles[i] / min (candles [i - 10 : i]) - 1) * 100 )

      if i == len(candles) - 1:

        print(f'rise value is {rise_value:.2f}')

    else:

      rise_value = 0

    if current_candles_criteria >= 3 \
      and max(candles_pct_list) > 0.65 \
      and sum(candles_pct_list) > 1.2 \
      and candles[i] > mean:
      # and candles_pct_list[- 1] < candles_pct_list[- 2]:
      
      current_candles_criteria = 0
      candles_pct_list = []

      time_short.append(time[i])
      points_short.append(candles[i])

      if i == len(candles) - 1:
          
        signal_short = 1

    if current_candles_criteria <= - 3 \
      and min(candles_pct_list) < - 0.65 \
      and sum(candles_pct_list) < - 1.2 \
      and candles[i] < mean \
      and not(permission_to_long):
      # and candles_pct_list[- 1] > candles_pct_list[- 2]:
      # and candles_pct_list[- 1] > candles_pct_list[- 3]:

      permission_to_long = 1
      signal_to_long = 0
      
    if permission_to_long:

      # if candle_pct > 0:
      #   signal_to_long += 1
      
      # if signal_to_long == 2:

      if candle_pct > 0.1 and candle_pct < 0.25 \
        and rise_value < 0.25:   

        time_long.append(time[i])
        points_long.append(candles[i])
        current_candles_criteria = 0
        candles_pct_list = []
        permission_to_long = 0

        if i == len(candles) - 1:
          
          signal_buy = 1
  
  return signal_buy, signal_short
  
  # return time_long, points_long, time_short, points_short

def impulse_function(f):

  f = np.array(f)
  # f = 100 * (f - min(f)) / (max(f) - min(f))

  imp = []
  imp.append(0)
  gist_1 = 1.0008  # 1.0008 
  gist_2 = 1.01  # 1.01
  relaxation_pct_1 = 0.1 # 0.5 / 100
  relaxation_pct_2 = 0.3  # 3 / 100
  base_line = f[0]
  rise = 1

  current_imp = 0

  for i in range(1, len(f)):

    if rise:

      if f[i] / base_line >= gist_1:

        current_imp += f[i] / base_line
        base_line = f[i]
      
      elif f[i] / base_line > 2 - gist_1:

        current_imp -= current_imp * relaxation_pct_1

      elif f[i] / base_line > 2 - gist_2:
        
        current_imp -= current_imp * relaxation_pct_2
      
      else:

        current_imp = 0   
        rise = 0
        base_line = f[i]
    
    else:
    
      if base_line / f [i]  >= gist_1:
        
        current_imp -= base_line /  f[i]
        base_line = f[i]
      
      elif base_line / f[i] > 2 - gist_1:
        current_imp += current_imp * relaxation_pct_1

      elif base_line / f[i] > 2 - gist_2:

        current_imp += current_imp * relaxation_pct_2
      
      else:

        current_imp = 0   
        rise = 1
        base_line = f[i]

    # gist_1 = min (0.1, current_imp * 1 / 100)
    # gist_2 = max (1, current_imp * 5 / 100 )
    imp.append(current_imp) 

  return imp, f

def three_impulse_criteria(f, time):

  signal_short = 0

  f = np.array(f)
  mean = np.mean(f)

  green_line = mean + 2 * np.std(f)

  current_number_rise_impulse = 0
  previous_impulse_value_list = [0, 0, 0]
  short_points_list = []
  time_points_list = []
  h_i = 1

  for i in range(h_i + 1, len(f)):

    if f[i - h_i] < green_line:

      if f[i - h_i] > mean \
        and f[i - h_i] > f[i - h_i - 1] \
        and f[i - h_i] > f[i]:
        
        if f[i - h_i] > previous_impulse_value_list[- 1]:
          current_number_rise_impulse += 1
          previous_impulse_value_list.append(f[i - h_i])        
        else:
          current_number_rise_impulse = 1 

    else:
      
      if f[i - h_i] > f[i - h_i - 1] \
        and f[i - h_i] > f[i - h_i + 1]:
        
        if f[i - h_i] > previous_impulse_value_list[- 1]:
          current_number_rise_impulse += 1  
          previous_impulse_value_list.append(f[i - h_i])               
        else:
          current_number_rise_impulse = 1 

        if current_number_rise_impulse > 2 \
          or (f[i - h_i] > mean +  5 * np.std(f) and f[i] < f[i - h_i]) :
                  
          short_points_list.append (f[i - h_i])
          time_points_list.append(time[i - h_i])

          if i == len(f) - 1:
          
            signal_short = 1
        
    if f[i - h_i] < mean or f[i - h_i] < previous_impulse_value_list[- 3]:

      current_number_rise_impulse = 0
      previous_impulse_value_list.append(0) 

  return signal_short, short_points_list, time_points_list

class heiken_ashi():

  def __init__(self, o, h, l, c, pct):
  
    # common parameters
    self.o = o
    self.h = h
    self.l = l
    self.c = c
    self.pct = pct

    if self.c > self.o and self.pct > 0.01:
      self.colour = 'green'
    else:
      self.colur = 'red'

def get_heiken_ashi_v2(df):
  df['ha_c'] = (df['open'] + df['close'] + df['high'] + df['low']) / 4
  df[0]['ha_o'] = df[0]['open']
  
  for i in range(1, df.shape[0]):
    df[i]['ha_o'] =  (df[i - 1]['ha_o'] + df[i - 1]['ha_c']) / 2
  
  return df


def get_heiken_ashi(candles):

  heiken_ashi_list = []

  ha_o = float (candles[0].o)
  ha_c = float (candles[0].c)
  ha_h = float (candles[0].h)
  ha_l = float (candles[0].l)

  if candles[0].o < candles[0].c:
      pct = float( (candles[0].c / candles[0].o - 1) * 100 )
  else:
      pct = - float( (candles[0].o / candles[0].c - 1) * 100 )

  heiken_ashi_candel = heiken_ashi(ha_o, ha_h, ha_l ,ha_c, pct)
  heiken_ashi_list.append(heiken_ashi_candel)
  
  for i in range(1, len(candles)):

    ha_c = float ((candles[i].o + candles[i].c + candles[i].h + candles[i].l) / 4)
    ha_o = float (( heiken_ashi_list[-1].o + heiken_ashi_list[-1].c ) / 2 )
    ha_h = float (max(candles[i].h, ha_o, ha_c))
    ha_l = float (min(candles[i].l, ha_o, ha_c))

    if ha_c > heiken_ashi_list[-1].c:
      pct = float( (ha_c / heiken_ashi_list[-1].c - 1) * 100 )
    else:
      pct = - float( (heiken_ashi_list[-1].c / ha_c - 1) * 100)
 
    heiken_ashi_candel = heiken_ashi(ha_o, ha_h, ha_l ,ha_c, pct)
    heiken_ashi_list.append(heiken_ashi_candel)

  return heiken_ashi_list

def heiken_ashi_short_criteria(ha_candles, candles, time, close_price_hour_list):

  min_pct_value = 0.01
  min_pct_value_reset = 0.01
  pct_short_value = 0.25
  number_consecutive_blue_candles = 0
  sum_consecutive_blue_candles = 0

  close_list  = []
  ha_close_list = []

  time_short = []
  points_short = []
  signal_short = 0

  for i in range(len(ha_candles)):
    ha_close_list.append(ha_candles[i].c)
    close_list.append(candles[i].c)

  for i in range(len(ha_close_list)):

    if ha_candles[i].c > ha_candles[i].o \
      and ha_candles[i].pct > 0.02: # candel is blue     

      number_consecutive_blue_candles += 1
      sum_consecutive_blue_candles += ha_candles[i].pct
    
    time_to_maximum = 10
    if i < time_to_maximum or  number_consecutive_blue_candles  + 3 >= time_to_maximum :
      cond_near_to_prev_max = True
    else:
      if i - time_to_maximum < i - number_consecutive_blue_candles  - 3:
        prev_max = max(close_list[i - time_to_maximum : i - number_consecutive_blue_candles  - 3])
      else:
        prev_max = max(close_list[i - number_consecutive_blue_candles  - 3 : i - time_to_maximum] )

      cond_near_to_prev_max = (close_list[i] / prev_max - 1) * 100 > 0.02

      # cond_near_to_prev_max = ha_close_list[i] > max(ha_close_list[i - time_to_maximum : i])
    
    if sum_consecutive_blue_candles - ha_candles[i].pct > pct_short_value \
      and (
        number_consecutive_blue_candles >= 12 \
        or (number_consecutive_blue_candles >= 7 and close_price_hour_list[-1] < close_price_hour_list[-2] and close_price_hour_list[-1] < close_price_hour_list[-3])
          )\
      and (ha_close_list[i - 1] / ha_close_list[i] - 1) * 100 > 0.03 \
      and cond_near_to_prev_max:
      # and grad(ha_close_list[:i]) < 0 \
      #  and ha_close_list[i] < ha_close_list[i - 1] \

      # points_short.append(ha_close_list[i])
      points_short.append(close_list[i - 1])
      time_short.append(time[i])

      if i == len(candles) - 1:
        
        signal_short = 1
        price_for_short = close_list[-2]
  
      number_consecutive_blue_candles = 0
      sum_consecutive_blue_candles = 0
    
    if ha_candles[i].c < ha_candles[i].o:

      if ha_candles[i].pct >= - 0.01:

         number_consecutive_blue_candles -= 1
         sum_consecutive_blue_candles += ha_candles[i].pct
      
      else:

        number_consecutive_blue_candles -= 2
        sum_consecutive_blue_candles += ha_candles[i].pct

    if number_consecutive_blue_candles < 0:

        number_consecutive_blue_candles = 0
        sum_consecutive_blue_candles = 0

    # if ha_candles[i].c < ha_candles[i].o \
    #    and ha_candles[i].pct < - min_pct_value_reset: # candel is red and pct less than min_pct_value

    #   number_consecutive_blue_candles = 0
    #   sum_consecutive_blue_candles = 0
  
  return signal_short, points_short, time_short


def heiken_ashi_short_long_criteria(ha_candles, candles, time, close_price_hour_list, debug_mode = 0):

  current_time = datetime.now()   

  min_pct_value = 0.01
  min_pct_value_reset = 0.01
  pct_short_long_value = 0.25
  pct_for_divergation = 0.5
  divergation = 0
  number_consecutive_red_candles = 0
  sum_consecutive_red_candles = 0
  price_for_short_long = 0
  price_for_short = 0

  close_list  = []
  ha_close_list = []

  time_short_long = []
  time_short = []
  points_short_long = []
  points_short = []

  sum_last_180_candles_list = []
  time_180_candles_list = []
  velocity_180_candles_list = [- 1]
  sum_last_60_candles_list = []
  time_60_candles_list = []
  velocity_60_candles_list = [- 1]

  sum_last_15_candles_list = []

  signal_short_long = 0
  signal_short = 0

  point_short_long_check = 1
  num_profit_short_long = 0
  num_loss_short_long = 0

  point_short_check = 1
  num_profit_short = 0
  num_loss_short = 0

  sum_last_180_candles = 0
  sum_last_60_candles = 0
  sum_last_15_candles = 0

  for i in range(len(ha_candles)):
    ha_close_list.append(ha_candles[i].c)
    close_list.append(candles[i].c)

    if i < 180:        

          sum_last_180_candles +=  ha_candles[i].pct

    else:

          sum_last_180_candles +=  ha_candles[i].pct - ha_candles[i - 180].pct 

    sum_last_180_candles_list.append(sum_last_180_candles)

    if i < 60:        

          sum_last_60_candles +=  ha_candles[i].pct

    else:

          sum_last_60_candles +=  ha_candles[i].pct - ha_candles[i - 60].pct 

    sum_last_60_candles_list.append(sum_last_60_candles)

    if i < 15:        

          sum_last_15_candles +=  ha_candles[i].pct

    else:

          sum_last_15_candles +=  ha_candles[i].pct - ha_candles[i - 15].pct 

    sum_last_15_candles_list.append(sum_last_15_candles)
  
  velocity_180_candles_list = get_velocity (SMA (sum_last_180_candles_list, 7))

    # if len (sum_last_180_candles_list) > 10:
    #   velocity_180_candles_list.append (get_velocity (SMA (sum_last_180_candles_list[-10:], 7))[ - 1])

  mean = np.mean (close_list)

  if __name__ == 'functions' and debug_mode:
      from_ = 2
  else:
      from_ = len(ha_close_list) - 30

  for i in range(from_, len(ha_close_list)):

    if not (point_short_long_check):

      if (float (close_list[i]) / points_short_long[- 1] - 1) * 100 > 0.12:

        num_profit_short_long += 1
        point_short_long_check = 1
      
      if (points_short_long[- 1] / float( close_list[i])  - 1) * 100 > 0.3:

        num_loss_short_long += 1
        point_short_long_check = 1

        if __name__ == 'functions':
          print(f'number loss point short long is {len(points_short_long)}')
    
    if not (point_short_check):

      if (points_short[- 1] / float (close_list[i])- 1) * 100 > 0.12:

        num_profit_short += 1
        point_short_check = 1
      
      if ( float( close_list[i]) / points_short[- 1] - 1) * 100 > 0.3:

        num_loss_short += 1
        point_short_check = 1

        if __name__ == 'functions':
          print(f'number loss point short is {len(points_short)}')

#region

    # last_candels_close_list = []

    # if i > 60:

    #   for j in range(i, i - 60, - 1):        
                 
    #       last_candels_close_list.append( ha_candles[j].c)
    
    #   if (ha_candles[i].c / min(last_candels_close_list) - 1) * 100 > 1:

    #     pct_short_long_value = 0.5

    #   else: 

    #     pct_short_long_value = 0.25

    # sum_last_180_candles = 0
    # if i > 180:
      
    #   for j in range(i, i - 180, - 1):    

    #       sum_last_180_candles +=  ha_candles[j].pct
      
    #   sum_last_180_candles_list.append(sum_last_180_candles)
    #   time_180_candles_list.append(time[i])


    #   if len (sum_last_180_candles_list) > 10:
    #     velocity_180_candles_list.append (get_velocity (SMA (sum_last_180_candles_list[-10:], 7))[ - 1])
    
    # sum_last_60_candles = 0
    # if i > 60:
      
    #   for j in range(i, i - 60, - 1):    

    #       sum_last_60_candles +=  ha_candles[j].pct
      
    #   sum_last_60_candles_list.append(sum_last_60_candles)
    #   time_60_candles_list.append(time[i])

      # if len (sum_last_60_candles_list) > 10:
      #   velocity_60_candles_list.append (get_velocity (SMA (sum_last_60_candles_list[-10:], 7))[ - 1])
    

    # sum_last_15_candles = 0
    # if i > 15:
      
    #   for j in range(i, i - 15, - 1):    

    #       sum_last_15_candles +=  ha_candles[j].pct
      
    #   sum_last_15_candles_list.append(sum_last_15_candles)

    # if i > 100:

    #   # downfall_speed = (max (close_list[i - 30 : i]) / close_list[i] - 1) * 100 
    #   j = close_list[i - 100 : i].index(max(close_list[i - 100 : i]))
    #   downfall_speed  = (max (close_list[i - 100 : i]) / close_list[i] - 1) * 100 / (j + 1)

    # else:

    #   downfall_speed = 0 
#end region

    if i > 10:

      # downfall_speed = (max (close_list[i - 30 : i]) / close_list[i] - 1) * 100 
      j = close_list[i - 10 : i].index(min(close_list[i - 10 : i]))
      rise_value  = float( (close_list[i] / min (close_list[i - 10 : i]) - 1) * 100 )

    else:

      rise_value = ha_candles[i].pct 
                 
    if ha_candles[i].c < ha_candles[i].o \
      and ha_candles[i].pct <  - min_pct_value: # candel is red     

      number_consecutive_red_candles += 1
      sum_consecutive_red_candles += ha_candles[i].pct
    
    cond_1 = sum_consecutive_red_candles + rise_value < - pct_short_long_value \
              and number_consecutive_red_candles > 3

    cond_is_rise_1 = ( ha_close_list[i - 1] / ha_close_list[i - 2] - 1) * 100 > 0.03 and \
      ha_close_list[i] >= ha_close_list[i - 1] \
        and close_list[i] > close_list [i - 1] 

    # cond_is_rise_3 = (ha_close_list[i] / ha_close_list[i - 1] - 1) * 100 > - 0.01
    cond_is_rise_3 = rise_value > 0.03 and close_list[i] > close_list [i - 1] 
    
    green_candels_sum = 0

    if i > 20:
      for j in range(i, i - 20, - 1):
        
        if ha_candles[j].c > ha_candles[j].o:
          
          green_candels_sum += ha_candles[j].pct
        
        else:

          break

    cond_is_rise_2 = green_candels_sum > 0.05 and close_list[i] > close_list [i - 1] 

    if divergation == 1 \
      and cond_1 \
      and (cond_is_rise_1 or cond_is_rise_2):

      divergation = 2

    # if sum_consecutive_red_candles < - 2 * pct_short_long_value \
    #    and divergation == 0 \
    #    and ha_candles[i - 2].pct >= 0.03 \
    #    and ha_candles[i - 2].c > ha_candles[i - 2].o:

    #   divergation = 1
    #   number_consecutive_red_candles = 0
    #   sum_consecutive_red_candles = 0

    # if i > 180:
    #   mean = np.mean (close_list[i -180 : i])
    # else:

    signal_short_long_2, signal_short_2 = two_anomalies_criteria(time_60_candles_list, sum_last_60_candles_list, time_180_candles_list, sum_last_180_candles_list, i)

    if i  > 180:
      perm_cond_1 = not( sum_last_15_candles_list[i] < sum_last_60_candles_list[i]
                    and sum_last_60_candles_list[i] < sum_last_180_candles_list[i] ) \
                    and sum_last_15_candles_list[i] > -0.45 \
                    and f_range(sum_last_60_candles_list[i], -1 , 0) \
                    and (f_range(sum_last_180_candles_list[i], - 1.4, -0.2)
                          or sum_last_180_candles_list[i] > 0.3 )\
                    and rise_value < 0.12            
    else:
      perm_cond_1 = True

    if point_short_long_check \
      and close_list[i] < mean \
      and rise_value < 0.07 \
      and ( (cond_1 \
      and (cond_is_rise_1 or cond_is_rise_2 or cond_is_rise_3) \
      and (divergation == 0 or divergation == 2) \
      and velocity_180_candles_list[i] > 0 \
      and perm_cond_1) or signal_short_long_2 ):
    
      points_short_long.append(float(close_list[i]) )
      time_short_long.append(time[i])
      point_short_long_check = 0

      if __name__ == 'functions' and False:
        print(f'Sum_last_180_candels is {sum_last_180_candles_list[i]}, sum_last_60_candels is {sum_last_60_candles_list[i]}, sum_last_15_candels is {sum_last_15_candles_list[i]}, rise_value is {rise_value}')
      # print(f'downfall_speed is {downfall_speed}, rise_value is {rise_value}')

      # print(sum_last_180_candles)

      if i == len(candles) - 1:
        
        signal_short_long = 1
        price_for_short_long = close_list[i]


        message = f'short long cond is active \n'
        write_log(os.path.join('torgach_logs','short_long_cond'), f'short_long_cond.txt', f'{current_time}: ' + message)
  
      number_consecutive_red_candles = 0
      sum_consecutive_red_candles = 0
      divergation = 0

    if point_short_check \
      and close_list[i] > mean \
      and rise_value > 0.3 \
      and signal_short_2:

      points_short.append(float(close_list[i]) )
      time_short.append(time[i])
      point_short_check = 0

      if i == len(candles) - 1:
        
        signal_short = 1
        price_for_short = close_list[i]

      if __name__ == 'functions':
        print(f'Sum_last_180_candels is {sum_last_180_candles_list[i]}, sum_last_60_candels is {sum_last_60_candles_list[i]}, sum_last_15_candels is {sum_last_15_candles_list[i]}, rise_value is {rise_value}')

    if ha_candles[i - 2].c > ha_candles[i - 2].o:
      # and not(cond_1):

      if ha_candles[i - 2].pct >= 0.03 or green_candels_sum >= 0.03:

         number_consecutive_red_candles = 0
         sum_consecutive_red_candles = 0
         divergation = 0
    
    if i == len(candles) - 1:
      break

  if __name__ == 'functions' and False:
    print(f'Sum_last_180_candels is {sum_last_180_candles}, sum_last_60_candels is {sum_last_60_candles}, cond_1 is {cond_1}, cond_is_rise_1 is {cond_is_rise_1}, cond_is_rise_2 is {cond_is_rise_2}, divergation is {divergation}')
    col.log(f'velocity_180_candles_list[ - 1] is {velocity_180_candles_list[ - 1]:.2f}')

    print(f'Number profit shorts long is {num_profit_short_long}, number loss shorts is {num_loss_short_long}')
    print(f'Number profit shorts is {num_profit_short}, number loss shorts is {num_loss_short}')

  return signal_short_long, price_for_short_long, points_short_long, time_short_long, time_60_candles_list, get_velocity(sum_last_60_candles_list) , time_180_candles_list, get_velocity(sum_last_180_candles_list),\
    signal_short, price_for_short, points_short, time_short


def two_anomalies_criteria(time_60_candles_list, sum_last_60_candles_list, time_180_candles_list, sum_last_180_candles_list, i):

  velocity_60_candles_list = get_velocity(sum_last_60_candles_list)
  velocity_180_candles_list = get_velocity(sum_last_180_candles_list)

  signal_short_long = 0
  signal_short = 0

  f180 = np.array(velocity_180_candles_list)
  f60 = np.array(velocity_60_candles_list)

  buy_line_180 = np.mean(f180) - 2 * np.std(f180)
  buy_line_60 = np.mean(f60) - 2 * np.std(f60)

  short_line_180 = np.mean(f180) + 2 * np.std(f180)
  short_line_60 = np.mean(f60) + 2 * np.std(f60)

  if velocity_180_candles_list[i] < buy_line_180 \
    and velocity_60_candles_list[i] < buy_line_60:

    signal_short_long = 1

  if velocity_180_candles_list[i] > short_line_180 \
    and velocity_60_candles_list[i] > short_line_60:

    signal_short = 1
  
  return signal_short_long, signal_short
        
        
def get_velocity(list):

  velocity_list = [0]

  for i in range(1, len(list)):

    velocity_list.append( list[i] - list [i - 1])

  return velocity_list