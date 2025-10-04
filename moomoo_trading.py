#%% IMPORT
import yfinance as yf

from datetime import datetime, timedelta, tzinfo
from datetime import date, timezone
import pandas as pd
pd.options.mode.chained_assignment = None 

from personal_settings import personal_settings as ps

import os, pathlib
import numpy as np
import pickle
import functions_2 as f2
from tradingInterface import TradeInterface
from tqdm import tqdm
import time
import moomoo as ft
from moomoo_api import Moomoo_API
import global_variables as gv
import init_parameters as init
import winsound
import math
import sql_db
import pytz
import shutil
import traceback
from typing import Union, Tuple
import logging
from tinkoff_api import Tinkoff_API
ta = Tinkoff_API()

logging.basicConfig(filename='MA50_MA5_sell_info.log', level=logging.INFO)
logger = logging.getLogger(__name__)


import warnings
warnings.filterwarnings('error')
warnings.filterwarnings(action="ignore", message="unclosed", category=ResourceWarning)

from colog.colog import colog
c = colog()
warning = colog(TextColor='purple')
alarm = colog(TextColor='red')
blue = colog(TextColor='blue')

from currency_converter import CurrencyConverter
cr = CurrencyConverter()
rate = cr.convert(1, 'AUD', 'USD')
tzinfo_ny = pytz.timezone('America/New_York')
#%% SETTINGS

# Trade settings 
default_buy_sum = 2500 # 3300 # in USD
min_buy_sum = 1000 # 2000 # in USD
max_buy_sum = 2500 # 3300 # in  USD
stop_trading_profit_value = -200 * rate # in AUD * rate = USD
max_stock_price = 1360 # 1050 # in  USD

order_1m_life_time_min = 1
order_1h_life_time_min = 1
order_before_market_open_life_time_min = 1440 #720 #1440
order_MA50_MA5_life_time_min = 7#7
order_MA5_MA120_DS_life_time_min = 15
place_trailing_stop_limit_order_imidiately = True

# Moomoo settings
moomoo_ps = ps.Moomoo()
ip = '127.0.0.1'
port = 11111
unlock_pwd = moomoo_ps.unlock_pwd
ACC_ID = moomoo_ps.acc_id
TRD_ENV = ft.TrdEnv.REAL
MARKET = 'US.'
# ft.SysConfig.set_all_thread_daemon(True)

# init parameters
stock_name_list  = init.stock_name_list
stock_name_list_opt = init.stock_name_list_opt
exclude_time_dist = {}

# settings for historical df from yfinance
period = '3mo'
interval = '1h'
prepost_1h = False
prepost_1m = False

# settings for buy condition Version 3.0
# is_near_global_max_prt = 120  
last_top_ratio = 1
RIV  = 0.15
buy_ratio_border = 0
bull_trend_coef = 1.07
number_tries_to_submit_order = {}

# orders settings
lose_coef_1m = 0.995
lose_coef_1h = 0.995
trail_spread_coef = 0.0003
trailing_stop_limit_act_coef = 1.002
aux_price_coef = 1.0005

# other settings
use_market_direction_condition = False

# All parameter should be False. Change to True if you need change\fix DB
load_from_csv = False
load_from_xslx = True
clean_placed_orders = False
clean_cancelled_orders = True
read_sql_from_df = False
test_trading = False
skip_market_direction_calc = True
buy_when_market_closed = True
# should be False for real trading:
test_buy_sim = False
override_time_is_correct = False

#%% FUNCTIONS and CLASSES

class Default():
    def __init__(self):
        self.gain_coef = 1.005
        self.gain_coef_speed_norm100 = 1.05
        self.gain_coef_before_market_open = 1.12
        self.gain_coef_MA50_MA5 = 1.12
        self.gain_coef_MA5_MA120_DS = 1.12
        self.lose_coef_before_market_open = 0.98
        self.lose_coef_1_MA50_MA5 = 0.98
        self.lose_coef_2_MA50_MA5 = 0.97
        self.lose_coef_MA5_MA120_DS = 0.98
        self.lose_coef_stopmarket_MA50_MA5 = 0.978 # !!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.lose_coef_stopmarket_MA5_MA120_DS = 0.994 # !!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.trailing_ratio = 0.45
        self.trailing_ratio_MA50_MA5 = 1.25
        self.trailing_ratio_MA5_MA120_DS = 1.51
default = Default()

class NumberCalls ():
    def __init__(self, type_=''):
        self.type = type_
        self.value= 0  
        self.time_ref = datetime.now()
        self.calls_history = pd.Series()
# yf_calls_per_minute = NumberCalls(type_='min')
# yf_calls_per_hours= NumberCalls(type_='hour')
yf_numbercalls = NumberCalls()

def number_function_calls(func):
    def wrapper(*args, **kwargs):
        global yf_calls_per_minute
        global yf_calls_per_hours
        result = func(*args, **kwargs)
        if yf_numbercalls.calls_history.empty:
            yf_numbercalls.calls_history = pd.Series([datetime.now()])
        else:
            yf_numbercalls.calls_history = pd.concat([yf_numbercalls.calls_history, pd.Series([datetime.now()])])
        time_cond = yf_numbercalls.calls_history > datetime.now() - timedelta(seconds=60)
        calls_per_minute = yf_numbercalls.calls_history.loc[time_cond]
        yf_numbercalls.value = calls_per_minute.count()
        # number_calls_per_minute = calls_per_minute.count()
        # if yf_calls_per_minute.time_ref > datetime.now() - timedelta(seconds=60):
        #   yf_calls_per_minute.value += 1
        # else:
        #   yf_calls_per_minute = 0
        #   yf_calls_per_minute.time_ref = datetime.now()
        # print(f'Number calls per minute is {number_calls_per_minute}')         
        return result
    return wrapper

def timedelta_minutes(order):
    delta = (datetime.now() - order['buy_time']) 
    deltatime_minutes = delta.seconds / 60 + delta.days * 24 * 60
    return deltatime_minutes

def augmentate_historical_df(df): 
    df['pct'] = np.where(
        df['open'] < df['close'],
        (df['close'] / df['open'] - 1) * 100,
        -(df['open'] / df['close'] - 1) * 100
    )
    df['chg'] = (df['close'] / df['close'].shift(1) - 1) * 100
    df = f2.get_heiken_ashi_v2(df)
    df = df[['open', 'high', 'low', 'close', 'pct', 'ha_pct', 'ha_colour', 'volume', 'chg']]
    df = MA(df, k=5) # add MA5 column to the df
    df = MA(df, k=7) # add MA7 column to the df
    df = MA(df, k=10) # add MA10 column to the df
    df = MA(df, k=20) # add MA20 column to the df
    df = MA(df, k=50) # add MA50 column to the df
    df = MA(df, k=120) # add MA120 column to the df
    df['MACD'], df['MACD_hist'] = MACD(df, fast=25, slow=50, signal=14, column='close') # add MACD columns to the df
    df['MA30_RSI10'] = MA(RSI(df, period=10, column='close'), k=30) # add RSI columns to the df
    df['VR'], df['VRMA'] = calculate_volume_ratio(df, window=26, MA_window=6) # add VR and VRMA columns to the df
    return df

@number_function_calls
def get_historical_df(ticker='', interval='1h', period='2y', start_date=date.today(), end_date=date.today(), prepost=False) -> pd.DataFrame:
    df = pd.DataFrame()
    try:
        insturument = yf.Ticker(ticker)
        try:
            df = insturument.history(period=period, interval=interval, prepost=prepost)
        except Exception as e:
            alarm.print(traceback.format_exc())
            time.sleep(15)          
        if df.empty:
            try:
                df = ta.get_minutes_candles(ticker, days=1)
            except Exception as e:
                alarm.print(traceback.format_exc())
        # last_price = ta.get_last_candle_close_price(ticker)      
        if not df.empty:
            df = df.rename(columns={"Open": "open", "Close" :'close', "High" : 'high', "Low": 'low', "Volume": 'volume'})
            # c.print(f'{ticker} yfinance last price is {df['close'].iloc[-1]}')
            # c.print(f'{ticker} tinkoff last price is {df_ta['close'].iloc[-1]}')
            df = augmentate_historical_df(df)
    except Exception as e:
        alarm.print(traceback.format_exc())
    return df 

def get_prediction_df(df, prediction=1.005):
    try:
        if not df.empty:
            df_pred = df[['open', 'high', 'low', 'close', 'volume']]
            pred_raw = pd.DataFrame({"open": df_pred['close'].iloc[-1],
                                    "high": df_pred['close'].iloc[-1] * prediction,
                                    "low": df_pred['close'].iloc[-1], 
                                    "close": df_pred['close'].iloc[-1] * prediction,
                                    "volume": df_pred['volume'].iloc[-1]     
                                    },
                                    index = [df_pred.index[-1] + timedelta(minutes=1.7)]
            )
            df_pred = pd.concat([df_pred, pred_raw])
            df_pred = augmentate_historical_df(df_pred)
            return df_pred
        else:
            return pd.DataFrame()    
    except Exception as e:
        alarm.print(traceback.format_exc())
        return pd.DataFrame()
    
def more_than_pct(value1, value2, pct):
    if 100 * (value1 / value2 - 1) > pct:
        return True
    else:
        return False

def is_near_local_min(df, k=30, max_dist=15):
    '''
    Returns True if the local minimum in the last k points is within max_dist ticks from the end,
    but not in the last 3 ticks.
    '''
    result = False
    # Safety check: ensure df has enough points
    if len(df) < k:
        return False
    local_min_index = np.argmin(df.iloc[-k:])  # index in the window, 0 is oldest, k-1 is newest
    # The position from the end: k - local_min_index
    # So, local_min_index > k - max_dist means it's within max_dist from the end
    # local_min_index < k - 3 means not too close to the end
    if local_min_index > k - max_dist and local_min_index < k - 3:
        result = True
    return result
 
def is_near_local_min_5points(df):
    return df.iloc[-1] > df.iloc[-2] < df.iloc[-3] < df.iloc[-4] < df.iloc[-5]

def is_near_global_max(df, k=150, close_dist=100):
    result = False
    if len(df) < k:
        return False
    global_max_index = np.argmax(df.iloc[-k:]) # index in the window, 0 is oldest, k-1 is newest
    if global_max_index > k - close_dist:
        result = True
    return result
  
def is_near_global_max_prt(df, i, k=400, prt=70):
    result = False
    try:
        if i > k:
            gmax = max(df['close'].iloc[i - k: i])
            gmin = min(df['close'].iloc[i - k: i])
            reference_point = df['close'].iloc[i - k]
        else:
            reference_point = df['close'].iloc[0]
            gmax = max(df['close'].iloc[0: i])
            gmin = min(df['close'].iloc[0: i])
        result = 100 * (df['close'].iloc[i] - gmin) / (gmax - gmin) > prt
    except Exception as e:
        alarm.print(traceback.format_exc())
    return result

def prt_from_local_min(df, k=30):
    '''
    Return percentage from local minimum to current price
    '''
    i = df.shape[0] - 1
    if i < k:
        local_min = df['close'].iloc[0 : k].min()
    else:
        local_min = df['close'].iloc[- k:].min()
    prt = 100 * (df['close'].iloc[i] - local_min) / (local_min + 0.0001)
    return prt

def calculate_volume_ratio(df,  window=26, MA_window=6) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate Volume Ratio (VR) indicator.

    Parameters:
    df (pd.DataFrame): DataFrame with 'Close' and 'Volume' columns.
    close_col (str): Column name for closing prices.
    volume_col (str): Column name for volume.
    window (int): Rolling window size.

    Returns:
    pd.Series: Volume Ratio values.
    """
    # Calculate price change direction
    df['diff'] = df['close'].diff()
    # Classify volume as up or down
    df['up_volume'] = df.apply(lambda row: row['volume'] if row['diff'] > 0 else 0, axis=1)
    df['down_volume'] = df.apply(lambda row: row['volume'] if row['diff'] < 0 else 0, axis=1)
    df['const_volume'] = df.apply(lambda row: row['volume'] if row['diff'] == 0 else 0, axis=1)

    # Rolling sums
    up_sum = df['up_volume'].rolling(window=window).sum()
    down_sum = df['down_volume'].rolling(window=window).sum()
    const_volume = df['const_volume'].rolling(window=window).sum()

    # Avoid division by zero
    df['VR'] = 100 * (up_sum * 2 + const_volume) / (down_sum * 2 + const_volume + 1e-10)
    df['VRMA'] = MA(df['VR'], k=MA_window)

    return df['VR'], df['VRMA']

def number_red_candles(df, i, k=11):
    if i < k:
        number_red_candles = (df['ha_colour'][0 : i] == 'red').sum()
    else:
        number_red_candles = (df['ha_colour'][i - k : i] == 'red').sum()
    return number_red_candles

def minmax(value, min, max):
    '''
    Limit the value beetween min and max values
    '''
    return np.maximum(np.minimum(value, max), min)

def MA(df, k=20):
    if type(df) is pd.DataFrame:
        df[f'MA{k}'] = df['close'].rolling(window=k).mean()
    elif type(df) is pd.Series:
        df = df.rolling(window=k).mean()
    return df

def MACD(df, fast=12, slow=26, signal=9, column='close'):
    """
    Calculate MACD, Signal line, and MACD Histogram.
    Args:
        df (pd.DataFrame): DataFrame with price data.
        fast (int): Fast EMA period.
        slow (int): Slow EMA period.
        signal (int): Signal line EMA period.
        column (str): Column to use for calculation (default 'close').
    Returns:
        pd.DataFrame: DataFrame with 'MACD', 'MACD_signal', and 'MACD_hist' columns.
    """
    exp1 = df[column].ewm(span=fast, adjust=False).mean()
    exp2 = df[column].ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = (macd - macd_signal ) * 2
    df['MACD'] = macd
    df['MACD_signal'] = macd_signal
    df['MACD_hist'] = macd_hist
    return macd, macd_hist
  
def RSI(df, period=14, column='close'):
  
    delta = df[column].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def maximum(df, i, k=20):
    '''
    Return maximum price from close/open values, with k window, including last value
    '''
    if i < k:
        return np.amax([df['close'].iloc[0 : k].max(), df['open'].iloc[0 :k].max(), df['close'].iloc[i], df['open'].iloc[i]])
    else: 
        return np.amax([df['close'].iloc[i -k : i].max(), df['open'].iloc[i - k : i].max(), df['close'].iloc[i], df['open'].iloc[i]])

def minimum(df, i, k=20):  
    '''
    Return minumum price from close/open values, with k window, including last value
    '''
    if i < k:
        return np.amin([df['close'].iloc[0 : k].min(), df['open'].iloc[0 : k].min(), df['close'].iloc[i], df['open'].iloc[i]])
    else:
        return np.amin([df['close'].iloc[i - k : i].min(), df['open'].iloc[i - k : i].min(), df['close'].iloc[i], df['open'].iloc[i]])

def norm(df, i, k=20):
    min = minimum(df,i,k)
    max = maximum(df,i,k)
    return (df['close'].iloc[-1] - min) / (max - min + 0.0001)

def get_amps(df, p=7):
    min = []
    max = []
    amp = []
    amps_avg = []
    for i in range(df.shape[0]):
        if i < p:
            min.append(minimum(df, i, p))
            max.append(maximum(df, i, p))
        else:
            min.append(minimum(df, i, p))
            max.append(maximum(df, i, p))
        amp.append(max[i] / min[i])
        if i < 300:
            amps_avg.append(sum(amp[:i+1]) / len(amp[:i+1]))
        else:
            amps_avg.append(sum(amp[i - 300 : i + 1]) / len(amp[i - 300 : i + 1]))
    return amps_avg

def get_speed(df, p):
    speed100 = []
    for i in range(df.shape[0]):
        if i < p:
            speed100.append(df['close'].iloc[i] / df['close'].iloc[0])
        else:
            speed100.append(df['close'].iloc[i] / df['close'].iloc[i - p])
    return speed100

def stock_buy_conditions(df, df_1m, ticker):
    '''
        Parameters:

        Returns:
        condition
    '''
    # change order to limit and correct buy price!!! 

    condition = False
    buy_price = 0
    condition_type = 'No one'
        
    range_ = range(df.shape[0] - 200, df.shape[0])

    i = df.shape[0] - 1
    i_1m = df_1m.shape[0] - 1

    cond_1 = df['pct'].iloc[-1] > 0.2
    cond_2 = df['close'].iloc[-1] > df['close'].iloc[-2]
    cond_3 = df_1m['close'].iloc[-40 : -1].max() / df_1m['close'].iloc[-1] > 1.01
    cond_4 = df_1m['close'].iloc[-25 : -1].max() / df_1m['close'].iloc[-1] > 1.007
    cond_5 = df_1m['close'].iloc[-15 : -1].max() / df_1m['close'].iloc[-1] > 1.003

    ha_cond = df_1m['ha_colour'].iloc[-1] == 'green' and df_1m['ha_colour'].iloc[-2] == 'green' \
        and  df_1m['ha_colour'].iloc[-3] == 'red' and df_1m['ha_colour'].iloc[-4] == 'red'
    

    cond_6 = df['pct'].iloc[-2] > 0.25
    cond_7 = df['pct'].iloc[-1] > 0.05 

    if df.index[-1].hour == 9:
        # c.green_red_print(cond_1, 'cond_1')
        # c.green_red_print(cond_2, 'cond_2')
        if cond_1 and cond_2 and ha_cond \
            and (cond_3 or cond_4 or cond_5):
            condition = True      
            condition_type = 'drowdown930'
            c.green_red_print(condition, 'buy condition drowdown930')
    else:
        c.green_red_print(cond_6, 'cond_6')
        c.green_red_print(cond_7, 'cond_7')
        if cond_6 and cond_7 and ha_cond \
            and (cond_3 or cond_4 or cond_5):
            condition = True
            condition_type = 'drowdown'
            c.green_red_print(condition, 'buy condition drowdown')

    c2 = False # bad condition
    c32 = False # bad condition
    # 930 and 1030 red candles put conditon with trying to get 0.5%? time beetween 11-30 and 11-40
    if df.index[-1].hour == 11 and df.index[-1].minute == 30:
        # -1 : 11:30,  -2 : 10:30, -3 : 9:30
        CM930 = df['close'].iloc[-3] 
        CM1030 = (df['close'].iloc[-2] + df['open'].iloc[-2]) / 2
        CM1130 = df['close'].iloc[-1]

        # 930 candle is red; and 1030 candle between -1% and 0.2% ?
        c2 = more_than_pct(CM1130, CM930, 0.21) and more_than_pct(CM1130, CM1030, 0.1) \
        and df['open'].iloc[-3] / df['close'].iloc[-3]  > 1.01 \
        and df['close'].iloc[-2] / df['close'].iloc[-1] < 1.002 \
        and df['close'].iloc[-2] / df['close'].iloc[-1] > 0.99 \
        and ha_cond and False
        c.green_red_print(more_than_pct(CM1130, CM1030, 0.01), 'more_than_pct(CM1130, CM1030, 0.01)')
        c.green_red_print(more_than_pct(CM930, CM1030, 0.01), 'more_than_pct(CM930, CM1030, 0.01)')
        c.green_red_print(more_than_pct(CM1130, CM930, 0.02), 'more_than_pct(CM1130, CM930, 0.02)')
        # and more_than_pct(CM930, CM1030, 0.01)
        c32 = more_than_pct(CM1130, CM1030, 0.01) and more_than_pct(CM1130, CM930, 0.02) \
        and more_than_pct(CM1030, CM930, 0.01)\
        and df_1m['ha_colour'].iloc[-1] == 'green' and df_1m['ha_colour'].iloc[-2] == 'green' \
        and df_1m.index[-1].minute < 42 and False
        
        c.print(f'CM930 is {CM930:.2f}, CM1030 is {CM1030:.2f}, CM1130 is {CM1130:.2f}', color='blue')
        c.green_red_print(ha_cond, 'ha_cond')

    if c2:
        condition = True
        condition_type = '1130_1'

    if c32:
        condition = True
        condition_type = '1130_2'

    # c32 = False
    # if df.index[i].hour == 12 and df.index[i].minute == 30:
    #   # i : 12hours, i - 1 : 11hours, i - 2: 10hours, i - 3 : 9hours
    #   CM930 = df['close'].iloc[i - 3]
    #   CM1030 = (df['close'].iloc[i - 2] + df['open'].iloc[i - 2]) / 2
    #   CM1130 = (df['close'].iloc[i - 1] + df['open'].iloc[i - 1]) / 2
    #   c32 = more_than_pct(CM1130, CM1030, 0.01) and more_than_pct(CM930, CM1030, 0.01) and more_than_pct(CM1130, CM930, 0.4) # local minumum (1.6)
    
    c42 = False
    if df.index[i].hour == 12 and df.index[i].minute == 30:
        # i : 11hours, i - 1 : 10hours, i - 2: 9hours, i - 3 : 15hours
        CM930 = df['close'].iloc[i - 3] 
        CM1030 = (df['close'].iloc[i - 2] + df['open'].iloc[i - 2]) / 2
        CM1130 = (df['close'].iloc[i - 1] + df['open'].iloc[i - 1]) / 2
        CM1230 = df['close'].iloc[i]
        c42 = more_than_pct(CM1230, CM1130, 0.01) and more_than_pct(CM930, CM1030, 0.01) and more_than_pct(CM1230, CM930, 0.4) \
        and df_1m['close'].iloc[i_1m - 30 : i_1m].max() / df_1m['close'].iloc[i_1m] < 1.002 \
        and ha_cond
        c.print(f'CM930 is {CM930:.2f}, CM1030 is {CM1030:.2f}, CM1130 is {CM1130:.2f}', color='blue')

    if c42:
        condition = True
        condition_type = '1230_2'
  
    if False:
        if (df_1m.index[-1].hour == 12 and df_1m.index[-1].minute > 50) \
        or (df_1m.index[-1].hour == 13 and df_1m.index[-1].minute < 20):        
            max = np.maximum(df_1m['close'].iloc[i_1m - 30 : i_1m].max(), df_1m['close'].iloc[-1])

        cond_8 = df_1m['close'].iloc[-1] / df_1m['close'].iloc[-25] > 1.001
        cond_9 = df_1m['pct'].iloc[-25:-1].sum() > 0
        c.green_red_print(cond_8, 'cond_8')
        c.green_red_print(cond_9, 'cond_9')

        if cond_8 and cond_9 and ha_cond:
            condition = True
            condition_type = '1230_1330_inertia'

    if (df_1m.index[-1].hour == 12 and df_1m.index[-1].minute >= 20):
        min = minimum(df_1m, 220)
        max = maximum(df_1m, 220)
        cond_8 = df_1m['close'].iloc[-1] / min > 1.0025
        cond_9 = (df_1m['close'].iloc[-1] - min) / (max - min + 0.0001) < 0.5
        c.green_red_print(cond_8, 'cond_8')
        c.green_red_print(cond_9, 'cond_9')
        c.print(f'''df_1m['close'].iloc[-1] / min is {df_1m['close'].iloc[-1] / min:.4f}''', color='yellow')
        if cond_8 and cond_9 and ha_cond:
            condition = True
            condition_type = '1220_1300_reverse_point'
    return condition, condition_type  

def stock_buy_condition_maxminnorm(df, df_1m, df_stats, display=False):

    i_1m = df_1m.shape[0] - 1
    i = df.shape[0] - 1
    ha_cond = df_1m['ha_colour'].iloc[-1] == 'green' and df_1m['ha_colour'].iloc[-2] == 'green' \
        and number_red_candles(df_1m, i_1m, k=11) > 5

    condition = False
    
    cv = df['close'].iloc[-2]
    min30  = minimum(df, i, 30)
    min100  = minimum(df, i, 100)
    max30  = maximum(df, i, 30)
    max100  = maximum(df, i, 100)
    # current_amp30 = max30 / min30
    current_amp100 = max100 / min100

    speed30 = df['close'].iloc[-2] / df['close'].iloc[-30]
    speed100 = df['close'].iloc[-2] / df['close'].iloc[-100]

    # norm30 = (current_amp30 - 1) / (df_stats[ticker]['amps_avg30'].iloc[i - 1] - 1)
    norm100 = (current_amp100 - 1) / (df_stats[ticker]['amps_avg100'].iloc[-1] - 1)

    cond_value_3 = max30 / cv 

    cond_1 = speed30 > 1 and speed100 < 1
    cond_2 = norm100 >= 1.7 and norm100 <= 2
    cond_3 = cond_value_3 <= 1.03
  
  
    if display:
        # warning.print(f'speed30: {speed30:.3f}, speed100: {speed100:.3f}, max30/cv: {max30/cv:.3f}, norm100: {norm100:.3f}')
        warning.print('maxminnorm parameters:')
        c.green_red_print(cond_1, f'cond_1, speed30: {speed30:.2f}, speed100: {speed100:.2f}')
        c.green_red_print(cond_2, f'cond_2, norm100: {norm100:.2f}')
        c.green_red_print(cond_3, f'max30/cv {cond_value_3:.2f}')
    
    if ha_cond \
        and cond_1 \
        and cond_2 \
        and cond_3 \
        and not (df.index[-1].hour in [15, 16]):
        condition = True
        
    if condition:
        c.green_red_print(condition, 'buy_condition_maxminnorm')
    return condition

def stats_calculation(stock_name_list):

  df_stats = {}
  current_hour = datetime.now().hour
  current_date = str(datetime.now().today()).split(' ')[0]
  file_name = f'stats/df_stats_{current_date}_{current_hour}H.plk'
  try:
    if not os.path.isfile(file_name):
      print('Statistic calculation')
      for ticker in tqdm(stock_name_list):
        # file_name = f'df_stock_{ticker}_period2y_interval1h.pkl'
        # path = os.path.join(parent_path, folder_saved_models, file_name)

        df = get_historical_df(ticker = ticker, period=period, interval=interval)
        if not df.empty:
          speed30 = get_speed(df, 30)
          speed100 = get_speed(df, 100)
          amps_avg30 = get_amps(df, 30)
          amps_avg100 = get_amps(df, 100)
          data =  {'amps_avg30': amps_avg30,
                  'amps_avg100': amps_avg100,
                  'speed30': speed30,
                  'speed100': speed100}
          # if not df.empty:
          df_stats[ticker] = pd.DataFrame(data, index=df.index)
          df_stats[ticker]['max100'] = df['close'].rolling(window=100).max()
          df_stats[ticker]['max100'] = df_stats[ticker]['max100'].bfill()
          df_stats[ticker]['min100'] = df['close'].rolling(window=100).min()
          df_stats[ticker]['min100'] = df_stats[ticker]['min100'].bfill()
          df_stats[ticker]['current_amp100'] = df_stats[ticker]['max100'] / df_stats[ticker]['min100']
          df_stats[ticker]['speed100_avg30'] = df_stats[ticker]['speed100'].rolling(window=30).mean()
          df_stats[ticker]['speed30_avg30'] = df_stats[ticker]['speed30'].rolling(window=30).mean()
          df_stats[ticker]['speed30_avg30'] = df_stats[ticker]['speed30_avg30'].bfill()
          df_stats[ticker]['speed100_avg30'] = df_stats[ticker]['speed100_avg30'].bfill()
          df_stats[ticker]['norm100'] = 0.0001
          index = df_stats[ticker]['amps_avg100'].index[-2]
          df_stats[ticker].loc[index, 'norm100'] = (df_stats[ticker]['current_amp100'].iloc[-2]- 1) / (df_stats[ticker]['amps_avg100'].iloc[- 301 : -2].mean() - 1)
          index = df_stats[ticker]['amps_avg100'].index[-1]
          df_stats[ticker].loc[index, 'norm100'] = (df_stats[ticker]['current_amp100'].iloc[-1]- 1) / (df_stats[ticker]['amps_avg100'].iloc[- 300 : -1].mean() - 1)

      with open(file_name, 'wb') as file:
        pickle.dump(df_stats, file)
    else:
      with open(file_name, 'rb') as file:
        df_stats = pickle.load(file)
  except Exception as e:
    alarm.print(traceback.format_exc())
  return df_stats

def stock_buy_condition_MA50_MA5(df, df_pred, df_1m, df_stats, ticker, display=False):
    '''
    buy condition:  
    buy_price: to have MA5[-1] > MA5[-2]: \\
            df['close'].iloc[-1] should more than df['close'].iloc[-6]\\ 
            and below this value on 0.5%
    sell_orders: 2 stop_limit loss orders, trailing order with modification \\ 
    order life time: 1hour \\
    profit value: trailing order with modification \\ 
    lose value:  -1% and -2% limit, and stop market if below -2.2%\\
    buy price: df_1m['close'].iloc[-1] + 0.05% during normal hours
    after-hours: NOT BUY
    '''
    global market_value, total_market_value ,total_market_value_2m , \
        total_market_value_5m, total_market_value_30m, total_market_value_60m, \
        total_market_direction_60m, total_market_direction_10m
    
    condition = False
    condition_type = 'MA50_MA5'

    i_1m = df_1m.shape[0] - 1
    i = df.shape[0] - 1
    
    ha_cond_1m = df_1m['ha_colour'].iloc[-1] == 'green' and df_1m['ha_colour'].iloc[-2] == 'green' \
        and number_red_candles(df_1m, i_1m, k=11) >= 5
    
    #conditions 
                
    MACD_hist = df['MACD_hist'].iloc[-1]
    
    cond_RSI = df['MA30_RSI10'].iloc[-1] >= 29 \
            and df['MA30_RSI10'].iloc[-1] <= 80 \
            and df['MA30_RSI10'].iloc[-1] >= df['MA30_RSI10'].iloc[-2]
    cond_RSI_pred = df_pred['MA30_RSI10'].iloc[-1] >= 29 \
            and df_pred['MA30_RSI10'].iloc[-1] <= 80 \
            and df_pred['MA30_RSI10'].iloc[-1] >= df_pred['MA30_RSI10'].iloc[-2]

    cond_value_RSI = df['MA30_RSI10'].iloc[-1]
    
    cond_grad_MACD = df['MACD'].iloc[-1] >= df['MACD'].iloc[-2] \
            and df['MACD'].iloc[-2] >= df['MACD'].iloc[-3]
    cond_grad_MACD_pred = df_pred['MACD'].iloc[-1] >= df_pred['MACD'].iloc[-2] \
            and df_pred['MACD'].iloc[-2] >= df_pred['MACD'].iloc[-3]
    
    cond_grad_MACD_hist = df['MACD_hist'].iloc[-1] > df['MACD_hist'].iloc[-2]
    MACD = df['MACD'].iloc[-1]
  
    # 1 min MACD gradient should be positive
    cond_grad_MACD_1m = df_1m['MACD'].iloc[-1] >= df_1m['MACD'].iloc[-2] \
            and df_1m['MACD'].iloc[-2] >= df_1m['MACD'].iloc[-3] \
            and df_1m['MACD'].iloc[-3] >= df_1m['MACD'].iloc[-4]
    MACD_1m = df_1m['MACD'].iloc[-1]
                    
    MACD_hist_speed = df['MACD_hist'].iloc[-1] / df['MACD_hist'].iloc[-2]
    if df['MACD_hist'].iloc[-1] < 0 or df['MACD_hist'].iloc[-2] < 0:
        cond_MACD_hist_speed = True
    else:
        cond_MACD_hist_speed = MACD_hist_speed > 1.07
        
    delta_MA50_MA120_1m = (df_1m['MA50'] / df_1m['MA120'] - 1) * 100
    delta_MA5_MA20_1m = (df_1m['MA5'] / df_1m['MA20'] - 1) * 100
    MA5_delta_MA5_MA20_1m = MA(delta_MA5_MA20_1m, k=5)
  
    # delta_MA50_M120_1m_120 gradient should be positive
    # and delta_MA50_M120_1m_120 should be more than 0.08 
    # or negative and sum delta_MA50_M120_1m_120 last 120 minutes should be negative
    cond_MA50_M120_1m_120 = delta_MA50_MA120_1m.iloc[-1] > delta_MA50_MA120_1m.iloc[-2] \
                and delta_MA50_MA120_1m.iloc[-2] > delta_MA50_MA120_1m.iloc[-3] \
                and (delta_MA50_MA120_1m.iloc[-1]  > 0.08 
                    or delta_MA50_MA120_1m.iloc[-120:].sum() < 0
                    or delta_MA50_MA120_1m.iloc[-1]  < -0.02 ) \
                and delta_MA50_MA120_1m.iloc[-1] < 0.4
        
    delta_MA50_M120 = (df['MA50'] / df['MA120'] - 1) * 100
    delta_MA5_M20 = (df['MA5'] / df['MA20'] - 1) * 100
    delta_MA5_M10 = (df['MA5'] / df['MA10'] - 1) * 100
    cond_delta_MA5_M10 = delta_MA5_M10.iloc[-1] > 0.05  \
                    and delta_MA5_M10.iloc[-1] < 0.5             
    delta_MA5_M10_pred = (df_pred['MA5'] / df_pred['MA10'] - 1) * 100
    cond_delta_MA5_M10_pred = delta_MA5_M10_pred.iloc[-1] > 0.05  \
                    and delta_MA5_M10_pred.iloc[-1] < 0.5    

    # new from 31/08/2025
    cond_delta_MA50_M120 = delta_MA50_M120.iloc[-1] > delta_MA50_M120.iloc[-2] \
                and delta_MA50_M120.iloc[-2] >= delta_MA50_M120.iloc[-3]
    
    cond_grad_delta_MA5_M20 = delta_MA5_M20.iloc[-1] > delta_MA5_M20.iloc[-2]
    cond_grad_delta_MA5_M10 = delta_MA5_M10.iloc[-1] > delta_MA5_M10.iloc[-2]
    cond_grad_delta_MA5_M10_pred = delta_MA5_M10_pred.iloc[-1] > delta_MA5_M10_pred.iloc[-2]
    
    # b1: MA50M120Diff cross zero (15min window) and grad MACD > 0 and MACD > 0
    # b2: MACD cross zero (15min window) and grad MA50M120Diff > 0 and MA50M120Diff > 0
    cond_grad_MA50_M120_1m_120 = delta_MA50_MA120_1m.iloc[-1] > delta_MA50_MA120_1m.iloc[-2] \
                and delta_MA50_MA120_1m.iloc[-2] > delta_MA50_MA120_1m.iloc[-3]
    
    #cond_b*
    cond_b1 = delta_MA50_MA120_1m.iloc[-1] > 0 \
                and (delta_MA50_MA120_1m.iloc[-25:-2] < 0).any() \
                and MACD_1m > 0.001
    cond_b2 = df_1m['MACD'].iloc[-1] > 0 \
                and (df_1m['MACD'].iloc[-25:-2] < 0).any() \
                # and delta_MA50_MA120_1m.iloc[-1] > 0
            
    is_near_global_max_MA50 = is_near_global_max(df_1m['MA50'])
    is_near_local_min_MA50 = is_near_local_min(df_1m['MA50'])
    cond_b3 = is_near_local_min_MA50 and not is_near_global_max_MA50

    is_near_global_max_MA20 = is_near_global_max(df_1m['MA20'])
    is_near_local_min_MA20 = is_near_local_min(df_1m['MA20'])
    cond_b4 = is_near_local_min_MA20 and not is_near_global_max_MA20
        
    cond_b5 = delta_MA5_MA20_1m.iloc[-1] > 0 \
                and (delta_MA5_MA20_1m.iloc[-25:-2] < 0).any() \
                and df_1m['MA20'].iloc[-1] > df_1m['MA20'].iloc[-2]

    cond_b6 = (df['ha_colour'].iloc[-3:-1] == 'red').any() \
        and df['ha_colour'].iloc[-1] == 'green'
                        
    # no big spikes in last 8 hours
    spikes_1 = (df['pct'].iloc[-8:] < 1.0071).all()
    # spikes_2 = ((df['close'] / df['close'].shift(1))[-8:] < 1.012).all()
    spikes_2 = (df['chg'].iloc[-3:] < 1.0071).all()
    spikes_3_value = df_1m['close'].iloc[-1] / minimum(stock_df, i, k=3) 
    spikes_3 = spikes_3_value < 1.0071
    no_big_spikes_1h = spikes_1 and spikes_2 and spikes_3 
  
    MA5_1h_is_near_local_min = is_near_local_min_5points(df['MA5'])
    permit_1h_cond = MA5_1h_is_near_local_min and \
        df['MA20'].iloc[-1] > df['MA20'].iloc[-2] 
        
    cond_grad_MA5 = df['MA5'].iloc[-1] > df['MA5'].iloc[-2]
    cond_grad_MA5_pred = df_pred['MA5'].iloc[-1] > df_pred['MA5'].iloc[-2]
    cond_grad_MA10 = df['MA10'].iloc[-1] > df['MA10'].iloc[-2]
    cond_grad_MA10_pred = df_pred['MA10'].iloc[-1] > df_pred['MA10'].iloc[-2]
    cond_grad_MA20 = df['MA20'].iloc[-1] > df['MA20'].iloc[-2]
    
    # one of the 3 1hour last heiken ashi candles is red
    ha_cond_1h = (df['ha_colour'].iloc[-4:-1] == 'red').any() \
        and df['ha_colour'].iloc[-1] == 'green'
    ha_cond_1h_pred = (df_pred['ha_colour'].iloc[-4:-1] == 'red').any() \
        and df_pred['ha_colour'].iloc[-1] == 'green'
    
    MA20Speed5 = (df['MA20'].iloc[-1] / df['MA20'].iloc[-6] - 1) * 1000 # range could be approx in (-30; +30)
    MA5Speed2_1m = (df['MA5'] / df['MA5'].shift(2) - 1) * 1000 # range could be approx in (-30; +30)
    cond_MA5Speed2_1m = MA5Speed2_1m.iloc[-1] > MA5Speed2_1m.iloc[-2] \
                    and MA5Speed2_1m.iloc[-2] >= MA5Speed2_1m.iloc[-3] \
    
    cond_RSI_1m = df_1m['MA30_RSI10'].iloc[-1] > df_1m['MA30_RSI10'].iloc[-2] \
                and df_1m['MA30_RSI10'].iloc[-2] >= df_1m['MA30_RSI10'].iloc[-3] \
                
    cond_grad_MA5_1m = df_1m['MA5'].iloc[-1] > df_1m['MA5'].iloc[-2]
    cond_grad_MA20_1m = df_1m['MA20'].iloc[-1] > df_1m['MA20'].iloc[-2]
    cond_grad_MA50_1m  = df_1m['MA50'].iloc[-1] > df_1m['MA50'].iloc[-2] \
            and df_1m['MA50'].iloc[-2] >= df_1m['MA50'].iloc[-3] \
    
    cond_VR_1m = df_1m['VR'].iloc[-1] < 120 \
        and df_1m['VRMA'].iloc[-1] < 120
        
    cond_MA5_delta_MA5_MA20_1m = MA5_delta_MA5_MA20_1m.iloc[-1] < 0.1 \
        and MA5_delta_MA5_MA20_1m.iloc[-1] > MA5_delta_MA5_MA20_1m.iloc[-2] 
    
    if  (cond_RSI or cond_RSI_pred) \
        and (cond_grad_MACD or cond_grad_MACD_pred) \
        and (cond_grad_MA10 or cond_grad_MA10_pred) \
        and (cond_grad_MA5 or cond_grad_MA5_pred) \
        and (cond_grad_delta_MA5_M10 or cond_grad_delta_MA5_M10_pred) \
        and (ha_cond_1h \
            or ha_cond_1h_pred
            or cond_delta_MA5_M10 \
            or cond_delta_MA5_M10_pred \
            ) \
        and cond_grad_delta_MA5_M20 \
        and no_big_spikes_1h \
        and cond_grad_MA50_1m \
        and cond_grad_MACD_1m \
        and cond_grad_MA5_1m \
        and cond_grad_MA20_1m \
        and cond_RSI_1m \
        and cond_VR_1m \
        and cond_MA5_delta_MA5_MA20_1m:
        condition = True    

        # important removed from 4/10/2025:
        # and (cond_b1 or cond_b2 or cond_b3 or cond_b4 or cond_b5 or cond_b6) \
        
        # removed from 1/10/2025:
        # and cond_grad_MA50_M120_1m_120 \
        # and (cond_MACD_hist_speed or permit_1h_cond or cond_b6) \
        # and cond_grad_MACD_hist \
        
        # and cond_delta_MA50_M120 \    # removed from 26/09/2025
        # and (cond_RSI or permit_1h_cond or cond_b6) \
        # cond_delta_MA50_M120 
        # cond_delta_MA5_M20 and cond_grad_MACD
        
        #  and ha_cond: # excluded from 21/09/2025
        
        
        # and cond_MACD_hist_max_amp_1m:
        # removed from 31/08/25
        # and cond_MACD_hist_1m 
        # and (cond_MACD_1m_sum20 or cond_positive_MACD_1m or cond2_positive_MACD_1m)
        
        # and cond_RSI_1m \ removed from 18/08/25
        
        # removed from 12/09/2025:
            # and (cond_sum_delta_MA50_MA120_1m or time_cond) \ 
      
    if df['MA30_RSI10'].iloc[-1]  < 25 \
        or (df['MACD'].iloc[-2] < df['MACD'].iloc[-3] \
        and df['MACD'].iloc[-3] < df['MACD'].iloc[-4] ):
        
        
        # or (delta_MA50_M120.iloc[-2] < delta_MA50_M120.iloc[-3] \
        #   and delta_MA50_M120.iloc[-3] < delta_MA50_M120.iloc[-4]):
        
            # or df['MACD_hist'].iloc[-2] < df['MACD_hist'].iloc[-3] \ # removeed from 15/09/2025
        # exclude company from _list optimal list
        warning.print(f'{ticker} is excluding from current optimal stock list')
        exclude_time_dist[ticker] = datetime.now()
        if ticker in stock_name_list_opt:
            stock_name_list_opt.remove(ticker)
            warning.print(f'Optimal stock name list len is {len(stock_name_list_opt)}')      
            market_value = -1 # start offset
            total_market_value = -99  
            # Calculate market direction
            total_market_value_2m = 0
            total_market_value_5m = 0
            total_market_value_30m = 0
            total_market_value_60m = 0
            total_market_direction_10m = 0
            total_market_direction_60m = 0
     
    if display:
        warning.print('MA50_MA5 conditions parameters:')
        c.print('AND CONDITIONS:', color='yellow')
        c.green_red_print(cond_grad_MACD_hist, f'cond_grad_MACD_hist, {df['MACD_hist'].iloc[-1]:.3f} > {df['MACD_hist'].iloc[-2]:.3f}')
        c.green_red_print(cond_grad_MACD, f'cond_grad_MACD, {df["MACD"].iloc[-1]:.3f} > {df["MACD"].iloc[-2]:.3f}')
        c.green_red_print(cond_grad_MA20, f'cond_grad_MA20, {df["MA20"].iloc[-1]:.3f} > {df["MA20"].iloc[-2]:.3f}')
        c.green_red_print(cond_grad_MA5, f'cond_grad_MA5, {df["MA5"].iloc[-1]:.3f} > {df["MA5"].iloc[-2]:.3f}')
        c.green_red_print(cond_RSI, f'cond_RSI (MA30_RSI10), {df["MA30_RSI10"].iloc[-1]:.3f} > {df["MA30_RSI10"].iloc[-2]:.3f}')
        # c.green_red_print(cond_delta_MA50_M120, f'cond_delta_MA50_M120, {delta_MA50_M120.iloc[-1]:.3f}')
        c.green_red_print(cond_grad_delta_MA5_M20, f'cond_grad_delta_MA5_M20, {delta_MA5_M20.iloc[-1]:.3f} > {delta_MA5_M20.iloc[-2]:.3f}')
        c.green_red_print(no_big_spikes_1h, f'no_big_spikes_1h spike_value: {spikes_3_value:.3f} < 1.012')
        c.green_red_print(ha_cond_1h, f'ha_cond_1h, one of the 3 last 1h ha candles is red and last is green')
        c.green_red_print(cond_MACD_hist_speed, f'(MACD_hist speed more than 1.07), MACD_hist_speed: {MACD_hist_speed:.3f}, MACD_hist: {MACD_hist:.3f}')
        c.green_red_print(cond_grad_MACD_1m, f'grad_MACD_1m > 0, {df_1m["MACD"].iloc[-1]:.3f} > {df_1m["MACD"].iloc[-2]:.3f}')
        c.green_red_print(cond_grad_MA50_M120_1m_120, f'cond_grad_MA50_M120_1m_120 {delta_MA50_MA120_1m.iloc[-1]:.3f} > {delta_MA50_MA120_1m.iloc[-2]:.3f}')
        c.green_red_print(cond_grad_MA5_1m, f'cond_grad_MA5_1m {df_1m["MA5"].iloc[-1]:.3f} > {df_1m["MA5"].iloc[-2]:.3f}')
        c.green_red_print(cond_RSI_1m, f'cond_RSI_1m, {df_1m["MA30_RSI10"].iloc[-1]:.3f} > {df_1m["MA30_RSI10"].iloc[-2]:.3f} ')
        c.green_red_print(cond_VR_1m, f'cond_VR, VR: {df_1m["VR"].iloc[-1]:.3f} < 120 , VRMA: {df_1m["VRMA"].iloc[-1]:.3f} < 120')
        c.green_red_print(cond_MA5Speed2_1m, f'cond_MA5Speed2_1m, {MA5Speed2_1m.iloc[-1]} > {MA5Speed2_1m.iloc[-2]}')
        # c.green_red_print(ha_cond, f'ha_cond, last two 1m ha candles are green and more than 5 red ha candles in last 11 candles')
        # c.green_red_print(cond_sum_delta_MA50_MA120_1m, f'cond_sum_delta_MA50_MA120_1m, {delta_MA50_MA120_1m_sum180:.3f}')
        c.green_red_print(permit_1h_cond, f'permit_1h_cond, MA5_1h_is_near_local_min: {MA5_1h_is_near_local_min}, MA20_1h_grad: {df["MA20"].iloc[-1] > df["MA20"].iloc[-2]}')
        c.print('OR CONDITIONS:', color='yellow')
        c.green_red_print(cond_b1, f'cond_b1 (MA50M120Diff_1m cross zero), {delta_MA50_MA120_1m.iloc[-1]:.3f}')
        c.green_red_print(cond_b2, f'cond_b2 (MACD_1m cross zero), {df_1m['MACD'].iloc[-1]:.3f}')
        c.green_red_print(cond_b3, f'cond_b3 (MA50_1m near local minimum and not near global maximum)')
        c.green_red_print(cond_b4, f'cond_b4 (MA20_1m near local minimum and not near global maximum)')
        c.green_red_print(cond_b5, f'cond_b5 (MA5_MA20Diff_1m cross zero), {delta_MA5_MA20_1m.iloc[-1]:.3f}')
        c.green_red_print(cond_b6, f'cond_b6 (1h ha candle is red and last is green)')

    if condition: 
        c.green_red_print(condition, condition_type)
        
    conditions_info =  f'''cond_b1:{cond_b1}, cond_b2:{cond_b2}, cond_b3:{cond_b3}, cond_b4:{cond_b4}, cond_b5:{cond_b5}, cond_b6: {cond_b6}\
    cond_RSI:{int(cond_RSI)}, cond_grad_MACD:{int(cond_grad_MACD)}, cond_grad_MA5:{int(cond_grad_MA5)}\
    cond_delta_MA50_M120:{int(cond_delta_MA50_M120)}, cond_grad_MACD_1m:{int(cond_grad_MACD_1m)},\
    no_big_spikes_1h:{int(no_big_spikes_1h)}, cond_grad_MA50_M120_1m_120:{int(cond_grad_MA50_M120_1m_120)},\
    cond_grad_MA5_1m:{int(cond_grad_MA5_1m)}, cond_RSI_1m:{int(cond_RSI_1m)}, cond_VR:{int(cond_VR_1m)}, ha_cond:{int(ha_cond_1m)},\
    permit_1h_cond:{int(permit_1h_cond)},\
    df['MACD'].iloc[-1]: {df['MACD'].iloc[-1]:.3f}, df['MACD'].iloc[-2]: {df['MACD'].iloc[-2]:.3f},\
    df['MACD_hist'].iloc[-1]: {df['MACD_hist'].iloc[-1]:.3f}, df['MACD_hist'].iloc[-2]: {df['MACD_hist'].iloc[-2]:.3f},\
    delta_MA50_M120.iloc[-1]: {delta_MA50_M120.iloc[-1]:.3f}, delta_MA50_M120.iloc[-2]: {delta_MA50_M120.iloc[-2]:.3f},\ 
    delta_MA5_M20.iloc[-1]: {delta_MA5_M20.iloc[-1]:.3f}, delta_MA5_M20.iloc[-2]: {delta_MA5_M20.iloc[-2]:.3f},\
    delta_MA5_M10.iloc[-1]: {delta_MA5_M10.iloc[-1]:.3f}, delta_MA5_M10.iloc[-2]: {delta_MA5_M10.iloc[-2]:.3f},\
    df['MA30_RSI10'].iloc[-1]: {df['MA30_RSI10'].iloc[-1]:.3f}, df['MA30_RSI10'].iloc[-2]: {df['MA30_RSI10'].iloc[-2]:.3f}\
    delta_MA50_MA120_1m.iloc[-1]: {delta_MA50_MA120_1m.iloc[-1]:.3f}, delta_MA50_MA120_1m.iloc[-2]: {delta_MA50_MA120_1m.iloc[-2]:.3f}\
    df_1m['MA30_RSI10'].iloc[-1]: {df_1m['MA30_RSI10'].iloc[-1]:.3f}, df_1m['MA30_RSI10'].iloc[-2]: {df_1m['MA30_RSI10'].iloc[-2]:.3f}\
    df_1m['MACD'].iloc[-1]: {df_1m['MACD'].iloc[-1]:.3f}, df_1m['MACD'].iloc[-2]: {df_1m['MACD'].iloc[-2]:.3f}\
    df_1m['MACD_hist'].iloc[-1]: {df_1m['MACD_hist'].iloc[-1]:.3f}, df_1m['MACD_hist'].iloc[-2]: {df_1m['MACD_hist'].iloc[-2]:.3f}\
    MACD_hist_speed: {MACD_hist_speed:.3f} (more than 1.07 is good)
    '''

    return condition, conditions_info
# version from 13/03/2025 with MA, ha_cond and 
def stock_buy_condition_before_market_open(df, df_1m, df_stats, display=False):

    condition = False
    condition_type = 'No cond'
    exclude = True
    
    new_york_time = datetime.now(pytz.timezone('US/Eastern'))
    i_1m = df_1m.shape[0] - 1
    i = df.shape[0] - 1

    MACD = df['MACD'].iloc[-1]
    
    delta_MA5_M20 = (df['MA5'] / df['MA20'] - 1) * 100
    cond_grad_delta_MA5_M20 = delta_MA5_M20.iloc[-1] > delta_MA5_M20.iloc[-2]
    cond_grad_MA10 = df['MA10'].iloc[-1] > df['MA10'].iloc[-2]
    cond_grad_MA20 = df['MA20'].iloc[-1] > df['MA20'].iloc[-2]
    
    cond_grad_MA120 = df['MA120'].iloc[-1] > df['MA120'].iloc[-2]
    cond_grad_MA20 = df['MA20'].iloc[-1] > df['MA20'].iloc[-2]
    
    # MA10_speed2 = (df['MA10'].iloc[-1] / df['MA10'].iloc[-3] - 1) * 1000 # range could be approx in (-30; +30)
    MA10_speed2 = (df['MA10'] / df['MA10'].shift(1) - 1) * 1000 # range could be approx in (-30; +30)
    # spikes_value = df['close'].iloc[-1] / minimum(stock_df, i, k=8) 
  
    # gram MACD > 0 or MACD > 0.1
    # cond_grad_MA20
    # delta_MA5_M20 > 0.3 (callucate procent from history!!!) or grad delta_MA5_M20 > 0
    # buy with price 15:30
    # if MA5 crossing MA20????
    # no high spikes in last 10 hours (1.012)
    # MA10Speed2 - key indicator; 
    
    if (MACD > 0.1 \
        or df['MACD'].iloc[-1] > df['MACD'].iloc[-2]) \
        and cond_grad_MA10 \
        and (MA10_speed2.iloc[-1] > MA10_speed2.iloc[-2] \
            and (MA10_speed2.iloc[-1] < -1 \
                    or (MA10_speed2.iloc[-1] < 2 and MA10_speed2.iloc[-2] > 0) \
                )
            ):
        exclude = False
        condition_type = 'before_market_open_1'
        condition = True
            
    # if cond_grad_MA120 \
    # and cond_grad_MA20\
    # and MACD > 0.1 \
    # and cond_grad_delta_MA5_M20:
    #   exclude = False
    #   condition_type = 'before_market_open_1'
    #   condition = True

    if display:
        warning.print('Before_market_open conditions parameters:')
        c.print('AND CONDITIONS:', color='yellow')
        c.green_red_print(cond_grad_MA120, f'cond_grad_MA120, {df["MA120"].iloc[-1]:.3f} > {df["MA120"].iloc[-2]:.3f}')
        c.green_red_print(cond_grad_MA20, f'cond_grad_MA20, {df["MA20"].iloc[-1]:.3f} > {df["MA20"].iloc[-2]:.3f}')
        c.green_red_print(MACD > 0.1, f'MACD, {MACD:.3f} > 0.1')
        c.green_red_print(cond_grad_delta_MA5_M20, f'cond_grad_delta_MA5_M20, {delta_MA5_M20.iloc[-1]:.3f} > {delta_MA5_M20.iloc[-2]:.3f}')
  
  # exclude company from _list optimal list
    if exclude and not market_time_and_1hour_before:
        warning.print(f'{ticker} is excluding from current optimal stock list')
        exclude_time_dist[ticker] = datetime.now()
        if ticker in stock_name_list_opt:
            stock_name_list_opt.remove(ticker)
            warning.print(f'Optimal stock name list len is {len(stock_name_list_opt)}')
      
    if condition: 
        c.green_red_print(condition, condition_type)
    return condition, condition_type

def sell_stock_condition(order, current_price):

    buy_price = order['buy_price']
    gain_coef = order['gain_coef']
    lose_coef = order['lose_coef']

    sell_condition = False

    if current_price / buy_price >= gain_coef:
        sell_condition = True  
    
    if current_price / buy_price <= lose_coef:
        sell_condition = True

    return sell_condition

def current_profit(df, hours=24):
    try:
        ref_time = datetime.now() - timedelta(hours=hours)
        if df.shape[0] == 0:
            current_profit = 0
        else:
            df = df[(df['sell_time'] >= ref_time)]
            current_profit = df['profit'].sum()
    except Exception as e:
        alarm.print(traceback.format_exc())
        current_profit = 10
    return float(current_profit)

def update_buy_order_based_on_platform_data(order, historical_orders = None):
    '''
    return: order
    '''
    retrieve_history_order = False
    try:
        if type(historical_orders) != pd.DataFrame:
            if historical_orders == None:
                retrieve_history_order = True
        else:
            if historical_orders.empty:
                retrieve_history_order = True
    except Exception as e:
        alarm.print(traceback.format_exc())
        retrieve_history_order = True
    
    if retrieve_history_order:
        ticker = order['ticker']
        historical_orders = ma.get_history_orders(code=ticker)
    #   order = orders.loc[(orders['trd_side'] == 'BUY') & (orders['order_status'] == 'FILLED_ALL')].sort_values('updated_time', ascending=True).iloc[-1]
    
    try:
        if type(historical_orders) == pd.DataFrame and not historical_orders.empty:
            history_order = historical_orders.loc[(historical_orders['order_id'] == order['buy_order_id'])]
            if history_order.shape[0] > 0 \
            and (history_order['order_status'].values[0] == ft.OrderStatus.FILLED_ALL 
                or history_order['order_status'].values[0] == ft.OrderStatus.CANCELLED_PART):                                
                order['buy_price'] = float(history_order['dealt_avg_price'].values[0])
                order['buy_sum'] = float(order['buy_price'] * history_order['qty'].values[0])
                buy_commission = ma.get_order_commission(order['buy_order_id'])             
                order['buy_commission'] = buy_commission
                order['stocks_number'] = int(history_order['qty'].values[0])
                order['status'] = 'bought'
                # df = ti.update_order(df, order)
                # add order['buy_time']?
                if history_order['order_status'].values[0] == ft.OrderStatus.CANCELLED_PART:
                    ma.modify_limit_if_touched_order(order, order['gain_coef'])
                    ma.modify_stop_order(order, order['lose_coef'])
    except Exception as e:
        alarm.print(traceback.format_exc())
    return order

def load_orders_from_csv():
    # FUNCTION TO UPDATE times from csv files to df with correct time format
    df = pd.read_csv('db/real_trade_db.csv', index_col='Unnamed: 0')
    df['buy_time'] = pd.to_datetime(df['buy_time'], dayfirst=False)
    df['sell_time'] = pd.to_datetime(df['sell_time'], dayfirst=False)
    gv.ORDERS_ID  = df['id'].max()
    if gv.ORDERS_ID == np.nan:
        gv.ORDERS_ID = 1
    return df

def load_orders_from_xlsx():
    try:
        date = str(datetime.now()).split(' ')[0]
        shutil.copyfile('db/real_trade_db.xlsx', f'db/bin/real_trade_db_{date}.xlsx')
    except Exception as e:
        alarm.print(traceback.format_exc())
    df = pd.read_excel('db/real_trade_db.xlsx', index_col='Unnamed: 0')
    gv.ORDERS_ID  = df['id'].max()
    if gv.ORDERS_ID == np.nan:
        gv.ORDERS_ID = 1
    return df

def get_orders_list_from_moomoo_orders(orders: pd.DataFrame):
    orders_list = []
    for index, row in orders.iterrows():
        ticker = row['code'].split('.')[1]
        orders_list.append(ticker)
    return orders_list

def isNaN(num):
    return num != num

def place_sell_order_if_it_was_not_placed(
    df, order, sell_orders, sell_orders_list,
    price, order_type, current_gain=1, low_trail_value=False
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    '''
      - type: buy | limit_if_touched | stop | trailing_LIT | trailing_stop_limit | stop_limit
    '''
    moomoo_order_id = None
    try:
        ticker = order['ticker']
        order_id_type = order_type + '_order_id'
        order_id = order[order_id_type]
        qty = order['stocks_number']
    except Exception as e:
        alarm.print(traceback.format_exc())
        order_id = None

    if (order_id in [None, '', 'FAXXXX'] 
        or isNaN(order_id)):
        # and ticker not in sell_orders_list:
        if order_type == 'limit_if_touched':
            moomoo_order_id = ma.place_limit_if_touched_order(ticker, price, qty)
        if order_type == 'stop':
            moomoo_order_id = ma.place_stop_order(ticker, price, qty)   
        if order_type == 'stop_limit':
            moomoo_order_id = ma.place_stop_limit_sell_order(ticker, price, qty)   
        if order_type == 'trailing_LIT':
            moomoo_order_id = ma.place_limit_if_touched_order(ticker, price, qty, aux_price_coef=aux_price_coef, remark='trailing_LIT')

        if order_type == 'trailing_stop_limit':
            trail_spread = order['buy_price'] * trail_spread_coef
            if order['buy_condition_type'] == '1230':
                if current_gain >= 1.005 and current_gain < 1.01:
                    trail_value = 0.15
                else:
                    trail_value = 0.3
                if (datetime.now().astimezone(tzinfo_ny).hour == 16 and datetime.now().astimezone(tzinfo_ny).minute > 55) \
                    or (datetime.now().astimezone(tzinfo_ny).hour == 17 and datetime.now().astimezone(tzinfo_ny).minute > 55) \
                    or low_trail_value:
                    trail_value = 0.02  
            elif order['buy_condition_type'] == '930-47':
                if current_gain >= 1.005 and current_gain < 1.01:
                    trail_value = 0.12
                else:
                    trail_value = default.trailing_ratio
            elif order['buy_condition_type'] == 'MA50_MA5':
                trail_value = default.trailing_ratio_MA50_MA5
            elif order['buy_condition_type'] == 'MA5_MA120_DS':
                trail_value = default.trailing_ratio_MA5_MA120_DS
            else:
                trail_value = default.trailing_ratio
            moomoo_order_id = ma.place_trailing_stop_limit_order(ticker, price, qty, trail_value=trail_value, trail_spread=trail_spread)
            order['trailing_ratio'] = trail_value

        if not (moomoo_order_id is None):
            order[order_id_type] = moomoo_order_id
            df = ti.update_order(df, order)
    else:
        if sell_orders.shape[0] > 0:
            sell_order = sell_orders.loc[sell_orders['order_id'] == order[order_id_type]]
        else:
            sell_order = pd.DataFrame()
        try:
            if sell_order.shape[0] > 0:
                if sell_order['order_status'].values[0] != ft.OrderStatus.SUBMITTED \
                    and sell_order['order_status'].values[0]  != ft.OrderStatus.SUBMITTING \
                    and sell_order['order_status'].values[0]  != ft.OrderStatus.WAITING_SUBMIT:
                    alarm.print(f'{ticker} {order_type} order has not been sumbitted')
            else:
                alarm.print(f'{ticker} {order_type} order has not been placed')
        except Exception as e:
            alarm.print(traceback.format_exc())
            alarm.print(f'{ticker} {order_type} order has not been placed')
    return df, order

def clean_cancelled_and_failed_orders_history(df, type) -> pd.DataFrame:
    try:
        historical_orders = ma.get_history_orders()
        if not historical_orders.empty:
            placed_stocks = df.loc[(df['status'] == type)]
            for index, row in placed_stocks.iterrows():
                ticker = row['ticker']
                buy_order = historical_orders.loc[historical_orders['order_id'] == row['buy_order_id']]
                if buy_order.shape[0] > 0:
                    if buy_order['order_status'].values[0] in [ft.OrderStatus.CANCELLED_ALL, ft.OrderStatus.FAILED]:
                        index = df.loc[(df['buy_order_id'] == row['buy_order_id']) & 
                                       (df['status'] == type) & (df['ticker'] == ticker)
                                      ].index
                        df.drop(index=index, inplace=True)
    except Exception as e:
        alarm.print(e)
    return df

def get_bought_and_placed_stock_list(df):
    if df.shape[0] > 0:
        bought_stocks = df.loc[(df['status'] == 'bought') | (df['status'] == 'filled part')]
        placed_stocks = df.loc[(df['status'] == 'placed')]
        bought_stocks_list = bought_stocks.ticker.to_list()
        placed_stocks_list = placed_stocks.ticker.to_list()
    else:
        bought_stocks_list = []
        placed_stocks_list = []
        bought_stocks = pd.DataFrame()
        placed_stocks = pd.DataFrame()
    return bought_stocks, placed_stocks, bought_stocks_list, placed_stocks_list

def trailing_stop_limit_order_trailing_ratio_modification(df, order, current_price) -> Tuple[pd.DataFrame, pd.DataFrame]:
    try:
        current_gain = current_price / order['buy_price']
        trail_spread = order['buy_price'] * trail_spread_coef
        trailing_ratio = order['trailing_ratio']
        
        print(f'Ticker is {order['ticker']}, current_gain is {current_gain:.4f}')
        if current_gain >= 1.003 and order['trailing_ratio'] == default.trailing_ratio:
            trailing_ratio = 0.4
        if current_gain >= 1.004 and order['trailing_ratio'] == 0.4:
            trailing_ratio = 0.3
        if current_gain >= 1.005 and order['trailing_ratio'] == 0.3:
            trailing_ratio = 0.25
        if current_gain >=  1.006 and order['trailing_ratio'] == 0.25:
            trailing_ratio = 0.2
        if current_gain >=  1.007 and order['trailing_ratio'] == 0.2:
            trailing_ratio = 0.15

        if order['trailing_ratio'] != trailing_ratio:
            order_id = ma.modify_trailing_stop_limit_order(order=order,
                                                    trail_value=trailing_ratio,
                                                    trail_spread=trail_spread)  
            if order_id != order['trailing_stop_limit_order_id']:
                    order['trailing_stop_limit_order_id'] = order_id
            order['trailing_ratio'] = trailing_ratio
            df = ti.update_order(df, order)
    except Exception as e:
        alarm.print
    return df, order

def check_buy_order_for_cancelation(df, ticker, historical_orders) -> pd.DataFrame:
  '''
   CHECKING FOR CANCELATION OF THE BUY ORDER
   Check current price and status of buy order
   If price more or equal than price*gain_coef CANCEL the BUY ORDER
   If status was not FILLED_PART, change status of order to 'cancel', update the order
  '''
  if ticker in list(set(placed_stocks_list + limit_buy_orders_list + limit_if_touched_buy_orders_list + stop_limit_buy_orders_list)):
    try:
      stock_df_1m = get_historical_df(ticker = ticker, period='max', interval='1m', prepost=prepost_1m)
      if not stock_df_1m.empty:
        current_price = stock_df_1m['close'].iloc[-1]
      else:
        current_price = 0    
      
      order = df.loc[(df['ticker'] == ticker) & df['status'].isin(['placed'])]
      if order.empty:
        order = df.loc[(df['ticker'] == ticker) & df['status'].isin([ 'cancelled'])]

      order_failed_or_cancelled = False
      if (ticker not in positions_list):
        code = 'US.' + ticker
        historical_order = historical_orders.loc[(historical_orders['code'] == code) \
                                    & (historical_orders['order_id'] == order['buy_order_id'].values[0])]
        if historical_order.shape[0] > 0:
          historical_order_status = historical_order['order_status'].values[0]
          if historical_order_status in [ft.OrderStatus.FAILED, ft.OrderStatus.CANCELLED_ALL]:
            order_failed_or_cancelled = True

      if not order.empty:
        order = order.sort_values('buy_time').iloc[-1]
        order_time = timedelta_minutes(order) 
        
        current_minute = datetime.now().astimezone().minute 
        current_hour = datetime.now().astimezone().hour
        tzinfo_ny = pytz.timezone('America/New_York')
        new_york_time = datetime.now().astimezone(tzinfo_ny)
        new_york_hour = new_york_time.hour
        new_york_minute = new_york_time.minute

        cond_1 = order['buy_condition_type'] in ['before_market_open_1', 'before_market_open_2', 'before_market_open_3'] \
                 and ((new_york_hour in [9, 10, 11, 12, 13, 14] and new_york_minute > 32) 
                  or order_time >= order_before_market_open_life_time_min)
        cond_2 = order['buy_condition_type'] == '1m' \
                 and order_time >= order_1m_life_time_min 
        cond_3 = order['buy_condition_type'] == 'MA50_MA5' \
                 and order_time >= order_MA50_MA5_life_time_min 
        cond_4 = not order['buy_condition_type'] in ['1m', 'before_market_open_1', 'before_market_open_2', 'before_market_open_3', 'MA50_MA5', 'MA5_MA120_DS'] \
                 and order_time > order_1h_life_time_min
        cond_5 = order['buy_condition_type'] == 'MA5_MA120_DS' \
                 and order_time > order_MA5_MA120_DS_life_time_min

        if current_price >= order['buy_price'] * (order['gain_coef'] + 0.015) / 0.93  \
          or cond_1 \
          or cond_2 \
          or cond_3 \
          or cond_4 \
          or cond_5 \
          or order_failed_or_cancelled:
                
          if buy_orders.shape[0] > 0:
            buy_order = limit_buy_orders.loc[
              buy_orders['order_id'] == order['buy_order_id']]
            if not buy_order.empty:
              if buy_order['order_status'].values[0] != ft.OrderStatus.FILLED_PART:
                try:
                  # cancel limit_if_touched_sell_order and stop_order if they was placed
                  if order['limit_if_touched_order_id'] not in ['', None, []]:                  
                      ma.cancel_order(order, order_type='limit_if_touched')    
                  if order['stop_order_id'] not in ['', None, []]:                  
                      ma.cancel_order(order, order_type='stop')
                  if order['trailing_stop_limit_order_id'] not in ['', None, []]:                  
                      ma.cancel_order(order, order_type='trailing_stop_limit')
                  if order['trailing_LIT_order_id'] not in ['', None, []]:                  
                      ma.cancel_order(order, order_type='trailing_LIT_order')
                except Exception as e:
                  alarm.print(traceback.format_exc())
              else:
                order['status'] = 'filled part'
            # cancel buy limit order
            ma.cancel_order(order, order_type='buy')
            order['status'] = 'cancelled'
            df = ti.update_order(df, order)
    except Exception as e:
      print(e)
  return df

def check_if_sell_orders_have_been_executed(df, ticker) -> pd.DataFrame:
  '''
  Checking if sell orders have been executed
  '''
  try:
    if ticker in bought_stocks_list:
      # order = df.loc[(df['ticker'] == ticker) & df['status'].isin(['bought', 'filled_part'])].sort_values('buy_time').iloc[-1]
      orders = df.loc[(df['ticker'] == ticker) & df['status'].isin(['bought', 'filled_part'])]
      if orders.shape[0] > 0 and not (type(historical_orders) == str):
        order = orders.sort_values('buy_time').iloc[-1]
        historical_limit_if_touched_order = historical_orders.loc[
          historical_orders['order_id'] == order['limit_if_touched_order_id']]
        historical_stop_order = historical_orders.loc[
          historical_orders['order_id'] == order['stop_order_id']]
        historical_trailing_LIT_order = historical_orders.loc[
          historical_orders['order_id'] == order['trailing_LIT_order_id']]
        historical_trailing_stop_limit_order = historical_orders.loc[
          historical_orders['order_id'] == order['trailing_stop_limit_order_id']]
        historical_stop_limit_order = historical_orders.loc[
          historical_orders['order_id'] == order['stop_limit_order_id']]
      
        # checking and update limit if touched order
        if historical_limit_if_touched_order.shape[0] > 0 \
          and  historical_limit_if_touched_order['order_status'].values[0] == ft.OrderStatus.FILLED_ALL\
          and order['status'] in ['bought', 'filled part']:
          sell_price = order['buy_price'] * order['gain_coef']
          order = ti.sell_order(order, sell_price=sell_price, historical_order = historical_limit_if_touched_order)
          df = ti.update_order(df, order)
          # play sound:
          winsound.PlaySound('SystemHand', winsound.SND_ALIAS)
          # cancel stop order
          ma.cancel_order(order, order_type='stop')
          ma.cancel_order(order, order_type='stop_limit')
          ma.cancel_order(order, order_type='trailing_LIT')
          ma.cancel_order(order, order_type='trailing_stop_limit')
      # checking and update stop order
        if historical_stop_order.shape[0] > 0 \
        and historical_stop_order['order_status'].values[0] == ft.OrderStatus.FILLED_ALL\
        and order['status'] in ['bought', 'filled part']:
          sell_price = order['buy_price'] * order['lose_coef']
          order = ti.sell_order(order, sell_price=sell_price, historical_order = historical_stop_order)
          df = ti.update_order(df, order)
          # cancel limit-if-touched order
          ma.cancel_order(order, order_type='limit_if_touched')
          ma.cancel_order(order, order_type='trailing_LIT')
          ma.cancel_order(order, order_type='trailing_stop_limit')
          ma.cancel_order(order, order_type='stop_limit')
      # checking and update trailing LIT order
        if historical_trailing_LIT_order.shape[0] > 0 \
          and  historical_trailing_LIT_order['order_status'].values[0] == ft.OrderStatus.FILLED_ALL\
          and order['status'] in ['bought', 'filled part']:
          sell_price = order['buy_price'] * order['trailing_LIT_gain_coef']
          order = ti.sell_order(order, sell_price=sell_price, historical_order = historical_trailing_LIT_order)
          df = ti.update_order(df, order)
          # play sound:
          winsound.PlaySound('SystemHand', winsound.SND_ALIAS)
          # cancel stop order and limit if touched 
          ma.cancel_order(order, order_type='stop')
          ma.cancel_order(order, order_type='limit_if_touched')
          ma.cancel_order(order, order_type='trailing_stop_limit')
          ma.cancel_order(order, order_type='stop_limit')
      # checking and update trailing stop limit order
        if historical_trailing_stop_limit_order.shape[0] > 0 \
          and  historical_trailing_stop_limit_order['order_status'].values[0] == ft.OrderStatus.FILLED_ALL\
          and order['status'] in ['bought', 'filled part']:
          sell_price = order['buy_price'] * order['trailing_LIT_gain_coef']
          order = ti.sell_order(order, sell_price=sell_price, historical_order = historical_trailing_stop_limit_order)
          df = ti.update_order(df, order)
          # play sound:
          winsound.PlaySound('SystemHand', winsound.SND_ALIAS)
          # cancel stop order and limit if touched 
          ma.cancel_order(order, order_type='stop')
          ma.cancel_order(order, order_type='limit_if_touched')
          ma.cancel_order(order, order_type='trailing_LIT')
          ma.cancel_order(order, order_type='stop_limit')
        if historical_stop_limit_order.shape[0] > 0 \
          and  historical_stop_limit_order['order_status'].values[0] == ft.OrderStatus.FILLED_ALL\
          and order['status'] in ['bought', 'filled part']:
          sell_price = order['buy_price'] * order['lose_coef']
          order = ti.sell_order(order, sell_price=sell_price, historical_order = historical_stop_limit_order)
          df = ti.update_order(df, order)
          # play sound:
          winsound.PlaySound('SystemHand', winsound.SND_ALIAS)
          # cancel stop order and limit if touched 
          ma.cancel_order(order, order_type='stop')
          ma.cancel_order(order, order_type='limit_if_touched')
          ma.cancel_order(order, order_type='trailing_LIT')
          ma.cancel_order(order, order_type='stop_limit')
  except Exception as e:
    alarm.print(traceback.format_exc())
  return df

def check_buy_order_info_from_order_history(df, ticker) -> pd.DataFrame:
    try: 
      if ticker in placed_stocks_list:
        orders = df.loc[(df['ticker'] == ticker) & df['status'].isin(['placed'])].sort_values('buy_time')
        if orders.shape[0] > 0:
          order = orders.iloc[-1]
          # 1.111 was set during buy order creation
          if order['buy_commission'] == 1.111 \
            or math.isnan(order['buy_commission']) \
            or order['buy_commission'] == None \
            or order['buy_commission'] == 0:
            order = update_buy_order_based_on_platform_data(order, historical_orders)
            if order['status'] == 'bought':
              df = ti.update_order(df, order)
    except Exception as e:
      alarm.print(traceback.format_exc())
    return df

def check_sell_order_has_been_placed(df, order, ticker, order_type) -> Tuple[pd.DataFrame, pd.DataFrame]:
  '''
  return: df, order
  '''
  sell_orders_name = order_type + '_sell_orders'
  sell_orders_list_name = order_type + '_sell_orders_list'
  try:
    sell_orders = globals()[sell_orders_name]
    sell_orders_list = globals()[sell_orders_list_name]
    if order_type == 'trailing_stop_limit':
      stock_df_1m = get_historical_df(ticker = ticker, period='max', interval='1m', prepost=prepost_1m)
      if not stock_df_1m.empty:
        current_price = stock_df_1m['close'].iloc[-1]
      else:
        current_price = 0
      if current_price >= order['buy_price'] * trailing_stop_limit_act_coef \
          or place_trailing_stop_limit_order_imidiately:
        permition = True
      else:
        permition = False
    else:
        permition = True

    if order_type in ['stop', 'stop_limit']:
      price = order['buy_price'] * order['lose_coef']
    elif order_type == 'limit_if_touched':
      price = order['buy_price'] * order['gain_coef']
    elif order_type == 'trailing_LIT':
      price = order['buy_price'] * order['trailing_LIT_gain_coef'] 
    else:
      price = order['buy_price'] * default.gain_coef

    if permition:
      df, order = place_sell_order_if_it_was_not_placed(df,
        order=order,
        sell_orders=sell_orders,
        sell_orders_list=sell_orders_list,
        price=price,
        order_type=order_type) 
      
  except Exception as e:
    alarm.print(traceback.format_exc())
  return df, order

def place_traililing_stop_limit_order_at_the_end_of_trading_day(df, order, ticker, current_price) -> Tuple[pd.DataFrame, pd.DataFrame]:
  # Place limit order if end of the trading day for 1230 buy condition
  try:
    current_gain = current_price / order['buy_price']
    print(f'Stock {ticker}, current gain is {current_gain:.4f}')

    # Modification from 18/03/2025
    place_order_cond1 = current_gain > 1.0015 \
                        and stock_df_1m['MA7'].iloc[-1] < stock_df_1m['MA7'].iloc[-2]
    place_order_cond2 = current_gain > 1.0015 \
                        and stock_df_1m['ha_colour'].iloc[-1] == 'red' \
                        and stock_df_1m['ha_colour'].iloc[-2] == 'red'
    if place_order_cond1 or place_order_cond2:
      low_trail_value = True
    else:
      low_trail_value = False

    if ( (datetime.now().astimezone(tzinfo_ny).hour >= 15 and datetime.now().astimezone(tzinfo_ny).minute > 55) \
          and current_gain > 0.998) \
            or current_gain >= 1.003 \
            or (current_gain > 0.998 and datetime.now().astimezone(tzinfo_ny).hour < 9) \
            or place_order_cond1 \
            or place_order_cond2:
      price = stock_df_1m['close'].iloc[-1]
      df, order = place_sell_order_if_it_was_not_placed(df,
        order=order,
        sell_orders=trailing_stop_limit_sell_orders,
        sell_orders_list=trailing_stop_limit_sell_orders_list,
        price=price,
        order_type='trailing_stop_limit',
        current_gain=current_gain,
        low_trail_value=low_trail_value) 
  except Exception as e:
    alarm.print(traceback.format_exc())      
  return df, order 

def modify_trailing_stop_limit_1230_order(df, order, current_price) -> Tuple[pd.DataFrame, pd.DataFrame]:
  # Modify order with buy_codition_type = 12:30
  try:
    order_id = order['trailing_stop_limit_order_id']
    current_gain = current_price / order['buy_price']
    if order['buy_condition_type'] == '1230' \
      and not(order_id in [None, '', 'FAXXXX'] 
          or isNaN(order_id)):
      trail_spread = order['buy_price'] * trail_spread_coef

      trailing_ratio = order['trailing_ratio']
      if current_gain >= 1.005:
        trailing_ratio = 0.15

      if current_gain <= 1.005 and order['trailing_ratio'] != 0.15:
        trailing_ratio = 0.3

      if ((datetime.now().astimezone(tzinfo_ny).hour == 16 and datetime.now().astimezone(tzinfo_ny).minute > 55) \
        or (datetime.now().astimezone(tzinfo_ny).hour == 17 and datetime.now().astimezone(tzinfo_ny).minute > 55)) \
          and current_gain >= 0.998:
        trailing_ratio = 0.02
  
      if order['trailing_ratio'] != trailing_ratio:
        order_id = ma.modify_trailing_stop_limit_order(order=order,
                                            trail_value=trailing_ratio,
                                            trail_spread=trail_spread)  
        if order_id != order['trailing_stop_limit_order_id']:
          order['trailing_stop_limit_order_id'] = order_id
        order['trailing_ratio'] = trailing_ratio
        df = ti.update_order(df, order)
  except Exception as e:
    alarm.print(traceback.format_exc())
  return df, order   

def modify_trailing_stop_limit_MA50_MA5_order(df, order, current_price, stock_df, stock_df_1m) -> Tuple[pd.DataFrame, pd.DataFrame]:
  try:
    i = stock_df.shape[0] - 1
    order_id = order['trailing_stop_limit_order_id']
    current_gain = current_price / order['buy_price']
    if order['buy_condition_type'] in ['MA50_MA5', 'before_market_open_1']\
      and not(order_id in [None, '', 'FAXXXX'] 
          or isNaN(order_id)):
      trail_spread = order['buy_price'] * trail_spread_coef
      trailing_ratio = order['trailing_ratio']
      
      grad_MA5 = stock_df_1m['close'].iloc[-1] >= stock_df['close'].iloc[-6] \
            and stock_df['MA5'].iloc[-2] >= stock_df['MA5'].iloc[-3]
            
      grad_MA50 = stock_df_1m['close'].iloc[-1] > stock_df['close'].iloc[-51] \
                  and stock_df['MA50'].iloc[-2] >= stock_df['MA50'].iloc[-3]
                  
      grad_MA50_prev = stock_df['MA50'].iloc[-4] <= stock_df['MA50'].iloc[-5] \
                    and stock_df['MA50'].iloc[-5] <= stock_df['MA50'].iloc[-6] \

      deltaMA5_MA50 = stock_df['MA5'].iloc[-1] / stock_df['MA50'].iloc[-1]    
      deltaMA5_MA50_b3 = stock_df['MA5'].iloc[-3] / stock_df['MA50'].iloc[-3]    
      
      deltaMA5_MA120_1m = stock_df_1m['MA5'].iloc[-1] / stock_df_1m['MA120'].iloc[-1]   
      deltaMA5_MA120_1m_b3 = stock_df_1m['MA5'].iloc[-3] / stock_df_1m['MA120'].iloc[-3]
      deltaMA5_MA50_1m = stock_df_1m['MA5'].iloc[-1] / stock_df_1m['MA50'].iloc[-1]   
      
      cond_1 = deltaMA5_MA50_b3 > 1.001  and deltaMA5_MA50 < 1
      cond_2 = not grad_MA5
      cond_3 = stock_df['MA50'].iloc[-1] <= stock_df['MA50'].iloc[-2] \
               and stock_df['MA50'].iloc[-2] <= stock_df['MA50'].iloc[-3] \
               and False
      
      # enable sell conditions if MACD grad is negative 
      # or last 1 MACD_hist grad is negative  and last 2 MA30_RSI10 is negative
      # or last 2 MACD_hist grad is negative 
      enable_sellings_cond = (stock_df['MACD'].iloc[-1] < stock_df['MACD'].iloc[-2] \
                              and stock_df['MACD'].iloc[-2] <= stock_df['MACD'].iloc[-3])\
        or (stock_df['MACD_hist'].iloc[-1] < stock_df['MACD_hist'].iloc[-2] \
          and stock_df['MA30_RSI10'].iloc[-1] <= stock_df['MA30_RSI10'].iloc[-2] \
          and stock_df['MA30_RSI10'].iloc[-2] <= stock_df['MA30_RSI10'].iloc[-3]) \
        or (stock_df['MACD_hist'].iloc[-1] < stock_df['MACD_hist'].iloc[-2] \
          and stock_df['MACD_hist'].iloc[-2] <= stock_df['MACD_hist'].iloc[-3] \
          and stock_df['MACD_hist'].iloc[-3] <= stock_df['MACD_hist'].iloc[-4]) \
        or maximum(stock_df, i, k=3) / minimum(stock_df, i, k=3) > 1.012 #??????????????????

      cond_MA5_crossingM120 = False
      # MA5 1h crossing MA120 1h condition
      if (cond_1 and cond_2) \
        or (cond_3 and cond_2) \
        or (deltaMA5_MA50 < 0.998 and not grad_MA5 and False) \
        and current_gain > 1.002 \
        and enable_sellings_cond \
        and order['trailing_ratio'] > 0.3:
        trailing_ratio = 0.3
        cond_MA5_crossingM120 = True
      
      # slow sell condition  
      cond_4 = stock_df['MACD'].iloc[-1] < 0 \
        and stock_df['MACD'].iloc[-1] < stock_df['MACD'].iloc[-2] \
        and stock_df['MACD'].iloc[-2] <= stock_df['MACD'].iloc[-3] \
        and stock_df['MACD_hist'].iloc[-1] < stock_df['MACD_hist'].iloc[-2] \
        and stock_df['MACD_hist'].iloc[-2] <= stock_df['MACD_hist'].iloc[-3]
        
      if cond_4 \
         and order['trailing_ratio'] > 0.05:
          trailing_ratio = 0.05
      
      # slow sell condition
      cond_5 = stock_df['MACD'].iloc[-1] < 0 \
        and stock_df['MACD'].iloc[-1] <= stock_df['MACD'].iloc[-2] \
        and stock_df['MACD'].iloc[-2] <= stock_df['MACD'].iloc[-3]
        
      # if grad MACD < 0 and MACD_hist[-1] < MACD_hist[-2] 
      # and MACD_hist[-2] <= MACD_hist[-3]:
      cond_6 = stock_df['MACD'].iloc[-1] < stock_df['MACD'].iloc[-2] \
        and stock_df['MACD_hist'].iloc[-1] < stock_df['MACD_hist'].iloc[-2] \
        and stock_df['MACD_hist'].iloc[-2] <= stock_df['MACD_hist'].iloc[-3]
      
      cond_7 = stock_df_1m['MACD'].iloc[-1] < 0 \
        and stock_df_1m['MACD_hist'].iloc[-1] < stock_df_1m['MACD_hist'].iloc[-2] \
        and (stock_df['MACD_hist'].iloc[-1] < stock_df['MACD_hist'].iloc[-2] \
            or stock_df['MACD_hist'].iloc[-1] / stock_df['MACD_hist'].iloc[-2] < 1.03)
      
      # almost like cond_6 but faster and looks for 1 minute MACD
      cond_8 = stock_df_1m['MACD'].iloc[-1] < 0 \
        and stock_df_1m['MACD'].iloc[-1] < stock_df_1m['MACD'].iloc[-2] \
        and stock_df_1m['MACD'].iloc[-2] <= stock_df_1m['MACD'].iloc[-3] \
        and ((stock_df['MACD'].iloc[-1] < stock_df['MACD'].iloc[-2] \
              and stock_df['MACD_hist'].iloc[-1] < stock_df['MACD_hist'].iloc[-2]
             ) \
              or stock_df['MACD_hist'].iloc[-1] / stock_df['MACD_hist'].iloc[-2] < 1.03
            )
      MA50_MA120_1m_120 = (stock_df_1m['MA50'] / stock_df_1m['MA120'] - 1) * 100
      cond_9 =  MA50_MA120_1m_120.iloc[-1] < 0 \
        and MA50_MA120_1m_120.iloc[-1] < MA50_MA120_1m_120.iloc[-2] \
        and MA50_MA120_1m_120.iloc[-2] <= MA50_MA120_1m_120.iloc[-3]
        
      cond_deltaMA5_MA50_1m = deltaMA5_MA50_1m < 0.9995 \
        and stock_df_1m['MACD'].iloc[-1] < 0 \
        and stock_df_1m['MACD_hist'].iloc[-1] < stock_df_1m['MACD_hist'].iloc[-2]
             
      #  and current_gain > 1.001: # removed from 09/07/2025
      # current_gain <=0.9985 could be cause to sell too quickly, consider to remove
      if order['buy_condition_type'] == 'MA50_MA5':
        if (cond_5 or cond_6 or cond_7 or cond_8 or cond_9 or cond_deltaMA5_MA50_1m) \
          and (enable_sellings_cond \
            or current_gain <= 0.995) \
          and (MA50_MA120_1m_120.iloc[-1] < MA50_MA120_1m_120.iloc[-2] 
              and MA50_MA120_1m_120.iloc[-2] <= MA50_MA120_1m_120.iloc[-3]) \
          and order['trailing_ratio'] > 0.05:
          trailing_ratio = 0.05
        
      enable_sellings_cond_before_market_open = stock_df['MACD'].iloc[-1] < stock_df['MACD'].iloc[-2] \
        or (stock_df['MA30_RSI10'].iloc[-1] < stock_df['MA30_RSI10'].iloc[-2]) \
        or (stock_df['MACD_hist'].iloc[-1] < stock_df['MACD_hist'].iloc[-2]) \
        or maximum(stock_df, i, k=3) / minimum(stock_df, i, k=3) > 1.012 #??????????????????
   
      if order['buy_condition_type'] == 'before_market_open_1':
        if (cond_5 or cond_6 or cond_7 or cond_8 or cond_9 or cond_deltaMA5_MA50_1m) \
          and (enable_sellings_cond_before_market_open \
            or current_gain <= 0.995) \
          and (MA50_MA120_1m_120.iloc[-1] < MA50_MA120_1m_120.iloc[-2] 
              and MA50_MA120_1m_120.iloc[-2] <= MA50_MA120_1m_120.iloc[-3]) \
          and order['trailing_ratio'] > 0.05:
          trailing_ratio = 0.05
          
      if (stock_df['chg'].iloc[-1] > 0.95 \
         or stock_df['chg'].iloc[-2] > 0.95 \
         or stock_df['pct'].iloc[-1] > 0.95 \
         or stock_df['pct'].iloc[-2] > 0.95 \
         or stock_df['close'].iloc[-1] / stock_df['open'].iloc[-1] > 1.011):
         if order['trailing_ratio'] > 0.31:
            trailing_ratio = 0.31
      else:
        if order['trailing_ratio'] == 0.31:
            trailing_ratio =  default.trailing_ratio_MA50_MA5
                      
      if stock_df['ha_colour'].iloc[-1] == 'green' \
         and stock_df['MA5'].iloc[-1] >= stock_df['MA5'].iloc[-2] \
         and order['trailing_ratio'] == 0.05 \
         and stock_df['MACD'].iloc[-1] > 0 \
         and not(cond_2):
           trailing_ratio = default.trailing_ratio_MA50_MA5
           
      # if current_gain <= 0.997 \
      #   and order['trailing_ratio'] > 0.01:
      #   trailing_ratio = 0.01
      print('--'*50)  
      blue.print(f'Modify order for stock {order['ticker']} information:')
      warning.print(f'Current gain is {current_gain:.3f}')
      warning.print(f'Current trailing ratio: {order['trailing_ratio']:.2f}, new trailing ratio: {trailing_ratio:.2f}')
      warning.print('Trailing ration 0.3 condition:') 
      warning.print(f'(cond_1: {cond_1} AND cond_2: {cond_2}) OR (cond_3: {cond_3} AND cond_4: {cond_4})')
      c.green_red_print(cond_MA5_crossingM120, 'cond_MA5_crossingM120')
      warning.print('Trailing ration 0.05 condition:') 
      c.green_red_print(cond_4, 'cond_4')
      c.green_red_print(cond_5, 'cond_5')
      c.green_red_print(cond_6, 'cond_6')
      c.green_red_print(cond_7, 'cond_7')
      c.green_red_print(cond_8, 'cond_8')
      c.green_red_print(cond_9, 'cond_9')    
      c.green_red_print(cond_deltaMA5_MA50_1m, 'cond_deltaMA5_MA50_1m')         
      print('--'*50) 
              
      if order['trailing_ratio'] != trailing_ratio:
        try:
            info = (
                f"time {datetime.now()}: Ticker {order['ticker']}, "
                f"cond1: {cond_1}, cond2: {cond_2}, cond3: {cond_3}, cond4: {cond_4}, "
                f"cond5: {cond_5}, cond6: {cond_6}, cond7: {cond_7}, cond8: {cond_8}, cond9: {cond_9}, "
                f"cond_deltaMA5_MA50_1m: {cond_deltaMA5_MA50_1m}, grad_MA5: {grad_MA5}, grad_MA50: {grad_MA50}, grad_MA50_prev: {grad_MA50_prev}, "
                f"deltaMA5_MA50: {deltaMA5_MA50:.4f}, deltaMA5_MA50_b3: {deltaMA5_MA50_b3:.4f}, trailing_ratio: {trailing_ratio:.2f}, "
                f"enable_sellings_cond: {enable_sellings_cond}, "
                f"MACD[-1]: {stock_df['MACD'].iloc[-1]:.4f}, MACD[-2]: {stock_df['MACD'].iloc[-2]:.4f}, "
                f"MACD_hist[-1]: {stock_df['MACD_hist'].iloc[-1]:.4f}, MACD_hist[-2]: {stock_df['MACD_hist'].iloc[-2]:.4f}"
                f"MACD_hist[-3]: {stock_df['MACD_hist'].iloc[-3]:.4f}, MACD_hist[-4]: {stock_df['MACD_hist'].iloc[-4]:.4f} "
                f"['MA30_RSI10'].iloc[-1]: {stock_df['MA30_RSI10'].iloc[-1]:.2f}, ['MA30_RSI10'].iloc[-2]: {stock_df['MA30_RSI10'].iloc[-2]:.2f}, "
                f"['MA30_RSI10'].iloc[-3]: {stock_df['MA30_RSI10'].iloc[-3]:.2f}"
                f"current_gain: {current_gain:.4f}"
            )
            print(info)
            logger.info(info)
        except Exception as e:
            alarm.print(traceback.format_exc())
          
        order_id = ma.modify_trailing_stop_limit_order(order=order,
                                            trail_value=trailing_ratio,
                                            trail_spread=trail_spread)  
        if order_id != order['trailing_stop_limit_order_id']:
          order['trailing_stop_limit_order_id'] = order_id
        order['trailing_ratio'] = trailing_ratio
        df = ti.update_order(df, order)
  except Exception as e:
    alarm.print(traceback.format_exc())
  return df, order   

def modify_trailing_stop_limit_MA5_MA120_DS_order(df, order, current_price, stock_df, stock_df_1m) -> Tuple[pd.DataFrame, pd.DataFrame]:
  try:
    order_id = order['trailing_stop_limit_order_id']
    current_gain = current_price / order['buy_price']
    
    deltaMA5_MA120_1m = stock_df_1m['MA5'].iloc[-1] / stock_df_1m['MA120'].iloc[-1]    
    deltaMA5_MA120_1m_b3 = stock_df_1m['MA5'].iloc[-3] / stock_df_1m['MA120'].iloc[-3]   
    deltaMA5_MA50_1m = stock_df_1m['MA5'].iloc[-1] / stock_df_1m['MA50'].iloc[-1]    
    deltaMA5_MA50_1m_b3 = stock_df_1m['MA5'].iloc[-3] / stock_df_1m['MA50'].iloc[-3]  
    
    if deltaMA5_MA50_1m < 1 and deltaMA5_MA120_1m < 1:
          df, order = check_sell_order_has_been_placed(df, order, ticker, order_type='trailing_stop_limit')
              
    if order['buy_condition_type'] == 'MA5_MA120_DS' \
      and not(order_id in [None, '', 'FAXXXX'] 
          or isNaN(order_id)):
      trail_spread = order['buy_price'] * trail_spread_coef
      trailing_ratio = order['trailing_ratio']

      # MA5 1m crossing MA120 1m 
      cond_1 = deltaMA5_MA120_1m_b3 >= 1 and deltaMA5_MA120_1m < 1
      cond_2 = deltaMA5_MA50_1m_b3 >= 1 and deltaMA5_MA50_1m < 1
        
      if (cond_1 or cond_2):
        if current_gain <= 1.0007:
          if order['trailing_ratio'] > 0.3:
            trailing_ratio = 0.3
        else:
            trailing_ratio = 0.05
    
      if deltaMA5_MA50_1m < 1 and deltaMA5_MA120_1m < 1 \
        and order['trailing_ratio'] > 0.05:
          trailing_ratio = 0.05
          
      if deltaMA5_MA120_1m >= 1.005 \
        or deltaMA5_MA50_1m >= 1.0005:
         trailing_ratio = default.trailing_ratio_MA5_MA120_DS
      
      if order['trailing_ratio'] != trailing_ratio:
        order_id = ma.modify_trailing_stop_limit_order(order=order,
                                            trail_value=trailing_ratio,
                                            trail_spread=trail_spread)  
        if order_id != order['trailing_stop_limit_order_id']:
          order['trailing_stop_limit_order_id'] = order_id
        order['trailing_ratio'] = trailing_ratio
        df = ti.update_order(df, order)
  except Exception as e:
    alarm.print(traceback.format_exc())
  return df, order   

def modify_stop_limit_before_market_open_order(df, order, current_price) -> Tuple[pd.DataFrame, pd.DataFrame]:
 
  try:
    order_id = order['stop_limit_order_id']
    current_gain = current_price / order['buy_price']

    if order['buy_condition_type'] in ['before_market_open_1', 'before_market_open_2', 'before_market_open_3'] \
      and not(order_id in [None, '', 'FAXXXX'] 
          or isNaN(order_id)):
      
      lose_coef = order['lose_coef']
      if current_gain < 1.01 and order['buy_condition_type'] == 'before_market_open_3' \
        and order['lose_coef'] < 1.003:
        lose_coef = 1.003
      if current_gain >= 1.01 and current_gain < 1.02 \
         and order['buy_condition_type'] == 'before_market_open_3':
         lose_coef = 1.005

      if current_gain >= 1.02 and current_gain < 1.03 and order['lose_coef'] < 1.0065:
        lose_coef = 1.0065
      elif current_gain >= 1.03 and current_gain < 1.04 \
        and order['lose_coef'] < 1.0075:
        lose_coef = 1.0075
      elif current_gain >= 1.04 and current_gain < 1.045 \
        and order['lose_coef'] < 1.0085:
        lose_coef = 1.0085
      elif current_gain >= 1.045 and current_gain < 1.05 \
        and order['lose_coef'] < 1.0099:
        lose_coef = 1.0099

      # Fast sold it gain more than 3%
      try:
        elapsed_time = datetime.now() - order['buy_time']
        elapsed_time_minutes = elapsed_time.days * 24 * 60 + elapsed_time.seconds / 60
        if current_gain > 1.03 and elapsed_time_minutes < 360:
          lose_coef = current_gain * 0.9999
      except Exception as e:
        alarm.print(traceback.format_exc())

      if order['lose_coef'] != lose_coef:
        order_id = ma.modify_stop_limit_order(order=order,lose_coef=lose_coef)  
        if order_id != order['stop_limit_order_id']:
          order['stop_limit_order_id'] = order_id
        order['lose_coef'] = lose_coef
        df = ti.update_order(df, order)

  except Exception as e:
    alarm.print(traceback.format_exc())
  return df, order   

def modify_buy_limit_if_touched_before_market_open_order(df, order, stock_df_1m) -> Tuple[pd.DataFrame, pd.DataFrame]:
 
  try:
    order_id = order['buy_order_id']
    # current_gain = current_price / order['buy_price']

    if order['buy_condition_type'] in ['before_market_open_1', 'before_market_open_2', 'before_market_open_3'] \
      and not(order_id in [None, '', 'FAXXXX'] 
          or isNaN(order_id)):
      
      buy_price = order['buy_price']
      
      if not market_time:
        if not market_time_and_2hour_before:
          buy_price = stock_df_1m['close'].iloc[-1] * 0.98
        elif not market_time_and_1hour_before: # between 1 hour and 2 hours before market open
          buy_price = stock_df_1m['close'].iloc[-1] * 0.99
        elif not market_time_and_30min_before: # between 30 min and 1 hour before market open
          buy_price = stock_df_1m['close'].iloc[-1] * 0.997
        else:
          buy_price = stock_df_1m['close'].iloc[-1]
    
      if buy_price / order['buy_price'] > 1.003 or buy_price / order['buy_price'] < 0.997:     
        order_id = ma.modify_limit_if_touched_order(order=order, price=buy_price, order_type='buy_order')  
        if order_id != order['buy_order_id']:
          order['buy_order_id'] = order_id
        order['buy_price'] = buy_price
        df = ti.update_order(df, order)

  except Exception as e:
    alarm.print(traceback.format_exc())
  return df, order   

def buy_price_based_on_condition(df, df_1m, condition_type):
  buy_price = 0
  try:
    if condition_type == '930-47':
      buy_price = df_1m['close'].iloc[-1]
    elif condition_type in ['drowdown930', 'drowdown', 'maxminnorm', 'speed_norm100']:
      # buy_price = df_1m['close'].iloc[-2] # ver. 1
      buy_price = minimum(df_1m, 4)
    elif condition_type in ['1130_1', '1130_2']:
      buy_price = df_1m['close'].iloc[-2]
    elif condition_type == '1230':
      buy_price = df_1m['close'].iloc[-2]
    elif condition_type in ['1230_2']: 
      buy_price = min(df_1m['close'].iloc[-4:-1].min(), df_1m['open'].iloc[-4:-1].min(), df_1m['close'].iloc[-1])
    elif condition_type == '1m':
      buy_price = min(df_1m['close'].iloc[-10:-1].min(), df_1m['open'].iloc[-10:-1].min())
    elif condition_type == '1h':
      buy_price = df['close'].iloc[-2]
    elif condition_type == '1230_1330_inertia':
      buy_price = df_1m['close'].iloc[-2]
    else:
      buy_price = df_1m['close'].iloc[-2]

    # if condition_type == 'before_market_open_1':
    #    buy_price = min(df_1m['close'].iloc[-1] * 0.95, df['close'].iloc[-5 : -1].min()) # ver.1: 0.996; ver.2: 0.975; ver.3: 0.965
    # elif condition_type == 'before_market_open_2':
    #    buy_price = min(df_1m['close'].iloc[-1] * 0.938, df['close'].iloc[-5 : -1].min()) # ver.1: 0.993; ver.2: 0.965 ver3: 0.953
    i = df.shape[0] - 1
    if condition_type in ['before_market_open_1', 'before_market_open_2']:
      if not market_time:
        if not market_time_and_2hour_before:
          buy_price = df_1m['close'].iloc[-1] * 0.98
        elif not market_time_and_1hour_before:
          buy_price = df_1m['close'].iloc[-1] * 0.99
        elif not market_time_and_30min_before:
          buy_price = df_1m['close'].iloc[-1] * 0.997
      elif market_time_30min_before_closing:
          buy_price = minimum(df,i,k=1)
      else:
        buy_price = 0
    
    if condition_type in ['MA50_MA5', 'MA5_MA120_DS']:
      
      if prt_from_local_min(df_1m) < 0.3:           
        buy_price = df_1m['close'].iloc[-1] * 1.0003
      else:
        buy_price = min(df_1m['close'].iloc[-15:-1].min(), df_1m['open'].iloc[-15:-1].min(), df_1m['close'].iloc[-1])

  except Exception as e:
    alarm.print(traceback.format_exc())        
  return buy_price

def recalculate_trailing_LIT_gain_coef(df, order, current_price) -> Tuple[pd.DataFrame, pd.DataFrame]:
  # Recalculate Trailing LIT gain coefficient:
  current_gain = current_price / order['buy_price']
  trailing_LIT_gain_coef = order['trailing_LIT_gain_coef']
  if current_gain >= 1.005 and current_gain < 1.0055:
    trailing_LIT_gain_coef = 1.0065
  elif current_gain >= 1.0055 and current_gain <= 1.006:
    trailing_LIT_gain_coef = 1.0075
  elif current_gain > 1.006:
      trailing_LIT_gain_coef = 1.008

  if current_gain >= 1.006:
    ma.cancel_order(order=order, order_type='trailing_LIT')
  elif order['trailing_LIT_gain_coef'] != trailing_LIT_gain_coef:
          order_id = ma.modify_limit_if_touched_order(order, trailing_LIT_gain_coef,
                                                      aux_price_coef=aux_price_coef,
                                                      order_type='trailing_LIT' 
          ) 
          order['trailing_LIT_gain_coef'] = trailing_LIT_gain_coef
          df = ti.update_order(df, order) 
  return df, order

def check_sell_orders_for_all_bougth_stocks():
    global df
    # 2. Check that for all bought stocks placed LIMIT_IF_TOUCHED, STOP orders, Trailing STOP LIMIT/trailing_LIT_order if needed
    for ticker in positions_list:
        try:
            if ticker in bought_stocks_list:
                order = bought_stocks.loc[bought_stocks['ticker'] == ticker].sort_values('buy_time').iloc[-1]
                stock_df_1m = get_historical_df(ticker=ticker, period='max', interval='1m', prepost=prepost_1m)
                stock_df = get_historical_df(ticker=ticker, period=period, interval=interval, prepost=prepost_1h)
                if not stock_df_1m.empty:
                    current_price = stock_df_1m['close'].iloc[-1]
                    current_gain = current_price / order['buy_price']
                else:
                    current_price = 0
                    current_gain = 0
                if not order['buy_condition_type'] in ['1230', 'before_market_open_1', 'before_market_open_2', 'before_market_open_3', 'MA50_MA5', 'MA5_MA120_DS']:
                    # Checking limit_if_touched_order
                    if False:
                        df, order = check_sell_order_has_been_placed(df, order, ticker, order_type='limit_if_touched')
                    # Checking stop_order
                    df, order = check_sell_order_has_been_placed(df, order, ticker, order_type='stop')
                    # Checking trailing_LIT_order
                    df, order = check_sell_order_has_been_placed(df, order, ticker, order_type='trailing_LIT')
                    # Checking trailing_stop_limit_order
                    df, order = check_sell_order_has_been_placed(df, order, ticker, order_type='trailing_stop_limit')
                    # Modification of trailing stop limit order based on current gain
                    df, order = trailing_stop_limit_order_trailing_ratio_modification(df, order, current_price)
                    # Recalculate Trailing LIT gain coefficient:
                    df, order = recalculate_trailing_LIT_gain_coef(df, order, current_price)
                if order['buy_condition_type'] == '1230':
                    # Place limit order if end of the trading day for 1230 buy condition
                    df, order = place_traililing_stop_limit_order_at_the_end_of_trading_day(df, order, ticker, current_price)
                    # Modify order with buy_codition_type = 12:30
                    df, order = modify_trailing_stop_limit_1230_order(df, order, current_price)
                if order['buy_condition_type'] in ['before_market_open_2', 'before_market_open_3']:
                    # Place limit if touched sell order 
                    df, order = check_sell_order_has_been_placed(df, order, ticker, order_type='limit_if_touched')
                    # Place stop limit sell order if gain more that 2%:
                    if order['buy_condition_type'] in ['before_market_open_2'] and current_gain >= 1.01:  # v1. >=1.02 v2.1.005
                        df, order = check_sell_order_has_been_placed(df, order, ticker, order_type='stop_limit')
                    if order['buy_condition_type'] == 'before_market_open_3' and current_gain > 1.003:
                        df, order = check_sell_order_has_been_placed(df, order, ticker, order_type='stop_limit')
                    # Modify stop limit sell order based on current gain and sell if fast rise 3%
                    df, order = modify_stop_limit_before_market_open_order(df, order, current_price)
                if order['buy_condition_type'] in ['MA50_MA5', 'MA5_MA120_DS', 'before_market_open_1']:
                    # Place limit if touched sell order 1
                    df, order = check_sell_order_has_been_placed(df, order, ticker, order_type='limit_if_touched')
                    df, order = check_sell_order_has_been_placed(df, order, ticker, order_type='stop_limit')
                    # if current_gain >= 1.001:  
                    df, order = check_sell_order_has_been_placed(df, order, ticker, order_type='trailing_stop_limit')
                    # Modify trailing stop limit sell order based on current gain
                    if order['buy_condition_type'] in ['MA50_MA5', 'before_market_open_1']:
                        df, order = modify_trailing_stop_limit_MA50_MA5_order(df, order, current_price, stock_df, stock_df_1m)
                    if order['buy_condition_type'] == 'MA5_MA120_DS':
                        df, order = modify_trailing_stop_limit_MA5_MA120_DS_order(df, order, current_price, stock_df, stock_df_1m)
            else:
                alarm.print(f'{ticker} is in positional list but not in bought stock lisk!')
                if ticker not in placed_stocks_list:
                    orders = ma.get_history_orders(code=ticker)
                    if not orders.empty:
                        order = orders.loc[(orders['trd_side'] == 'BUY') & (orders['order_status'] == 'FILLED_ALL')].sort_values('updated_time', ascending=True).iloc[-1]
                        if not (order is None or order.empty):
                            buy_price = order['dealt_avg_price']
                            buy_condition_type = 'MA50_MA5'
                            buy_order_type = 'limit_if_touched'
                            buy_sum = order['dealt_qty'] * buy_price  + 1
                            order['buy_order_id'] = order['order_id']
                            order_df_format = ti.buy_order(
                                ticker=ticker,
                                buy_price=buy_price,
                                buy_condition_type=buy_condition_type,
                                default=default,
                                buy_sum=buy_sum,
                                order_type=buy_order_type,
                                skip_buy_checks=True,
                                add_order=order
                            )
                            order_df_format['gain_coef'] = default.gain_coef_MA50_MA5
                            order_df_format['lose_coef'] = default.lose_coef_1_MA50_MA5
                            df = ti.record_order(df, order_df_format)
        except Exception as e:
            alarm.print(traceback.format_exc())

def check_stocks_for_inclusion():

  warning.print('Checking optimal stock list for inclusion ...')
  for ticker in stock_name_list: 
    try:
      if ticker in exclude_time_dist:
        delta = datetime.now() - exclude_time_dist[ticker]
        deltatime_minutes = delta.seconds / 60 + delta.days * 24 * 60
        if deltatime_minutes >= 30 \
          and ticker not in stock_name_list_opt:
          stock_name_list_opt.append(ticker)
          exclude_time_dist.pop(ticker, None)
          warning.print(f'Stock {ticker} is including to optimal stock list')
    except Exception as e:
      alarm.print(traceback.format_exc())
    
  try:
      file_name = f'temp/exclude_time_dist.plk'
      with open(file_name, 'wb') as file:
        pickle.dump(exclude_time_dist, file)
        warning.print(f'Optimal stock name list len after inclustion check is {len(stock_name_list_opt)}')
      
      file_name = f'temp/stock_name_list_opt.plk'
      with open(file_name, 'wb') as file:
        pickle.dump(stock_name_list_opt, file)

  except Exception as e:
      alarm.print(traceback.format_exc())
#%% MAIN
if __name__ == '__main__':
  
    # Interface initialization
    alarm.print('YOU ARE RUNNING REAL TRADE ACCOUNT')
    ma = Moomoo_API(ip, port, trd_env=TRD_ENV, acc_id = ACC_ID)
    ti = TradeInterface(platform='moomoo', df_name='real_trade_db', moomoo_api=ma)
    # Load trade history
    if load_from_csv:
        df = load_orders_from_csv()
    elif load_from_xslx:
        df = load_orders_from_xlsx()
    else:
        df = ti.load_trade_history() # load previous history
    df = df.drop_duplicates()
    # Cleaning
    if clean_placed_orders:
        df = clean_cancelled_and_failed_orders_history(df, type='placed')
    if clean_cancelled_orders:
        df = clean_cancelled_and_failed_orders_history(df, type='cancelled')
    # Save csv/excel/df/
    ti.__save_orders__(df)
    # Print last hours profit
    profit_24hours = current_profit(df, hours=24)
    warning.print(f'Last 24 hours profit is {profit_24hours:.2f}')
    profit_48hours = current_profit(df, hours=48)
    warning.print(f'Last 48 hours profit is {profit_48hours:.2f}')

    file_name = f'temp/exclude_time_dist.plk'
    file = pathlib.Path(file_name)
    if file.is_file():
        with open(file_name, 'rb') as file:
            exclude_time_dist  = pickle.load(file)

    file_name = f'temp/stock_name_list_opt.plk'
    file = pathlib.Path(file_name)
    if file.is_file():
        with open(file_name, 'rb') as file:
            stock_name_list_opt = pickle.load(file)

    # SQL INIT
    try:
        parent_path = pathlib.Path(__file__ ).parent
        folder_path = pathlib.Path.joinpath(parent_path, 'sql')
        db = sql_db.DB_connection(folder_path, 'trade.db', df)
        if load_from_csv or load_from_xslx or read_sql_from_df:
            db.update_db_from_df(df)
    except Exception as e:
        alarm.print(traceback.format_exc())

    us_cash = ma.get_us_cash()
    profit_24hours = current_profit(df)
    c.print(f'Availble withdrawal cash is {us_cash}')
    c.print(f'Last 24 hours profit is {profit_24hours}')

    # Algorithm !!! not up-to-date
    if True:
        pass  
        # Moomoo trade algo:
        # 1. Check what stocks are bought based on MooMoo 
            # Recheck all placed buy orders from the order history
            # Recheck all placed buy orders from the orders by cancel time condition
            # Check statuses of all bought stocks if they not in positional list:
        # 2. Check that for all bought stocks placed LIMIT_IF_TOUCHED, STOP orders, Trailing STOP LIMIT/trailing_LIT_order if condition
        # 3. For optimal stock list:
        # 3.1 If counter condition:
            # Recheck for all tickers in positional list that LIMIT/trailing_LIT_order is placed if condition
            # Recalculate Trailing LIT gain coefficient:
            # Recheck all placed order information  
        # 3.2 Get historical data, current_price for bought stocks
        # 3.3 DYNAMIC GAIN COEFFICIENT
            # if gain_coef\lose_coef doesn't match order's modify the order:
        # 3.4. BUY SECTION:
        # 3.5. Recheck buy order information including commission from the order history
        # 3.6. CHECKING FOR CANCELATION OF THE BUY ORDER
            # Check current price and status of buy order
            # If price more or equal than price*gain_coef CANCEL the BUY ORDER
            # If status was not FILLED_PART, change status of order to 'cancel', update the order
        # 3.7 Checking if sell orders have been executed

        # Orders types:
        # buy_order
        # Sell orders:
        # limit_if_touched_order
        # stop_order
        # trailing_LIT_order
        # trailing_stop_limit

    market_value = -1 # start offset
    total_market_value = -99  
    # Calculate market direction
    total_market_value_2m = 0
    total_market_value_5m = 0
    total_market_value_30m = 0
    total_market_value_60m = 0
    total_market_direction_10m = 0
    total_market_direction_60m = 0
    hash_df_1m = {}
    
    if not skip_market_direction_calc:  
        warning.print('Calculation of market direction:')
        for ticker in tqdm(stock_name_list_opt):
            try:
                stock_df_1m = get_historical_df(ticker = ticker, period='max', interval='1m')
                total_market_value_2m += minmax(stock_df_1m['pct'].iloc[-2:-1].sum(), -3, 3)
                total_market_value_5m += minmax(stock_df_1m['pct'].iloc[-5:-1].sum(), -3, 3)
                total_market_value_30m += minmax(stock_df_1m['pct'].iloc[-30:-1].sum(), -3, 3)
                total_market_value_60m += minmax(stock_df_1m['pct'].iloc[-60:-1].sum(), -3, 3)
                if stock_df_1m['pct'].iloc[-60:-1].sum() > 0:
                    total_market_direction_60m += 1 
                else:
                    total_market_direction_60m -= 1
                if stock_df_1m['pct'].iloc[-10:-1].sum() > 0:
                    total_market_direction_10m += 1 
                else:
                    total_market_direction_10m -= 1
            except Exception as e:
                alarm.print(traceback.format_exc())
        warning.print(f'Total markert direction 10m is {total_market_direction_10m}')
        warning.print(f'Total markert direction 60m is {total_market_direction_60m}')

    while True:
    
        alarm.print('YOU ARE RUNNING REAL TRADE ACCOUNT')  
        # 1. Check what stocks are bought based on MooMoo (position list) and df
        positions_list = ma.get_positions()
        # Statistic calculations 
        df_stats = {}
        # df_stats = stats_calculation(stock_name_list_opt)
        # ReLoad trade history (syncronization) as maybe losing df somewhere
        if load_from_csv:
            df = load_orders_from_csv()
        elif load_from_xslx:
            df = load_orders_from_xlsx()
        else:
            df = ti.load_trade_history() # load previous history

        check_stocks_for_inclusion()
    
        # working with market time
        current_minute = datetime.now().astimezone().minute 
        current_hour = datetime.now().astimezone().hour
        tzinfo_ny = pytz.timezone('America/New_York')
        new_york_time = datetime.now().astimezone(tzinfo_ny)
        new_york_hour = new_york_time.hour
        new_york_minute = new_york_time.minute
        new_york_week = new_york_time.weekday() 
        market_time = (new_york_hour >= 10 and new_york_hour < 16) \
                    or (new_york_hour ==9 and new_york_minute >= 30) \
                    and new_york_week in [0, 1, 2 ,3, 4] 
        market_time_before_1430 = (new_york_hour >= 10 and new_york_hour < 14) \
                    or (new_york_hour == 9 and new_york_minute >= 30) \
                    or (new_york_hour == 14 and new_york_minute <= 30) \
                    and new_york_week in [0, 1, 2 ,3, 4]
        market_time_and_1hour_before = (new_york_hour >= 9 and new_york_hour < 16) \
                    or (new_york_hour == 8 and new_york_minute >= 30) \
                    and new_york_week in [0, 1, 2 ,3, 4]
        market_time_and_2hour_before = (new_york_hour >= 8 and new_york_hour < 16) \
                    or (new_york_hour == 7 and new_york_minute >= 30) \
                    and new_york_week in [0, 1, 2 ,3, 4]
        market_time_and_30min_before = (new_york_hour >= 9 and new_york_hour < 16) \
                    and new_york_week in [0, 1, 2 ,3, 4]
        market_time_30min_before_closing = (new_york_hour == 15 and new_york_minute >= 30)

        # Get current orders and they lists:
        limit_if_touched_sell_orders, stop_sell_orders, limit_buy_orders, \
        limit_if_touched_buy_orders, trailing_LIT_sell_orders, trailing_stop_limit_sell_orders, \
            stop_limit_buy_orders, stop_limit_sell_orders = ma.get_orders()
        buy_orders = pd.concat([limit_if_touched_buy_orders, limit_buy_orders])
        limit_buy_orders_list = get_orders_list_from_moomoo_orders(limit_buy_orders)
        limit_if_touched_buy_orders_list = get_orders_list_from_moomoo_orders(limit_if_touched_buy_orders)
        limit_if_touched_sell_orders_list = get_orders_list_from_moomoo_orders(limit_if_touched_sell_orders)
        stop_sell_orders_list = get_orders_list_from_moomoo_orders(stop_sell_orders)
        trailing_LIT_sell_orders_list = get_orders_list_from_moomoo_orders(trailing_LIT_sell_orders)
        trailing_stop_limit_sell_orders_list = get_orders_list_from_moomoo_orders(trailing_stop_limit_sell_orders)
        stop_limit_sell_orders_list = get_orders_list_from_moomoo_orders(stop_limit_sell_orders)
        stop_limit_buy_orders_list = get_orders_list_from_moomoo_orders(stop_limit_buy_orders)

        historical_orders = ma.get_history_orders()
        # Check bought stocks and placed based on df 
        bought_stocks, placed_stocks, bought_stocks_list, placed_stocks_list = get_bought_and_placed_stock_list(df)
    
        for ticker in list(set(placed_stocks_list + limit_buy_orders_list + limit_if_touched_buy_orders_list + stop_limit_buy_orders_list)):
            # Recheck all placed buy orders from the order history
            df = check_buy_order_info_from_order_history(df, ticker)
            # Recheck all placed buy orders from the orders by cancel time condition  
            df = check_buy_order_for_cancelation(df, ticker, historical_orders)
    
        if not market_time:
            for ticker in placed_stocks_list:
                try:
                    orders = df.loc[(df['ticker'] == ticker) & df['status'].isin(['placed'])]
                    if orders.shape[0] > 0:
                        order = orders.sort_values('buy_time').iloc[-1]
                        stock_df_1m = get_historical_df(ticker = ticker, period='max', interval='1m', prepost=prepost_1m)
                        df, order = modify_buy_limit_if_touched_before_market_open_order(df, order, stock_df_1m)
                except Exception as e:
                    alarm.print(traceback.format_exc())
    
        for ticker in bought_stocks_list:
            # Checking if sell orders have been executed
            df = check_if_sell_orders_have_been_executed(df, ticker)

        # Check statuses of all bought stocks if they not in positional list:
        try:
        # ticker in bought stocks list after confirmation of the buy order
            for ticker in bought_stocks_list:
                orders = df.loc[(df['ticker'] == ticker) & df['status'].isin(['bought', 'filled_part'])]
                if orders.shape[0] > 0:
                    order = orders.sort_values('buy_time').iloc[-1]
                    if ticker not in positions_list:
                        historical_orders_= historical_orders.loc[(
                            historical_orders['order_status']  == ft.OrderStatus.FILLED_ALL) &
                            (historical_orders['code'] == MARKET + ticker) &
                            (historical_orders['trd_side'] == ft.TrdSide.SELL) &
                            (historical_orders['qty'] == order['stocks_number'])
                        ]
                        if historical_orders_.shape[0] > 0:
                            historical_order = historical_orders_.sort_values('updated_time').iloc[-1]
                            order = ti.sell_order(order, sell_price=0.01, historical_order = historical_order)
                            df = ti.update_order(df, order)
                            if order['trailing_LIT_order_id'] not in ['', None, []]:                  
                                ma.cancel_order(order, order_type='trailing_LIT')    
                            if order['stop_order_id'] not in ['', None, []]:                  
                                ma.cancel_order(order, order_type='stop')
                            if order['trailing_stop_limit_order_id'] not in ['', None, []]:                  
                                ma.cancel_order(order, order_type='trailing_stop_limit')
                            if order['limit_if_touched_order_id'] not in ['', None, []]:                  
                                ma.cancel_order(order, order_type='limit_if_touched')            
        except Exception as e:
            alarm.print(traceback.format_exc())

        # 2. Check that for all bought stocks placed LIMIT_IF_TOUCHED, STOP orders, Trailing STOP LIMIT/trailing_LIT_order if needed
        check_sell_orders_for_all_bougth_stocks()

        # 3. For optimal stock list:
        counter = 0
        market_value_2m = 0
        market_value_5m = 0
        market_value_30m = 0
        market_value_60m = 0
        market_direction_10m = 0
        market_direction_60m = 0
        
        blue.print(f'total market_value {total_market_value:.2f}, 2m is {total_market_value_2m:.2f}, 5m is {total_market_value_5m:.2f},')
        blue.print(f'30m is {total_market_value_30m:.2f}, 60m is {total_market_value_60m:.2f}')
        blue.print(f'total market direction is {total_market_direction_10m:.2f}')
        blue.print(f'total market direction is {total_market_direction_60m:.2f}')
        us_cash = ma.get_us_cash()
        c.print(f'Available cash is {us_cash:.2f}, min buy sum is {min_buy_sum}', color='turquoise')
     
        if us_cash > min_buy_sum \
        or test_buy_sim:

            for ticker in list(set(stock_name_list_opt) - set(placed_stocks_list) - set(bought_stocks_list)):
                print(f'Stock is {ticker}')

                order = []    
                # 3.1 If counter condition:  
                counter += 1
                if counter > 20:
                    us_cash = ma.get_us_cash()
                    bought_stocks, placed_stocks, bought_stocks_list, placed_stocks_list = get_bought_and_placed_stock_list(df)
                    historical_orders = ma.get_history_orders()
                    # Get current orders and they lists:
                    limit_if_touched_sell_orders, stop_sell_orders, limit_buy_orders, \
                    limit_if_touched_buy_orders, trailing_LIT_sell_orders, trailing_stop_limit_sell_orders, \
                        stop_limit_buy_orders, stop_limit_sell_orders = ma.get_orders()
                    buy_orders = pd.concat([limit_if_touched_buy_orders, limit_buy_orders])
                    limit_buy_orders_list = get_orders_list_from_moomoo_orders(limit_buy_orders)
                    limit_if_touched_buy_orders_list = get_orders_list_from_moomoo_orders(limit_if_touched_buy_orders)
                    limit_if_touched_sell_orders_list = get_orders_list_from_moomoo_orders(limit_if_touched_sell_orders)
                    stop_sell_orders_list = get_orders_list_from_moomoo_orders(stop_sell_orders)
                    trailing_LIT_sell_orders_list = get_orders_list_from_moomoo_orders(trailing_LIT_sell_orders)
                    trailing_stop_limit_sell_orders_list = get_orders_list_from_moomoo_orders(trailing_stop_limit_sell_orders)
                    stop_limit_sell_orders_list = get_orders_list_from_moomoo_orders(stop_limit_sell_orders)
                    stop_limit_buy_orders_list = get_orders_list_from_moomoo_orders(stop_limit_buy_orders)

                    # Recheck for all tickers in positional list that LIMIT/trailing_LIT_order are placed if condition
                    # and trailing ratio modification
                    for ticker3 in placed_stocks_list:
                        # Recheck all placed order information
                        df = check_buy_order_info_from_order_history(df, ticker3)
                    
                    # include if they were placed and modification of sell orders
                    check_sell_orders_for_all_bougth_stocks()
                                                
                    for ticker4 in bought_stocks_list:
                        # Checking if sell orders have been executed
                        df = check_if_sell_orders_have_been_executed(df, ticker4)

                    counter = 0
                    if us_cash < min_buy_sum \
                        and not test_buy_sim:
                        break

                # 3.2 Get historical data, current_price for stocks in optimal list
                try:
                    stock_df = get_historical_df(ticker = ticker, period=period, interval=interval, prepost=prepost_1h)
                    stock_df_pred = get_prediction_df(stock_df, prediction=1.005)
                    try:
                        stock_df_1m = get_historical_df(ticker = ticker, period='max', interval='1m', prepost=prepost_1m)
                        if stock_df_1m.shape[0] > 0:
                            current_price = stock_df_1m['close'].iloc[-1]
                            current_timezone = datetime.now().astimezone().tzinfo
                            time_is_correct =  (datetime.now().astimezone() - stock_df_1m.index[-1].astimezone(current_timezone)).seconds  < 60 * 3
                            if time_is_correct:
                                market_value += stock_df_1m['pct'].iloc[-2]
                            else:
                                market_value  += 0
                            market_value_2m += minmax(stock_df_1m['pct'].iloc[-2:-1].sum(), -3, 3)
                            market_value_5m += minmax(stock_df_1m['pct'].iloc[-5:-1].sum(), -3 ,3)
                            market_value_30m += minmax(stock_df_1m['pct'].iloc[-30:-1].sum(), -3, 3)
                            market_value_60m += minmax(stock_df_1m['pct'].iloc[-60:-1].sum(), -3, 3)
                            if stock_df_1m['pct'].iloc[-10:-1].sum() > 0.02:
                                market_direction_10m += 1
                            else:
                                market_direction_10m -= 1            
                            if stock_df_1m['pct'].iloc[-60:-1].sum() > 0.2:
                                market_direction_60m += 1 
                            else:
                                market_direction_60m -= 1
                        
                    # warning.print(f'market_value is {market_value:.2f},market_value_2m is {market_value_2m:.2f},  market_value_5m is {market_value_5m:.2f}')
                    # warning.print(f'market_value_30m is {market_value_30m:.2f}, market_value_60m is {market_value_60m:.2f}')
                    except Exception as e:
                        if not(stock_df is None):
                            current_price = stock_df['close'].iloc[-1]
                        alarm.print(traceback.format_exc()) 
                except Exception as e:
                    alarm.print(traceback.format_exc())
                    stock_df = None
            
                # 3.4 BUY SECTION:
                conditions_info = ''
                try:
                    if not(stock_df is None) and not(stock_df_1m is None) and stock_df_1m.shape[0] > 0:
                        if not(ticker in bought_stocks_list or ticker in placed_stocks_list):
                            # if market_time_before_1430 or test_buy_sim:
                            if market_time_and_1hour_before or test_buy_sim:
                                buy_condition_MA50_MA5, conditions_info = stock_buy_condition_MA50_MA5(
                                stock_df, stock_df_pred, stock_df_1m, df_stats, ticker, display=True)
                            else: 
                                buy_condition_MA50_MA5 = False
                                # buy_condition_MA5_MA120_DS = stock_buy_condition_MA5_MA120_DS(stock_df, stock_df_1m, df_stats, ticker, display=True)
                                buy_condition_MA5_MA120_DS = False
                                # buy_condition, buy_condition_type_1h  = stock_buy_conditions(stock_df, stock_df_1m, ticker)
                                buy_condition, buy_condition_type_1h = False, 'No cond'
                                # buy_condition_930_47 = stock_buy_condition_930_47(stock_df_1m)
                                buy_condition_930_47, buy_condition = False, False
                                # buy_condition_1230 = stock_buy_condition_1230(stock_df, stock_df_1m, display=False)
                                buy_condition_1230 = False
                                # buy_condition_maxminnorm = stock_buy_condition_maxminnorm(stock_df, stock_df_1m, df_stats, display=False)
                                buy_condition_maxminnorm = False
                                # buy_condition_speed_norm100 = stock_buy_condition_speed_norm100(stock_df, stock_df_1m, df_stats, display=True)
                                buy_condition_speed_norm100 = False
                            if not market_time and buy_when_market_closed:
                                buy_condition_before_market_open, condition_type_before_market_open = \
                                stock_buy_condition_before_market_open(stock_df, stock_df_1m, df_stats, display=True)
                            else:
                                buy_condition_before_market_open, condition_type_before_market_open = False, 'No cond' 
                        else:
                            buy_condition = False
                            buy_condition_1230 = False
                            buy_condition_930_47 = False
                            buy_condition_maxminnorm = False
                            buy_condition_speed_norm100 = False
                            buy_condition_before_market_open = False
                            buy_condition_MA50_MA5 = False
                            buy_condition_MA5_MA120_DS = False
                            buy_condition_type_1h = 'No cond'
                            condition_type_before_market_open = 'No cond'
                            
                        buy_condition_type = 'No cond'
                        if buy_condition:
                            buy_condition_type = buy_condition_type_1h 
                        if buy_condition_1230:
                            buy_condition_type = '1230'
                        if buy_condition_930_47:
                            buy_condition_type = '930-47'
                        if buy_condition_maxminnorm:
                            buy_condition_type = 'maxminnorm'
                        if buy_condition_speed_norm100:
                            buy_condition_type = 'speed_norm100'
                        if buy_condition_before_market_open:
                            buy_condition_type = condition_type_before_market_open
                        if buy_condition_MA5_MA120_DS:
                            buy_condition_type = 'MA5_MA120_DS'
                        if buy_condition_MA50_MA5:
                            buy_condition_type = 'MA50_MA5'  
                                
                        if buy_condition:
                            c.green_red_print(buy_condition, 'buy condition')
                            c.print(f'buy_condition_type is {buy_condition_type}', color = 'green')

                        print(f'stock {ticker}, time: {stock_df.index[-1]} last price is {stock_df['close'].iloc[-1]:.2f}, pct is {stock_df['pct'].iloc[-1]:.2f}')
                        c.print(f'_'*100)
                        current_timezone = datetime.now().astimezone().tzinfo
                        try:
                            time_is_correct =  (datetime.now().astimezone() - stock_df.index[-1].astimezone(current_timezone)).seconds  < 60 * 60 * 1 + 60 * 5 
                            if override_time_is_correct:
                                time_is_correct = True
                        except Exception as e:
                            time_is_correct = False
                            alarm.print(traceback.format_exc())
                        # c.print(f'Time is correct condition {time_is_correct}', color='yellow')

                        buy_price = buy_price_based_on_condition(stock_df, stock_df_1m, buy_condition_type)
                        
                        if prt_from_local_min(stock_df_1m) < 0.3 and market_time:           
                            buy_order_type = 'limit' 
                        else:
                            buy_order_type = 'limit_if_touched'

                            # do_not_buy_schedule = (stock_df_1m.index[-1].hour == 10 and stock_df_1m.index[-1].minute >= 40) \
                            #   or (stock_df_1m.index[-1].hour == 11 and stock_df_1m.index[-1].minute <= 5)
                            # if do_not_buy_schedule: # change to 10:30???
                            #   alarm.print(f'Do not buy schedule is active')
                        do_not_buy_schedule = False

                        if (buy_condition  \
                            or buy_condition_1230 \
                            or buy_condition_930_47 \
                            or buy_condition_maxminnorm \
                            or buy_condition_speed_norm100 \
                            or buy_condition_before_market_open \
                            or buy_condition_MA50_MA5 \
                            or buy_condition_MA5_MA120_DS \
                            or test_buy_sim)\
                            and buy_price != 0 \
                            and (time_is_correct or buy_condition_before_market_open)  \
                            and not do_not_buy_schedule:

                            us_cash = ma.get_us_cash()
                            profit_24hours = current_profit(df)
                            c.print(f'Availble withdrawal cash is {us_cash}')
                            c.print(f'Last 24 hours profit is {profit_24hours}')

                            # Check last 24 hours profit condition and
                            # Check available money for trading based on Available Funds
                            security_condition = True
                            security_condition_1m = True
                            security_condition_1h = True
                            if us_cash < min_buy_sum:
                                security_condition = False
                                alarm.print('Available funds less than minumim buy sum')

                            # add condtion based on money permitted for trade !!!

                            if profit_24hours <= stop_trading_profit_value:
                                security_condition = False
                                alarm.print(f'Profit value for last 24 hours {profit_24hours} is less {stop_trading_profit_value}')

                            if stock_df['close'].iloc[-1] > max_stock_price:
                                security_condition = False
                                alarm.print(f'''Stock price {stock_df['close'].iloc[-1]} more than maximum allowed price {max_stock_price}''')
                                
                            if use_market_direction_condition:
                                if new_york_hour == 9:
                                    if total_market_direction_10m < 0:
                                        security_condition = False
                                        alarm.print(f'''Market direction 10m {total_market_direction_10m} is negative''')
                                elif total_market_value_60m <= 0 or total_market_direction_60m <= 0:
                                    security_condition = False
                                    alarm.print(f'''Total market value 60m {total_market_value_60m:.2f} or
                                                total market direction 60m {total_market_direction_60m} are negative''')

                            if (total_market_value_2m < 0 and total_market_value_5m < -1) or total_market_value_30m < -10:
                                security_condition_1m = False
                                # alarm.print(f'Market seems to be dropping')
                            
                            # if total_market_value_30m < -15:
                            #   security_condition_1h = False
                            #   alarm.print(f'Market seems to be dropping')

                            # Calculate buy_sum based on available money and min and max buy_sum condition
                            if us_cash < default_buy_sum:
                                buy_sum = us_cash
                            else:
                                buy_sum = default_buy_sum
                            if buy_sum < min_buy_sum:
                                buy_sum = 0
                            if buy_sum > max_buy_sum:
                                buy_sum = max_buy_sum

                            # if all conditions are met place buy order
                            if security_condition \
                                and (market_time or buy_when_market_closed):
                                    if (security_condition_1h and buy_condition) \
                                    or buy_condition_1230 \
                                    or buy_condition_930_47 \
                                    or buy_condition_maxminnorm \
                                    or buy_condition_speed_norm100 \
                                    or buy_condition_before_market_open \
                                    or buy_condition_MA50_MA5 \
                                    or buy_condition_MA5_MA120_DS \
                                    or test_buy_sim:
                                        # play sound:
                                        winsound.PlaySound('SystemHand', winsound.SND_ALIAS)
                                        order = ti.buy_order(ticker=ticker, buy_price=buy_price, buy_condition_type=buy_condition_type, default=default, buy_sum=buy_sum, order_type=buy_order_type)

                                        if buy_condition_type == 'speed_norm100':
                                            order['gain_coef'] = default.gain_coef_speed_norm100
                                        elif buy_condition_type in ['before_market_open_1', 'before_market_open_2', 'before_market_open_3']:
                                            order['gain_coef'] = default.gain_coef_before_market_open
                                            order['lose_coef'] = default.lose_coef_before_market_open
                                        elif buy_condition_type == 'MA50_MA5':
                                            order['gain_coef'] = default.gain_coef_MA50_MA5
                                            order['lose_coef'] = default.lose_coef_1_MA50_MA5
                                        elif buy_condition_type == 'MA5_MA120_DS':
                                            order['gain_coef'] = default.gain_coef_MA5_MA120_DS
                                            order['lose_coef'] = default.lose_coef_MA5_MA120_DS
                                        else:
                                            order['gain_coef'] = default.gain_coef
                                        
                                        if buy_condition_type == '1h':
                                                order['lose_coef'] = lose_coef_1h
                                
                                        #                   order['tech_indicators'] = f'mv :{market_value:.2f}, mv_2m:{market_value_2m:.2f},\
                                        # mv_5m : {market_value_5m:.2f}, mv_30m : {market_value_30m:.2f}, mv_60m: {market_value_60m:.2f}, \
                                        # md_60m: {market_direction_60m}' + conditions_info
                                        order['tech_indicators'] = conditions_info
                                        if order['status'] == 'placed':
                                            placed_stocks_list.append(ticker)
                                            df = ti.record_order(df, order)
                                            bought_stocks, placed_stocks, bought_stocks_list, placed_stocks_list = get_bought_and_placed_stock_list(df)
                                            # Recheck all placed buy orders from the order history
                                            df = check_buy_order_info_from_order_history(df, ticker)
                                            if place_trailing_stop_limit_order_imidiately \
                                            and not (order['buy_condition_type'] in ['1230', 'speed_norm100', 'before_market_open_1', 
                                                                                    'before_market_open_2', 'before_market_open_3',
                                                                                    'MA50_MA5', 'MA5_MA120_DS']):
                                                price = order['buy_price'] * default.gain_coef
                                                # Checking trailing_stop_limit_order
                                                # df, order = check_sell_order_has_been_placed(df, order, ticker, order_type='trailing_stop_limit')
                                                df, order = place_sell_order_if_it_was_not_placed(df,
                                                order=order,
                                                sell_orders=trailing_stop_limit_sell_orders,
                                                sell_orders_list=trailing_stop_limit_sell_orders_list,
                                                price=price,
                                                order_type='trailing_stop_limit') 
                    else:
                        if stock_df is None:
                            alarm.print(f'Historical data is None for {ticker}')
                        if stock_df_1m is None or stock_df_1m.shape[0] == 0:
                            alarm.print(f'Historical 1m data is None or empty for {ticker}')
                        print('--'*50)
                except Exception as e:
                    alarm.print(traceback.format_exc())

                # 3.5 Recheck placed orders information including commission from the order history
                df = check_buy_order_info_from_order_history(df, ticker)
                # 3.6 CHECKING FOR CANCELATION OF THE BUY ORDER
                df = check_buy_order_for_cancelation(df, ticker, historical_orders)
                # 3.7 Checking if sell orders have been executed
                df = check_if_sell_orders_have_been_executed(df, ticker)

        total_market_value = market_value
        total_market_value_2m = market_value_2m
        total_market_value_5m = market_value_5m
        total_market_value_30m = market_value_30m 
        total_market_value_60m = market_value_60m
        total_market_direction_60m = market_direction_60m

        # Update SQL DB FROM df each full cycle!!!
        # try:
        #   db.update_db_from_df(df)
        # except Exception as e:
        #   alarm.print(traceback.format_exc())
        print('Waiting progress:')
        print(f'Number calls per minute is {yf_numbercalls.value}')
        if us_cash > min_buy_sum:
            for i in tqdm(range(25)):
                time.sleep(1)
        else:
            for i in tqdm(range(20)):
                time.sleep(1)        
        if not market_time_and_2hour_before:
            for i in tqdm(range(30)):
                time.sleep(1)    
# %%
