#region import
#%%
'''
preparing data set - personal index investing strategy
'''

# import for graphics:
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

import os, pathlib
import sys
from numpy.core.fromnumeric import argmax, size
from numpy.lib.function_base import append
from numpy.lib.ufunclike import fix
from pandas.core import series
from pandas.io.formats.format import return_docstring
#from pydantic.types import Json

import tinvest as ti
import investpy
from twelvedata import TDClient
import yfinance as yf

from datetime import datetime, timedelta, timezone
import time
import functions_2 as f2
from functions_2 import pct_ratio
# from pytz import timezone

import pickle
import numpy as np
from sklearn.metrics import mean_absolute_error
import pandas as pd
pd.options.mode.chained_assignment = None 

import telebot
from tinvest.clients import T
from termcolor import colored

dirname = os.path.dirname(__file__)
parent_dirname = os.path.dirname(dirname)
sys.path.append(parent_dirname)
from tqdm import tqdm

if __name__ == '__main__' or __name__ == 'piis_condition':
  import settings
  from personal_settings import personal_settings as ps
  from functions import SMA, WMA
  import schedule as sch
  from colog import colog
# else:
#   from torgach import settings
#   from torgach import schedule as sch
#   from torgach.personal_settings import personal_settings as ps
#   from torgach.functions import SMA, WMA
#   from torgach.colog import colog

# reper_point = 0
# for key, value in functions.__dict__.items() :  
#   if key == 'reper_function':
#     reper_point = 1
#   if type(value).__name__ == 'function' and reper_point == 1:    
#     vars()[key] = value


#endregion

col = settings.col
col_short = settings.col_short
sandbox_token = ps.sandbox_token
client_sandbox = ti.SyncClient(sandbox_token, use_sandbox=True)

def get_candle_list(client, stock_name, interval = '15min', outputsize = 5000, number_intervals = 1, display = False):

    close_price_list = []
    volume_list = []
    time_list = []
    candles_list = []
    candles_pct = []
    df = pd.DataFrame()
    
    for i in range(number_intervals, 0, -1):
      from_ = datetime(datetime.now().year, datetime.now().month, datetime.now().day, 0, 0, 1, 0, tzinfo=timezone.utc) - timedelta(days = i)
      to_ = datetime(datetime.now().year, datetime.now().month, datetime.now().day, 0, 0, 1, 0, tzinfo=timezone.utc) - timedelta(days = i - 1)
      
      read_success = 0
      while not(read_success):
        try:
          # Construct the necessary time series
          
          # ts = client.time_series(
          #     symbol=stock_name,
          #     interval=interval,
          #     outputsize=outputsize,
          #     timezone="America/New_York"
          # )

          # Returns pandas.DataFrame
          # new_df = ts.as_pandas()
          # new_df = yf.Ticker(stock_name).history(period='60d', interval='15m')
          new_df = yf.Ticker(stock_name).history(period='7d', interval='1m')
          new_df = new_df.rename(columns={"Open": "open", "Close" :'close',"High" : 'high', "Low": 'low'})
          new_df = new_df[['open', 'high', 'low', 'close']]

          new_df = f2.get_heiken_ashi_v2(new_df)
          df = pd.concat([df, new_df]) #'columns: open, high, low, close, volume'

          if display:
            print(f'Requst number is {i}')
          
          if new_df.shape[0] > 1:
            read_success = 1 
        except Exception as E:
          print(f'sleeping, {E}')
          time.sleep(60)

      if i == 100:
        time.sleep(60)

    
    df['pct'] = np.where(df['open'] < df['close'],  
                         (df['close'] / df['open'] - 1) * 100,
                         -(df['open'] / df['close'] - 1) * 100

    )

    return df

def get_index_list(df_stock_dict, k, window):
  
  df = pd.DataFrame()
  
  n = len(list(df_stock_dict.keys()))
  for i in range(n):
    stock_name = list(df_stock_dict.keys())[i]
    df[stock_name] = df_stock_dict[stock_name]['pct'][k - window: k]

  df_corr =  df.corr().sum().sort_values(ascending=False) / n
  index_stock_names = list(df_corr[df_corr >= df_corr.mean()].index)
  
  df_index = df[index_stock_names]
  df_index['index'] = df_index.sum(axis = 1)
  df_index = df_index.sort_index(ascending=False)

  index_sum = df_index['index'].sum() / len(index_stock_names)

  return  index_sum, df_corr

def heiken_ashi_condition_f(df):
  df = df.sort_index()
  ha_pct_sum = df['ha_pct'].sum()

  return ha_pct_sum


def main():
    
    candle_resolution = 5
    # window = 10 

    folder_load_path = os.path.join(dirname, f'historical_data\\onestock\\{candle_resolution}m_raw_test_start_20230716')
    folder_save_path = os.path.join(dirname, f'historical_data\\onestock\\{candle_resolution}m_raw_test_start_20230716_aug')
    if not(os.path.exists(folder_save_path)):
        os.mkdir(folder_save_path)

    # param_list = []
    # for window  in [1, 3, 5, 10, 20, 30, 50, 100, 150, 300]:
    #         param_list.append(f'pct_sum_{window}')
    #         param_list.append(f'ha_pct_sum_{window}')
           
    # param_list += ['green_red_prop_10', 'green_red_prop_15', 'green_red_prop_5']

    for item in os.listdir(folder_load_path):
        
        dataset_path = os.path.join(folder_load_path, item)
        print(f'Dataset is {dataset_path}')
        file = open(dataset_path, 'rb')
        df_stock = pickle.load(file)
        file.close()

        x = item.find('.pkl')
        item_name_without_pkl = item[:x]
        save_path = os.path.join(folder_save_path, f'{item_name_without_pkl}_aug.pkl')

        if not(os.path.exists(save_path)):

            # for param in param_list:
            #     df_stock[param] = np.nan

            df_stock['win_short'] = np.nan
            df_stock['win_long'] = np.nan
                
            for i in tqdm(range(300, df_stock.shape[0])):
            
                # for window in [1, 3, 5, 10, 20, 30, 50, 100, 150, 300]:
                #     locals()[f'pct_sum_{window}'] = df_stock['pct'][i - window: i - 1].sum()
                #     locals()[f'ha_pct_sum_{window}'] = df_stock['ha_pct'][i - window: i - 1].sum()
                    
                # green_red_prop_15 = (df_stock['ha_colour'][i - 15 : i - 1] == 'green').sum() / ((df_stock['ha_colour'][i - 15 : i - 1] == 'red').sum() + 1)
                # green_red_prop_10 = (df_stock['ha_colour'][i - 10 : i - 1] == 'green').sum() / ((df_stock['ha_colour'][i - 10 : i - 1] == 'red').sum() + 1)
                # green_red_prop_5 = (df_stock['ha_colour'][i - 5 : i - 1] == 'green').sum() / ((df_stock['ha_colour'][i - 5 : i - 1] == 'red').sum() + 1)
                
                #modeling buy the stock
                j = 0
                while_cond = True
                win_long = 0
                while j < df_stock.shape[0] - i and while_cond and j < 240:
                    if pct_ratio(df_stock['high'][i + j], df_stock['open'][i], with_sign=True) > 0.4:
                        while_cond = False
                        win_long = 1

                    if pct_ratio(df_stock['low'][i + j], df_stock['open'][i], with_sign=True) < -0.4:
                        while_cond = False
                
                    j += 1

                # #modeling buy the stock
                # j = 0
                # while_cond = True
                # win_short = 0
                # while j < df_stock.shape[0] - i and while_cond and j < 240:
                #     if pct_ratio(df_stock['low'][i + j], df_stock['open'][i], with_sign=True) < -0.4:
                #         while_cond = False
                #         win_short = 1

                #     if pct_ratio(df_stock['high'][i + j], df_stock['open'][i], with_sign=True) > 0.4:
                #         while_cond = False
                    
                #     j += 1
                
                df_stock['win_short'][i] = int(not(bool(win_long)))
                df_stock['win_long'][i] = win_long

                # for param in param_list:
                #     df_stock[param][i] = locals()[param]

            file = open(save_path, 'wb')
            pickle.dump(df_stock, file)
            file.close()
            print(f'File {save_path} save completed')

if __name__ == '__main__':

  print('Code begin')

  fig, ax1 = plt.subplots()
  ax2 = ax1.twinx()

  main()
  print('Code end')


# %%
