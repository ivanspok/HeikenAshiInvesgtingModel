
#%%
'''
preparing data set - personal index investing strategy
'''
#region 

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
else:
  from torgach import settings
  from torgach import schedule as sch
  from torgach.personal_settings import personal_settings as ps
  from torgach.functions import SMA, WMA
  from torgach.colog import colog

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


def main():
    
    candle_resolution = 5
    # window = 10 

    folder_load_path = os.path.join(dirname, 'historical_data\\onestock\\1m')

    dataset_name = 'df_stocks_dict_2022-06-01_2023-06-05_1m.pkl'
    dataset_name_without_resolution = dataset_name[0:37]
    
    folder_save_path = os.path.join(dirname, f'historical_data\\onestock\\{candle_resolution}m')
    if not(os.path.exists(folder_save_path)):
        os.mkdir(folder_save_path)

    list_5m = list(range(0,65,5))
    list_10m = list(range(0,65,10))

    for item in os.listdir(folder_load_path):
        

        dataset_path = os.path.join(folder_load_path, item)
        print(f'Dataset is {dataset_path}')

        x = item.find('1m.pkl')
        item_name_without_resolution = item[:x]
        save_path = os.path.join(folder_save_path, f'{item_name_without_resolution}{candle_resolution}m.pkl')
        if not(os.path.exists(save_path)):

            file = open(dataset_path, 'rb')
            df_stock = pickle.load(file)
            file.close()

            df_stock = df_stock.between_time('9:30', '15:59')

            # df_stock_new = pd.DataFrame(columns = {'open', 'high', 'low', 'close'})
            new_row = {}
            df_stock_new = pd.DataFrame(columns = {'open', 'high', 'low', 'close', 'pct', 'ha_pct', 'ha_colour'})

            for i in tqdm(range(0, df_stock.shape[0] - candle_resolution)):
            
                time_cond = False
                if candle_resolution == 5 and df_stock.index[i].minute in list_5m: time_cond = True
                if candle_resolution == 10 and df_stock.index[i].minute in list_10m: time_cond = True

                if time_cond:

                    time = df_stock.index[i]

                    new_row['open'] = df_stock['open'][i]
                    new_row['high'] = df_stock['high'][i : i + candle_resolution].max()
                    new_row['low'] =  df_stock['low'][i : i + candle_resolution].min()
                    new_row['close'] = df_stock['close'][i + candle_resolution - 1]

                    df_stock_new.loc[time] = new_row
            
            df_stock_new = df_stock_new.between_time('9:30', '15:59')
            df_stock_new['pct'] = np.where(df_stock_new['open'] < df_stock_new['close'],  
                            (df_stock_new['close'] / df_stock_new['open'] - 1) * 100,
                            -(df_stock_new['open'] / df_stock_new['close'] - 1) * 100
            )

            df_stock_new = f2.get_heiken_ashi_v2(df_stock_new)

            df_stock  = df_stock_new
            df_stock.dropna(inplace = True)
            
            file = open(save_path, 'wb')
            pickle.dump(df_stock, file)
            file.close()
            print(df_stock)

print('Start')       
main()
print('Finish')