
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

    dataset_name = 'df_stocks_dict_2022-06-01_2023-06-05_1m.pkl'
    dataset_name_without_resolution = dataset_name[0:37]
    dataset_path = os.path.join(dirname, 'historical_data', dataset_name)
    file = open(dataset_path, 'rb')
    df_stocks_dict= pickle.load(file)
    file.close()
    stock_name_list = list(df_stocks_dict.keys())

    df_stocks_dict_new = pd.DataFrame(columns = {'open', 'high', 'low', 'close'})

    new_row = {}

    for stock in stock_name_list:
        df_stocks_dict[stock].dropna(inplace = True)
        print(f'Stock is {stock}')
        df_stocks_dict_new = pd.DataFrame(columns = {'open', 'high', 'low', 'close', 'pct', 'ha_pct', 'ha_colour'})

        list_5m = list(range(0,65,5))
        list_10m = list(range(0,65,10))
        
        for i in tqdm(range(0, df_stocks_dict[stock].shape[0] - candle_resolution)):

            if df_stocks_dict[stock].index[i].minute in list_10m:

                time = df_stocks_dict[stock].index[i]

                new_row['open'] = df_stocks_dict[stock]['open'][i]
                new_row['high'] = df_stocks_dict[stock]['high'][i : i + candle_resolution].max()
                new_row['low'] =  df_stocks_dict[stock]['low'][i : i + candle_resolution].min()
                new_row['close'] = df_stocks_dict[stock]['close'][i + candle_resolution - 1]

                df_stocks_dict_new.loc[time] = new_row

        df_stocks_dict_new['pct'] = np.where(df_stocks_dict_new['open'] < df_stocks_dict_new['close'],  
                        (df_stocks_dict_new['close'] / df_stocks_dict_new['open'] - 1) * 100,
                        -(df_stocks_dict_new['open'] / df_stocks_dict_new['close'] - 1) * 100
        )

        df_stocks_dict_new = f2.get_heiken_ashi_v2(df_stocks_dict_new)

        df_stocks_dict[stock]  = df_stocks_dict_new
        df_stocks_dict[stock].dropna(inplace = True)
        print(df_stocks_dict[stock])

    save_path = os.path.join(dirname, 'historical_data', f'{dataset_name_without_resolution}{candle_resolution}m.pkl')
    file = open(save_path, 'wb')
    pickle.dump(df_stocks_dict, file)
    print(df_stocks_dict)

print('Start')       
main()      