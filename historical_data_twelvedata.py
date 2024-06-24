import finnhub
import datetime, time
import pandas as pd
pd.options.mode.chained_assignment = None 

from personal_settings import personal_settings as ps
from datetime import date, timedelta
import os, pathlib
import numpy as np
import pickle
import functions_2 as f2
global request_number

# Setup client
finnhub_client = finnhub.Client(api_key=ps.finnhub_token)

def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)

# Stock candles
def get_historical_df(ticker = 'AAPL', resolution = '15', start_date = date.today() , end_date=date.today()):
    global request_number
    df = pd.DataFrame()

    counter = 0
    for single_date in daterange(start_date, end_date):
        
        if counter ==0:
            counter += 1

            _from = int(time.mktime(single_date.timetuple()))
            to = int(time.mktime((single_date + timedelta(days=30)).timetuple()))
            request_number += 1
            if request_number > 29:
                time.sleep(5)
                print('sleep, place 1')
                request_number = 0
            success = False
            while not success:
                try:
                 res = finnhub_client.stock_candles(ticker, resolution, _from, to)
                 success = True
                except Exception as exc:
                 print(f'{exc}::: sleep, place 2')
                 time.sleep(10)
            
            try:
                res_df = pd.DataFrame(res)
                print(f'res_df shape is {res_df.shape}, _from is {_from} to is {to} ')
                df = pd.concat([df, res_df]) 
            except:
                print(f'from {_from} to {to} there is no data')
            
        else:
            counter += 1

        if counter == 30 : counter = 0

    df = df.rename(columns={"o": "open", "c" :'close', "h" : 'high', "l": 'low'})

    df['pct'] = np.where(df['open'] < df['close'],  
                         (df['close'] / df['open'] - 1) * 100,
                         -(df['open'] / df['close'] - 1) * 100
    )

    df.index = pd.to_datetime(df['t'], unit='s', utc=True).map(lambda x: x.tz_convert('America/New_York'))
    df = f2.get_heiken_ashi_v2(df)

    df = df[['open', 'high', 'low', 'close', 'pct', 'ha_pct', 'ha_colour']]

    return df 

if __name__ == '__main__':
    print('START')
    request_number = 0 
    df_stocks_dict = {}
    # local parameters:
    folder_name = 'historical_data'
    # stock_name_list  = ['V', 'AMD', 'AAPL', 'MSFT', 'MA', 'NVDA']
    stock_name_list  = ['AAPL','V', 'AMD', 'MSFT', 'MA', 'NVDA']
    resolution = '1'
    start_date = date(2024, 5, 22)
    end_date  = date(2024, 6, 21)
    # code
    for stock in stock_name_list:
        print(f'stock is {stock}')
        stock_df = get_historical_df(ticker = stock, resolution = resolution , start_date = start_date, end_date = end_date)

        df_stocks_dict[stock] = stock_df
        print(df_stocks_dict[stock].shape)

    df_merge = pd.DataFrame()
    for stock in stock_name_list:
        if df_merge.shape == (0, 0):
            df_merge = df_stocks_dict[stock]
        else:
            df_merge, _ =  df_merge.align(df_stocks_dict[stock], join = 'inner', axis=0)
    df_merge.drop_duplicates()

           # if len(df_stocks_dict.keys()) > 0:
        #     df_merge_left, df_merge_right = df_stocks_dict[list(df_stocks_dict.keys())[-1]].align(stock_df, join = 'inner')
    for stock in stock_name_list:
      df_stocks_dict[stock]  =  df_stocks_dict[stock].loc[df_merge.index]
      df_stocks_dict[stock] = df_stocks_dict[stock][~df_stocks_dict[stock].index.duplicated(keep='first')]
      df_stocks_dict[stock] = df_stocks_dict[stock].sort_index()
      print(f'final shape for {stock} is {df_stocks_dict[stock].shape}')
      df_stocks_dict[stock] = df_stocks_dict[stock].dropna()
    
    parent_path = pathlib.Path(__file__).parent
    folder_path = pathlib.Path.joinpath(parent_path, folder_name)
    file_path = pathlib.Path.joinpath(folder_path, f'df_stocks_dict_{start_date}_{end_date}_{resolution}m.pkl')
    file = open(file_path, 'wb')
    pickle.dump(df_stocks_dict, file)
    file.close()
    print(df_stocks_dict)
    print('File save completed')
    # print(f'df shape is {df_stocks_dict}')
# df = get_historical_df(start_date=date(2023, 4, 1), end_date=date(2023, 6, 1))
# print(df.shape)
# print(df)

