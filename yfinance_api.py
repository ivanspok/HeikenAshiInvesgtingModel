import yfinance as yf
# import matplotlib

import datetime, time
import pandas as pd
pd.options.mode.chained_assignment = None 

from personal_settings import personal_settings as ps
from datetime import date, timedelta
import os, pathlib
import numpy as np
import pickle
import functions_2 as f2

import warnings
warnings.filterwarnings('error')
# Setup client

def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)

def save_df(stock_df, ticker, period, interval, folder_name):
    parent_path = pathlib.Path(__file__).parent

    folder_path = pathlib.Path.joinpath(parent_path, folder_name)
    if not(os.path.isdir(folder_path)):
        os.mkdir(folder_path)

    folder_path = pathlib.Path.joinpath(folder_path, f'period{period}interval{interval}')
    if not(os.path.isdir(folder_path)):
        os.mkdir(folder_path)

    file_path = pathlib.Path.joinpath(folder_path, f'df_stock_{ticker}_period{period}_interval{interval}.pkl')
    file = open(file_path, 'wb')
    pickle.dump(stock_df, file)
    file.close()
    print(stock_df)
    print('File save completed')

# Stock candles
def get_historical_df(ticker='AAPL', interval='1h', period='2y', start_date=date.today(), end_date=date.today()):
    
  
    insturument = yf.Ticker(ticker)
    df = insturument.history(period=period, interval=interval)

    df = df.rename(columns={"Open": "open", "Close" :'close', "High" : 'high', "Low": 'low'})

    df['pct'] = np.where(df['open'] < df['close'],  
                         (df['close'] / df['open'] - 1) * 100,
                         -(df['open'] / df['close'] - 1) * 100
    )
    # df.index = pd.to_datetime(df['t'], unit='s', utc=True).map(lambda x: x.tz_convert('America/New_York'))
    df = f2.get_heiken_ashi_v2(df)
    df = df[['open', 'high', 'low', 'close', 'pct', 'ha_pct', 'ha_colour']]

    return df 

if __name__ == '__main__':
   
    print('START GETTING DATA')
    request_number = 0 
    df_stocks_dict = {}
    # local parameters:
    folder_name = 'historical_data'
    # stock_name_list  = ['AAPL','V', 'AMD', 'MSFT', 'MA', 'NVDA']
    # stock_name_list  = ['GOOGL','CVS', 'AMZN', 'CRM', 'DIS', 'MCD']
 
    # stock_name_list = ['AAPL' ,'MSFT','AMZN', 'NVDA','TSLA','GOOGL', 'META']
    stock_name_list = []
    stock_name_list += ['GOOG','JPM','XOM','UNH','JNJ','V','AVGO','PG','LLY','MA','HD','CVX','MRK', 
                       'PEP','COST','ABBV','ADBE','KO','CRM','WMT','MCD','CSCO','BAC','PFE','TMO','ACN','NFLX','ABT','AMD','LIN','ORCL','CMCSA',
                       'TXN','DIS','WFC','DHR','PM','NEE','VZ','INTC','RTX','HON','LOW','UPS','INTU','SPGI','NKE','COP','QCOM','BMY','CAT','UNP','BA','ISRG',
                        'GE','IBM','AMGN','AMAT','MDT','SBUX','PLD','NOW','MS','DE','BLK','GS','T','LMT','AXP','BKNG','SYK','ADI','TJX','ELV','MDLZ','GILD','ADP','MMC',
                        'C','AMT','CVS','VRTX','SCHW','LRCX','MO','TMUS','SLB', 'ETN', 'ZTS', 'CI', 'PYPL']

    # stock_name_list += ['FI','CB','SO','REGN','BSX','EQIX','BDX','PANW','DUK','EOG','MU','AON','ITW','CSX','SNPS','PGR','APD','KLAC','CME','NOC','CDNS','ICE',
    #                    'CL','SHW','WM','HCA','TGT','FCX','FDX','F','ORLY','MMM','CMG','EW','GM','MCK','NXPI','MCO','NSC','HUM','EMR','DXCM','PNC','PH','MPC','APH',
    #                    'ROP','FTNT','MCHP','PXD','USB','CCI','MAR','MSI','GD','PSA','JCI','PSX','SRE','ADSK','AZO','TDG','ECL','AJG','KMB','TEL','TT','AEP','EL','PCAR',
    #                    'OXY','TFC','CARR','D','IDXX','GIS','ON','COF','ADM','MNST','NUE','CTAS','AIG','EXC','VLO','MRNA','ANET','WMB','O','STZ','IQV','HLT','CHTR','WELL',
    #                    'BIIB','SPG','MSCI','DHI','ROK']

    # stock_name_list = ['PXD']
    
    period = '2y'
    interval = '1h'
    start_date = date(2022, 5, 22)
    end_date  =  date.today()
 
    # code
    for ticker in stock_name_list:
        print(f'Stock is {ticker}')
        try:
            stock_df = get_historical_df(ticker = ticker, period=period, interval=interval, start_date = start_date, end_date = end_date)
            df_stocks_dict[ticker] = stock_df
            save_df(stock_df, ticker, period, interval, folder_name)
        except Exception as e:
            print(e)
        # print(df_stocks_dict[stock].shape)

# :Parameters:
#             period : str
#                 Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
#                 Either Use period parameter or use start and end
#             interval : str
#                 Valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
#                 Intraday data cannot extend last 60 days
#             start: str
#                 Download start date string (YYYY-MM-DD) or _datetime, inclusive.
#                 Default is 99 years ago
#                 E.g. for start="2020-01-01", the first data point will be on "2020-01-01"
#             end: str
#                 Download end date string (YYYY-MM-DD) or _datetime, exclusive.
#                 Default is now
#                 E.g. for end="2023-01-01", the last data point will be on "2022-12-31"