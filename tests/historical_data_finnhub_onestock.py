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
    folder_name = 'historical_data\\onestock\\5m_raw_test_start_20230716'
    # stock_name_list  = ['V', 'AMD', 'AAPL', 'MSFT', 'MA', 'NVDA']

    # stock_name_list = ['JPM', 'KO', 'AMD', 'NKE', 'JD', 'SPG', 'K', 'HOLX', 'MMM', 'T', \
    #                'CAT', 'XOM', 'HPQ', 'INTC', 'MA', 'V', 'BA', 'AAPL', 'BAC', 'F', \
    #                'MSFT', 'IBM', 'EBAY', 'CRM', 'NVDA', 'DIS', 'CSCO', 'TTE']
    
    # stock_name_list = ['GOOGL', 'DIS', 'GME', 'ACB', 'PTGX']
    # stock_name_list += ['ABBV', 'AMZN', 'ALL', 'ARR']
    # stock_name_list += ['AC', 'AI', 'ADNT', 'AGS', 'AQN']

    stock_name_list = ['AAPL' ,'MSFT','AMZN', 'NVDA','TSLA','GOOGL', 'META','BRK.B','GOOG','JPM','XOM','UNH','JNJ','V','AVGO','PG','LLY','MA','HD','CVX','MRK', 
                       'PEP','COST','ABBV','ADBE','KO','CRM','WMT','MCD','CSCO','BAC','PFE','TMO','ACN','NFLX','ABT','AMD','LIN','ORCL','CMCSA',
                       'TXN','DIS','WFC','DHR','PM','NEE','VZ','INTC','RTX','HON','LOW','UPS','INTU','SPGI','NKE','COP','QCOM','BMY','CAT','UNP','BA','ISRG',
                        'GE','IBM','AMGN','AMAT','MDT','SBUX','PLD','NOW','MS','DE','BLK','GS','T','LMT','AXP','BKNG','SYK','ADI','TJX','ELV','MDLZ','GILD','ADP','MMC',
                        'C','AMT','CVS','VRTX','SCHW','LRCX','MO','TMUS','SLB', 'ETN', 'ZTS', 'CI', 'PYPL']

    stock_name_list += ['FI','CB','SO','REGN','BSX','EQIX','BDX','PANW','DUK','EOG','MU','AON','ITW','CSX','SNPS','PGR','APD','KLAC','CME','NOC','CDNS','ICE','ATVI',
                       'CL','SHW','WM','HCA','TGT','FCX','FDX','F','ORLY','MMM','CMG','EW','GM','MCK','NXPI','MCO','NSC','HUM','EMR','DXCM','PNC','PH','MPC','APH',
                       'ROP','FTNT','MCHP','PXD','USB','CCI','MAR','MSI','GD','PSA','JCI','PSX','SRE','ADSK','AZO','TDG','ECL','AJG','KMB','TEL','TT','AEP','EL','PCAR',
                       'OXY','TFC','CARR','D','IDXX','GIS','ON','COF','ADM','MNST','NUE','CTAS','AIG','EXC','VLO','MRNA','ANET','WMB','O','STZ','IQV','HLT','CHTR','WELL',
                       'BIIB','SPG','MSCI','DHI','ROK']
    
    stock_name_list = ['APH',   'TEL',  'BRK.B',  'BLK',  'MMC',  'HON',  'ECL',  'MSFT',  'AAPL',  'CTAS',  'JCI',  'ITW',  'ADP',  'ACN',  'EMR',  'MSI',  'DIS', 
             'ISRG',  'ROP',  'ETN',  'AXP',  'LIN',  'ADI',  'AVGO',  'INTU',  'MCO',  'TXN',  'PH',  'SBUX',  'CSCO',  'AJG',  'NXPI',  'HD',  'JPM',  'CDNS', 
             'ROK',  'QCOM',  'MCHP',  'SPG',  'CARR',  'COST',  'SYK',  'IQV',  'TT',  'GS',  'MSCI',  'TDG',  'GOOGL',  'V',  'ADSK']

    resolution = '5'
    # start_date = date(2021, 6, 6)
    # end_date  = date(2023, 7, 15)
    start_date = date(2023, 7, 16)
    end_date = date.today()
    # code
    
    parent_path = pathlib.Path(__file__).parent
    folder_path = pathlib.Path.joinpath(parent_path, folder_name)
    if not(os.path.exists(folder_path)):
        os.makedirs(folder_path)

    for stock in stock_name_list:
        file_path = pathlib.Path.joinpath(folder_path, f'df_{stock}_{start_date}_{end_date}_{resolution}m.pkl')
        print(f'stock is {stock}')
        if not(os.path.exists(file_path)):
            df_stock = get_historical_df(ticker = stock, resolution = resolution , start_date = start_date, end_date = end_date)
            # df_stock =  df_stock.between_time('9:30', '15:59')
            file = open(file_path, 'wb')
            pickle.dump(df_stock, file)
            file.close()
            print(f'File {file_path} save completed')

# print(f'df shape is {df_stocks_dict}')
# df = get_historical_df(start_date=date(2023, 4, 1), end_date=date(2023, 6, 1))
# print(df.shape)
# print(df)

