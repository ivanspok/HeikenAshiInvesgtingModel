import os
# from datetime import timedelta, timezone, datetime

from datetime import datetime, timedelta, tzinfo
from datetime import date, timezone

from zoneinfo import ZoneInfo
import pandas as pd

from tinkoff.invest import CandleInterval, Client
from tinkoff.invest.schemas import CandleSource
# from tinkoff.invest.utils import now
from personal_settings import personal_settings as ps
from tinkoff.invest import Client, InstrumentStatus, SharesResponse, InstrumentIdType
from tinkoff.invest.services import InstrumentsService, MarketDataService
import numpy as np
import pickle
import functions_2 as f2
import os, pathlib



TOKEN = ps.tinkoff

class Tinkoff_API():

    def __init__(self):
        pass

    def _convert_tf_price(self, value):
        return value.units + value.nano / 10**9
    
    def get_instrument_id(self, ticker):
        df = pd.read_csv('tickers_list.csv')
        figi = df[df['ticker'] == ticker ]['figi'].iloc[0]
        return figi

    def get_hours_candles(self, ticker):
        instrument_id = self.get_instrument_id(ticker)
        with Client(TOKEN) as client:
            candles_list = list(client.get_all_candles(
                instrument_id=instrument_id,
                from_= datetime.now(datetime.UTC) - timedelta(days=60),
                interval=CandleInterval.CANDLE_INTERVAL_HOUR
            ))
        columns = ['time','open', 'high', 'low', 'close']
        df = pd.DataFrame(columns=columns)
        for candle in candles_list:
            open = self._convert_tf_price(candle.open)
            high = self._convert_tf_price(candle.high)
            low = self._convert_tf_price(candle.low)
            close = self._convert_tf_price(candle.close)
            time = candle.time
            time = time.astimezone(ZoneInfo('US/Eastern'))
            
            df = pd.concat([pd.DataFrame([[time, open, high, low, close]], columns=columns), df], ignore_index=True)
        
        # df['pct'] = np.where(df['open'] < df['close'],  
        #                  (df['close'] / df['open'] - 1) * 100,
        #                  -(df['open'] / df['close'] - 1) * 100
        # )
        # df = f2.get_heiken_ashi_v2(df)
        # df = df[['open', 'high', 'low', 'close', 'pct', 'ha_pct', 'ha_colour']]

        return df
    
    def get_minutes_candles(self, ticker, days):
        instrument_id = self.get_instrument_id(ticker)
        with Client(TOKEN) as client:
            candles_list = list(client.get_all_candles(
                instrument_id=instrument_id,
                from_=datetime.now() - timedelta(days=days),
                to = datetime.now() - timedelta(days=days-1),
                interval=CandleInterval.CANDLE_INTERVAL_1_MIN
            ))
        columns = ['time','open', 'high', 'low', 'close', 'volume']
        df = pd.DataFrame(columns=columns)
        for candle in candles_list:
            open = self._convert_tf_price(candle.open)
            high = self._convert_tf_price(candle.high)
            low = self._convert_tf_price(candle.low)
            close = self._convert_tf_price(candle.close)
            time = candle.time
            time = time.astimezone(ZoneInfo('US/Eastern'))
            
            # df_add = pd.DataFrame([[time, open, high, low, close]], columns=columns)
            if open != 0:
                df.loc[df.shape[0]] = [time, open, high, low, close]
        # if not df.empty:
        #     df['pct'] = np.where(df['open'] < df['close'],  
        #                     (df['close'] / df['open'] - 1) * 100,
        #                     -(df['open'] / df['close'] - 1) * 100
        #     )
        #     df = f2.get_heiken_ashi_v2(df)
        #     df = df[['time','open', 'high', 'low', 'close', 'pct', 'ha_pct', 'ha_colour']]
        #     df = df.set_index('time')

        return df

    
    def get_last_candle_close_price(self, ticker, days=1):
        instrument_id = self.get_instrument_id(ticker)        
        with Client(TOKEN) as client:
            candels_list = list(client.get_all_candles(
                instrument_id=instrument_id,
                from_=datetime.now(timezone.utc) - timedelta(days=1),
                interval=CandleInterval.CANDLE_INTERVAL_1_MIN
            ))
        candel = candels_list[-1]
        close_price = self._convert_tf_price(candel.close)
        return close_price
    
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
    

if __name__ == '__main__':
    # ticker = 'AMD'
    # ta = Tinkoff_API()
    # # df = ta.get_hours_candles(ticker)
    # columns = ['time','open', 'high', 'low', 'close']
    # df = pd.DataFrame(columns=columns)
    # for days in range(90,1,-1):
    #     df_add = ta.get_minutes_candles(ticker, days=days)
    #     if not df_add.empty:
    #         df = pd.concat([df, df_add], ignore_index=True)
    #         # df = ta.get_minutes_candles(ticker, days=45)
    # print(df)
    ta = Tinkoff_API()
    folder_name = 'historical_data'
    period = 90 # 'max'
    interval = '1m' # 1m

    stock_name_list = []
    # 'GOOG','JPM'
    stock_name_list += ['XOM','UNH','JNJ','V','AVGO','PG','LLY','MA','HD','CVX','MRK', 
                       'PEP','COST','ABBV','ADBE','KO','CRM','WMT','MCD','CSCO','BAC','PFE','TMO','ACN','NFLX','ABT','AMD','LIN','ORCL','CMCSA',
                       'TXN','DIS','WFC','DHR','PM','NEE','VZ','INTC','RTX','HON','LOW','UPS','INTU','SPGI','NKE','COP','QCOM','BMY','CAT','UNP','BA','ISRG',
                        'GE','IBM','AMGN','AMAT','MDT','SBUX','PLD','NOW','MS','DE','BLK','GS','T','LMT','AXP','SYK','ADI','TJX','ELV','MDLZ','GILD','ADP','MMC',
                        'C','AMT','CVS','VRTX','SCHW','LRCX','MO','TMUS','SLB', 'ETN', 'ZTS', 'CI', 'PYPL']

    stock_name_list += ['FI','CB','SO','REGN','BSX','EQIX','BDX','PANW','DUK','EOG','MU','AON','ITW','CSX','SNPS','PGR','APD','KLAC','CME','NOC','CDNS','ICE',
                        'CL','SHW','WM','HCA','TGT','FCX','FDX','F','MMM','CMG','EW','GM','MCK','NXPI','MCO','NSC','HUM','EMR','DXCM','PNC','PH','MPC','APH',
                        'ROP','FTNT','MCHP','PXD','USB','CCI','MAR','MSI','GD','PSA','JCI','PSX','SRE','ADSK','AZO','TDG','ECL','AJG','KMB','TEL','TT','AEP','EL','PCAR',
                        'OXY','TFC','CARR','D','IDXX','GIS','ON','COF','ADM','MNST','NUE','CTAS','AIG','EXC','VLO','MRNA','ANET','WMB','O','STZ','IQV','HLT','CHTR','WELL',
                        'BIIB','SPG','MSCI','DHI','ROK']
    
    columns = ['open', 'high', 'low', 'close', 'pct', 'ha_pct', 'ha_colour']
    for ticker in stock_name_list:
        try:
            print(f'ticker is {ticker}')
            # close_price = ta.get_last_candle_close_price(ticker, days=1)
            # print(close_price)
            # df = pd.DataFrame(columns=columns)
            # for days in range(period,1,-1):
            #     df_add = ta.get_minutes_candles(ticker, days=days)
            #     if not df_add.empty:
            #         df = pd.concat([df, df_add])
            # print(df)
            # save_df(df, ticker, period, interval, folder_name) 
            df_add = ta.get_minutes_candles(ticker, days=1)  
            print(df_add['close'].iloc[-1]) 
        except Exception as e:
            print(e)  
            

    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    #     print(df)




# def run():
#     with Client(TOKEN) as cl:
#         instruments: InstrumentsService = cl.instruments
#         market_data: MarketDataService = cl.market_data
 
#         # r = instruments.share_by(id_type=InstrumentIdType.INSTRUMENT_ID_TYPE_FIGI, id="BBG004S683W7")
#         # print(r)
 
#         l = []
#         for method in ['shares']: # , 'currencies', 'futures']:
#             for item in getattr(instruments, method)().instruments:
#                 l.append({
#                     'ticker': item.ticker,
#                     'figi': item.figi,
#                     'type': method,
#                     'name': item.name,
#                     'country_of_rick': item.country_of_risk,
#                 })
 
#         df = pd.DataFrame(l)
#         # print(df)
#         # df.to_json()
#         df.to_csv('tickers_list.csv')
#         with pd.option_context('display.max_rows', 3000, 'display.max_columns', 3000):  # more options can be specified also
#             print(df)
 
#         df = df[df['ticker'] == TICKER]
#         if df.empty:
#             print(f"Нет тикера {TICKER}")
#             return
#         print(df['figi'].iloc[0])