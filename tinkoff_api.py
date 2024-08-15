import os
from datetime import timedelta, timezone
from zoneinfo import ZoneInfo
import pandas as pd

from tinkoff.invest import CandleInterval, Client
from tinkoff.invest.schemas import CandleSource
from tinkoff.invest.utils import now
from personal_settings import personal_settings as ps
from tinkoff.invest import Client, InstrumentStatus, SharesResponse, InstrumentIdType
from tinkoff.invest.services import InstrumentsService, MarketDataService

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
                from_=now() - timedelta(days=60),
                interval=CandleInterval.CANDLE_INTERVAL_HOUR
            ))
        columns = ['time','open', 'high', 'low', 'close']
        df_candles = pd.DataFrame(columns=columns)
        for candle in candles_list:
            open = self._convert_tf_price(candle.open)
            high = self._convert_tf_price(candle.high)
            low = self._convert_tf_price(candle.low)
            close = self._convert_tf_price(candle.close)
            time = candle.time
            time = time.astimezone(ZoneInfo('US/Eastern'))
            
            df_candles = pd.concat([pd.DataFrame([[time, open, high, low, close]], columns=columns), df_candles], ignore_index=True)

        return df_candles
    
    def get_last_candle_close_price(self, ticker, days):
        instrument_id = self.get_instrument_id(ticker)        
        with Client(TOKEN) as client:
            candels_list = list(client.get_all_candles(
                instrument_id=instrument_id,
                from_=now() - timedelta(days=1),
                interval=CandleInterval.CANDLE_INTERVAL_1_MIN
            ))
        candel = candels_list[-1]
        close_price = self._convert_tf_price(candel.close)
        return close_price
    

if __name__ == '__main__':
    ticker = 'AMD'
    ta = Tinkoff_API()
    df = ta.get_hours_candles(ticker)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(df)




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