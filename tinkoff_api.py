import os
from datetime import timedelta
import pandas as pd

from tinkoff.invest import CandleInterval, Client
from tinkoff.invest.schemas import CandleSource
from tinkoff.invest.utils import now
from personal_settings import personal_settings as ps

TOKEN = ps.tinkoff

# import pycurl
# # import StringIO

# # response = StringIO.StringIO()
# c = pycurl.Curl()
# c.setopt(c.URL, 'https://invest-public-api.tinkoff.ru/history-data?figi=BBG00QKJSX05&year=2022')
# # c.setopt(c.WRITEFUNCTION, response.write)
# c.setopt(c.header, ps.tinkoff)

# c.perform()
# c.close()
# print(response.getvalue())
# response.close()

# curl -X GET --location "https://invest-public-api.tinkoff.ru/history-data?figi=BBG00QKJSX05&year=2022" \
# -H "Authorization: Bearer token"

import requests
response = requests.get('https://invest-public-api.tinkoff.ru/history-data?figi=BBG000B9XRY4&year=2022')
print(response)


def main():
    df = pd.DataFrame()
    client = Client(TOKEN)
    with Client(TOKEN) as client:
        for candle in client.get_all_candles(
            instrument_id="BBG000B9XRY4",
            from_=now() - timedelta(days=10),
            interval=CandleInterval.CANDLE_INTERVAL_1_MIN
        ):
            print(candle)
            # df.append(candle)




# if __name__ == "__main__":
#     main()

# from tinkoff.invest import Client, InstrumentStatus, SharesResponse, InstrumentIdType
# from tinkoff.invest.services import InstrumentsService, MarketDataService

# TICKER = "AAPL"
# def run():
#     with Client(TOKEN) as cl:
#         instruments: InstrumentsService = cl.instruments
#         market_data: MarketDataService = cl.market_data
 
#         # r = instruments.share_by(id_type=InstrumentIdType.INSTRUMENT_ID_TYPE_FIGI, id="BBG004S683W7")
#         # print(r)
 
#         l = []
#         for method in ['shares', 'bonds', 'etfs']: # , 'currencies', 'futures']:
#             for item in getattr(instruments, method)().instruments:
#                 l.append({
#                     'ticker': item.ticker,
#                     'figi': item.figi,
#                     'type': method,
#                     'name': item.name,
#                 })
 
#         df = DataFrame(l)
#         print(df)
#         # df.to_json()
 
#         df = df[df['ticker'] == TICKER]
#         if df.empty:
#             print(f"Нет тикера {TICKER}")
#             return
#         print(df['figi'].iloc[0])
 
 
# if __name__ == '__main__':
#     print("** Hola Hey, Azzrael Code YT subs!!!\n")
#     run()