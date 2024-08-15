from personal_settings import personal_settings as ps
import requests
import pandas as pd
import json


APIKEY= ps.alpha_vantage

# replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key
# url = 'https://www.alphavantage.co/query?function=HISTORICAL_OPTIONS&symbol=IBM&apikey=1M5TTUF2IZCJ3SNO&date=2023-11-15&datatype=csv'
# r = requests.get(url)
# data = r.json()

# print(r)

# https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=IBM&interval=60min&month=2009-01&outputsize=full&apikey=1M5TTUF2IZCJ3SNO



def get_montly_data(symbol, month, interval = '60min'):
  function = 'TIME_SERIES_INTRADAY'
  url = 'https://www.alphavantage.co/query?function={}&symbol={}&interval={}&month={}&outputsize=full&apikey={}'.format(
    function,
    symbol,
    interval,
    month,
    APIKEY
  )
  r = requests.get(url)
  data = r.json()
  data = data['Time Series (60min)']
  with open('temp.json', 'w') as json_file:
    json.dump(data, json_file, indent=4)
  
  df = pd.read_json('temp.json').T
  for column in df.columns:
    df.rename(columns={column: column.split('*. ')[1]})
  return df

print(get_montly_data('AAPL', '2024-01'))

 
