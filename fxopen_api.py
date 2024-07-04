API_KEY = 'eyJhbGciOiJFUzI1NiIsIng1dCI6IjI3RTlCOTAzRUNGMjExMDlBREU1RTVCOUVDMDgxNkI2QjQ5REEwRkEifQ.eyJvYWEiOiI3Nzc3NSIsImlzcyI6Im9hIiwiYWlkIjoiMTA5IiwidWlkIjoiRjRtQ1pqQ2RVVzFGdWlRZmVUTUJ2Zz09IiwiY2lkIjoiRjRtQ1pqQ2RVVzFGdWlRZmVUTUJ2Zz09IiwiaXNhIjoiRmFsc2UiLCJ0aWQiOiIyMDAyIiwic2lkIjoiYWUxY2FiODg5NThhNGQzZjllYzhkNGYxMzQ3MGM1NzUiLCJkZ2kiOiI4NCIsImV4cCI6IjE3Mjc2MDEyODAiLCJvYWwiOiIxRiIsImlpZCI6IjdmZjFiYWYyZTYwZTRhMTgzNWUyMDhkYzk3ZjRkMjEzIn0.Kpi9lTTMDbEv0DteXeq_mUe1E5LooOAHR_e0RO2C7ilGgkZ82hUNM6h23wfyl19wnFzD-qj3AxTpre5Q_gwReQ'
from saxo_openapi import API
import saxo_openapi.endpoints.rootservices as rs
from pprint import pprint
import saxo_openapi.endpoints.trading as tr
import saxo_openapi.endpoints.portfolio as pf
from saxo_openapi.contrib.orders import tie_account_to_order, MarketOrderFxSpot
from saxo_openapi.contrib.session import account_info
import saxo_openapi.endpoints.chart as chart
import json

token = API_KEY
client = API(access_token=token)

# lets make a diagnostics request, it should return '' with a state 200
# r = rs.diagnostics.Get()
# print("request is: ", r)
# rv = client.request(r)
# assert rv is None and r.status_code == 200
# print('diagnostics passed')

# # request available rootservices-features
# r = rs.features.Availability()
# rv = client.request(r)
# print("request is: ", r)
# print("response: ")
# pprint(rv, indent=2)
# print(r.status_code)

# ai = account_info(client)
# print(ai)

# params = {
#   'AssetType': 'Stock',
#   'Horizon' : '60',
#   'Uic' : '211'
#           }
# r = chart.charts.GetChartData(params=params)
# print(r)
# res = client.request(r)
# print(json.dumps(rv, indent=2))


    # >>> import saxo_openapi
    #     >>> import saxo_openapi.endpoints.chart as chart
    #     >>> import json
    #     >>> client = saxo_openapi.API(access_token=...)
    #     >>> params = {_v3_GetChartData_params}
    #     >>> r = chart.charts.GetChartData(params=params)
    #     >>> client.request(r)
    #     >>> print(json.dumps(rv, indent=2))

  ### saxo-get-uic.py v1.0 ###

### imports section ###  
import requests
import json
### end # imports section ### 

### Application configuration ###
def load_config(filename='config.json'):  # Load application configuration from a JSON file
    """Load application configuration from a JSON file."""
    try:
        with open(filename, 'r') as file:
            return json.load(file)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return None

# config = load_config()
# if config is None:
#     exit("Failed to load configuration, terminating program.")

# app_data = {
#     "AppName": "saxo-get-uic",
#     "AppKey": config.get("AppKey", ""),
#     "OpenApiBaseUrl": "https://gateway.saxobank.com/sim/openapi/"
# }
app_data = {
    "AppName": "saxo-get-uic",
    "AppKey": API_KEY,
    "OpenApiBaseUrl": "https://gateway.saxobank.com/sim/openapi/"
}
### end # Application configuration ###

def get_historical_data(token, ticker):
  """Resolve a trading symbol to its UIC using Saxo Bank's OpenAPI."""
  url = f"{app_data['OpenApiBaseUrl']}chart/v1/charts"
  headers = {"Authorization": f"Bearer {API_KEY}"}
  params = {"AssetType": 'FxSpot',
            "Horizon" : 60,
            "Uic" : 16}
  try:
      response = requests.get(url, headers=headers, params=params)
      response.raise_for_status()  # Raises an HTTPError if the response was an error
      data = response.json()
      if 'Data' in data and data['Data']:
          return data['Data'][0]['Identifier']
  except requests.RequestException as e:
      print(f"Error resolving UIC for symbol: {ticker}. Error: {e}")
  return None
get_historical_data(token, 'AAPL')

### Function to resolve symbol to UIC ###
def resolve_symbol_to_uic(access_token, symbol):
    """Resolve a trading symbol to its UIC using Saxo Bank's OpenAPI."""
    url = f"{app_data['OpenApiBaseUrl']}ref/v1/instruments"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    params = {"Keywords": symbol, "AssetTypes": "Stock"}
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()  # Raises an HTTPError if the response was an error
        data = response.json()
        if 'Data' in data and data['Data']:
            return data['Data'][0]['Identifier']
    except requests.RequestException as e:
        print(f"Error resolving UIC for symbol: {symbol}. Error: {e}")
    return None
### end # Function to resolve symbol to UIC ###

### Main program ###
if __name__ == "__main__":
    pass
    
    symbol = "AAPL"  # STOCK SYMBOL TO GET UIC FOR 
    # access_token = config.get("AccessToken", token)
    access_token = API_KEY
    if not access_token:
        print("Access token is missing in the configuration.")
        exit()
        
    uic = resolve_symbol_to_uic(access_token, symbol)
    if uic:
        print(f"Resolved UIC for {symbol}: {uic}")
    else:
        print(f"Failed to resolve UIC for symbol: {symbol}")
### end # Main program ###

### end # saxo-get-uic.py v1.0 ###
### support@sugra.systems ###