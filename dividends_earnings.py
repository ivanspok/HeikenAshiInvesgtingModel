# URLs for different event types
# URL_EARNINGS = "https://api.nasdaq.com/api/calendar/earnings"
# URL_DIVIDENDS = "https://api.nasdaq.com/api/calendar/dividends"
# URL_SPLITS = "https://api.nasdaq.com/api/calendar/splits"
# URL_IPO = "https://api.nasdaq.com/api/calendar/ipos"

import os, pathlib
import pickle
from finance_calendars import finance_calendars as fc # type: ignore
import pandas as pd
import datetime
import yfinance as yf
import requests
import unittest

from colog.colog import colog
c = colog()
warning = colog(TextColor='purple')
alarm = colog(TextColor='red')
blue = colog(TextColor='blue')

def get_next_days_dividends(days=7):
    today = datetime.date.today()
    all_events = []
    for i in range(days):
        date_ = today + datetime.timedelta(days=i)
        df = fc.get_dividends_by_date(date_)
        df["date"] = date_
        all_events.append(df)
    # Index(['date', 'companyName', 'dividend_Ex_Date', 'payment_Date',
    #        'record_Date', 'dividend_Rate', 'indicated_Annual_Dividend',
    #        'announcement_Date'],
    #       dtype='object')
    dividends = pd.concat(all_events, ignore_index=True)
    if not dividends.empty:    
        dividends['ticker'] = dividends['companyName'].apply(company_name_to_ticker)
        dividends['dividend_Ex_Date'] = pd.to_datetime(dividends['dividend_Ex_Date'])
    return dividends

def get_next_days_earnings(days=7):
    today = datetime.date.today()
    all_events = []
    for i in range(days):
        date_ = today + datetime.timedelta(days=i)
        df = fc.get_earnings_by_date(date_)
        df["date"] = date_
        all_events.append(df)

    # Index(['date', 'time', 'name', 'marketCap', 'fiscalQuarterEnding',
    #    'epsForecast', 'noOfEsts', 'lastYearRptDt', 'lastYearEPS'],
    #   dtype='object')
    earnings = pd.concat(all_events, ignore_index=True)
    if not earnings.empty:   
        earnings['ticker'] = earnings['name'].apply(company_name_to_ticker)
        earnings['date'] = pd.to_datetime(earnings['date'])
    return earnings

def get_ticker_from_yahoo(query):
    url = f"https://query1.finance.yahoo.com/v1/finance/search?q={query}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9"
    }
    response = requests.get(url, headers=headers)
    data = response.json()
    while True:
        if 'quotes' in data and len(data['quotes']) > 0:
            return data['quotes'][0]['symbol']
        else:
            parts = query.strip().split()
            if len(parts) > 1:
                query = ' '.join(parts[:-1])
                url = f"https://query1.finance.yahoo.com/v1/finance/search?q={query}"
                response = requests.get(url, headers=headers)
                data = response.json()
            else: 
                return 'No Ticker Found'

def company_name_to_ticker(name):
    try:
        name = name.replace('Common Stock', '').strip()
        result = yf.Search(name)
        if len(result.quotes) > 0:
            return result.quotes[0]['symbol']   # best match
        else:
            ticker = get_ticker_from_yahoo(name)
            if ticker:
                return ticker
            else:
                return 'No Ticker Found'
    except Exception as e:
        print(f"Error finding ticker for {name}: {e}")
        return 'Error Finding Ticker'
 
def save_file(df, file_path:str ):
    try:
      with open(file_path, 'wb') as file:
        pickle.dump(df, file)
        c.print(f'File (pkl) {file_path} has been saved', color='green')
        try:
          xlsx_path = str(pathlib.Path(file_path).with_suffix(".xlsx"))
          df.to_excel(xlsx_path)
          c.print('cvs file saved', color='green')
        except Exception as e:
          alarm.print(f'{e}')
    except Exception as e:
        alarm.print(f'{e}')

def check_dividends_conditions(df_dividends, ticker:str, ex_date=datetime.date.today()) -> bool:
    '''
    Check if dividents whould be paid next traiding day or was paid last 5 traiding days
    '''
    try:

        ex_date_dt = pd.to_datetime(ex_date)
        df = df_dividends[df_dividends['ticker'] == ticker]
        close_dividents = df[ 
            ((ex_date_dt - df['dividend_Ex_Date'] <= pd.Timedelta(days=5)) &
             (ex_date_dt + pd.Timedelta(days=1) >= df['dividend_Ex_Date']))
        ]
        
        if close_dividents.empty:
            return True
        else:
            return False
    except Exception as e:
        print(f"Error checking dividents for {ticker}: {e}")
        return True

def check_earnings_conditions(df_earnings, ticker:str, ex_date=datetime.date.today()) -> bool:
    '''
    Check if earnings whould be happen next traiding day or was happen last 5 traiding days
    '''
    try:
        ex_date_dt = pd.to_datetime(ex_date)
        df = df_earnings[df_earnings['ticker'] == ticker]
        close_earnings = df[ 
            ((ex_date_dt - df['date'] <= pd.Timedelta(days=5)) &
             (ex_date_dt + pd.Timedelta(days=1) >= df['date']))
        ]
        if close_earnings.empty:
            return True
        else:
            return False
    except Exception as e:
        print(f"Error checking earnings for {ticker}: {e}")
        return True

def run_dividends_events():
    
    print("Running dividents events module...")
    folder_name = 'divedents_earings_data'
    parent_path = pathlib.Path(__file__).parent
    folder_path = pathlib.Path.joinpath(parent_path, folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_name = 'divedents_data_7days_from_' + datetime.date.today().strftime('%Y%m%d')
    file_path = pathlib.Path.joinpath(folder_path, file_name + '.pkl')
    if not os.path.exists(file_path):
        df_dividends = get_next_days_dividends(7)
        save_file(df_dividends, file_path)
    else:
        print('Loading dividents data from file...')
        with open(file_path, 'rb') as file:
            df_dividends = pickle.load(file)
            print('Dividents data loaded from file')
    return df_dividends

def run_earnings_events():
    
    print("Running earings events module...")
    folder_name = 'divedents_earings_data'
    parent_path = pathlib.Path(__file__).parent
    folder_path = pathlib.Path.joinpath(parent_path, folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_name = 'earnings_data_7days_from_' + datetime.date.today().strftime('%Y%m%d')
    file_path = pathlib.Path.joinpath(folder_path, file_name + '.pkl')
    if not os.path.exists(file_path):
        df_earings = get_next_days_earnings(7)
        save_file(df_earings, file_path)
    else:
        print('Loading dividents data from file...')
        with open(file_path, 'rb') as file:
            df_earings = pickle.load(file)
            print('Dividents data loaded from file')
    return df_earings

class TestCheckDividentsConditions(unittest.TestCase):
    
    def test_OCCI_dividents(self):
        ticker = 'OCCI'
        ex_date = datetime.date(2026, 1, 13)
        expected_result = True
        result = check_dividends_conditions(df_dividends, ticker, ex_date)
        self.assertEqual(result, expected_result, f"Check for the ticker {ticker} on {ex_date} failed.")
        
        ex_date = datetime.date(2026, 1, 14)
        expected_result = False
        result = check_dividends_conditions(df_dividends, ticker, ex_date)
        self.assertEqual(result, expected_result, f"Check for the ticker {ticker} on {ex_date} failed.")

        ex_date = datetime.date(2026, 1, 15)
        expected_result = False
        result = check_dividends_conditions(df_dividends, ticker, ex_date)
        self.assertEqual(result, expected_result, f"Check for the ticker {ticker} on {ex_date} failed.")
       
        ex_date = datetime.date(2026, 1, 16)
        expected_result = False
        result = check_dividends_conditions(df_dividends, ticker, ex_date)
        self.assertEqual(result, expected_result, f"Check for the ticker {ticker} on {ex_date} failed.")

        ex_date = datetime.date(2026, 1, 17)
        expected_result = False
        result = check_dividends_conditions(df_dividends, ticker, ex_date)
        self.assertEqual(result, expected_result, f"Check for the ticker {ticker} on {ex_date} failed.")

        ex_date = datetime.date(2026, 1, 20)
        expected_result = False
        result = check_dividends_conditions(df_dividends, ticker, ex_date)
        self.assertEqual(result, expected_result, f"Check for the ticker {ticker} on {ex_date} failed.")
        
        ex_date = datetime.date(2026, 1, 21)
        expected_result = True
        result = check_dividends_conditions(df_dividends, ticker, ex_date)
        self.assertEqual(result, expected_result, f"Check for the ticker {ticker} on {ex_date} failed.")
    
    def test_SIFY_dividents(self):
        ticker = 'SIFY'
        ex_date = datetime.date(2026, 1, 10)
        expected_result = True
        result = check_earnings_conditions(df_earnings, ticker, ex_date)
        self.assertEqual(result, expected_result, f"Check for the ticker {ticker} on {ex_date} failed.")
        
        ex_date = datetime.date(2026, 1, 11)
        expected_result = False
        result = check_earnings_conditions(df_earnings, ticker, ex_date)
        self.assertEqual(result, expected_result, f"Check for the ticker {ticker} on {ex_date} failed.")

        ex_date = datetime.date(2026, 1, 12)
        expected_result = False
        result = check_earnings_conditions(df_earnings, ticker, ex_date)
        self.assertEqual(result, expected_result, f"Check for the ticker {ticker} on {ex_date} failed.")
       
        ex_date = datetime.date(2026, 1, 13)
        expected_result = False
        result = check_earnings_conditions(df_earnings, ticker, ex_date)
        self.assertEqual(result, expected_result, f"Check for the ticker {ticker} on {ex_date} failed.")

        ex_date = datetime.date(2026, 1, 14)
        expected_result = False
        result = check_earnings_conditions(df_earnings, ticker, ex_date)
        self.assertEqual(result, expected_result, f"Check for the ticker {ticker} on {ex_date} failed.")

        ex_date = datetime.date(2026, 1, 15)
        expected_result = False
        result = check_earnings_conditions(df_earnings, ticker, ex_date)
        self.assertEqual(result, expected_result, f"Check for the ticker {ticker} on {ex_date} failed.")
        
        ex_date = datetime.date(2026, 1, 17)
        expected_result = False
        result = check_earnings_conditions(df_earnings, ticker, ex_date)
        self.assertEqual(result, expected_result, f"Check for the ticker {ticker} on {ex_date} failed.")
        
        ex_date = datetime.date(2026, 1, 18)
        expected_result = True
        result = check_earnings_conditions(df_earnings, ticker, ex_date)
        self.assertEqual(result, expected_result, f"Check for the ticker {ticker} on {ex_date} failed.")
    
if __name__ == "__main__":
    df_dividends = run_dividends_events()
    df_earnings = run_earnings_events()
    unittest.main()
    print(df_dividends[['ticker', 'companyName', 'dividend_Ex_Date']].head(20))
