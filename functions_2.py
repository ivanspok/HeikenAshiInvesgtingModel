import numpy as np
import pandas as pd
# import pandas_ta as ta

from colog.colog import colog
c = colog()
warning = colog(TextColor='purple')
alarm = colog(TextColor='red')
blue = colog(TextColor='blue')
green = colog(TextColor='green')

class heiken_ashi():

  def __init__(self, o, h, l, c, pct):
  
    # common parameters
    self.o = o
    self.h = h
    self.l = l
    self.c = c
    self.pct = pct

    if self.c > self.o and self.pct > 0.01:
      self.colour = 'green'
    else:
      self.colur = 'red'

def pct_ratio(value_1 ,value_2, with_sign = 0):
  '''
  return percantage ratio of bigger value relative to low
  if with_sign = 1, return negative value if value_1 lower value_2
  '''
  if value_1 == 0 or value_2 == 0\
    or value_1 == value_2:
    
    return 0

  elif abs(value_1) > abs(value_2):
    
    return (value_1 / value_2 - 1) * 100

  else:
    if with_sign == 1:
      return - (value_2 / value_1 - 1) * 100
    else:
      return (value_2 / value_1 - 1) * 100

def get_heiken_ashi_v2(df):
  if not df.empty:
    df = df.sort_index()
    df['ha_c'] = (df['open'] + df['close'] + df['high'] + df['low']) / 4
    df['ha_o'] = 0.0
    df['ha_o'] = df['ha_o'].astype('float64')
    df['ha_pct'] = 0.0
    df['ha_colour'] = 'green' if df['close'].iloc[0] > df['open'].iloc[0]  else 'red'
    row = df.iloc[0].name
    df.at[row, 'ha_o'] = df['open'].iloc[0]
    
    for i in range(1, df.shape[0]):
      row = df.iloc[i].name
      df.at[row, 'ha_o'] =  (df['ha_o'].iloc[i - 1] + df['ha_c'].iloc[i - 1]) / 2
      df.at[row, 'ha_colour'] = 'green' if df['ha_o'].iloc[i]  < df['ha_c'].iloc[i] else 'red'
      
    df['ha_pct'] = np.where(df['ha_o'] < df['ha_c'],  
                          (df['ha_c'] / df['ha_o'] - 1) * 100,
                          -(df['ha_o'] / df['ha_c'] - 1) * 100)
      
  return df

def rma(x, n):
    """Running moving average"""
    a = np.full_like(x, np.nan)
    a[n] = x[1:n+1].mean()
    for i in range(n+1, len(x)):
        a[i] = (a[i-1] * (n - 1) + x[i]) / n
    return a

def data_augmentation(df, trade_type):
    
    df['pct_1'] = df['pct'].shift(1)
    df['ha_colour_1'] = df['ha_colour'].shift(1)
    df['ha_colour_2'] = df['ha_colour'].shift(2)
    
    df['ha_colour'] = np.where(df['ha_colour'] == 'red', 0, 1)
    df['ha_colour_1'] = np.where(df['ha_colour_1'] == 'red', 0, 1)
    df['ha_colour_2'] = np.where(df['ha_colour_2'] == 'red', 0, 1)

    for window in [3, 5, 10, 20, 30, 50, 100, 150, 300]:
      df[f'pct_sum_{window}'] = df['pct'].rolling(window = window).sum()
      df[f'pct_sum_{window}'] = df[f'pct_sum_{window}'].shift(1)
      df[f'ha_pct_sum_{window}'] = df['ha_pct'].rolling(window = window).sum()
      df[f'ha_pct_sum_{window}'] = df[f'ha_pct_sum_{window}'].shift(1)

    for window in [3, 5, 10, 15]:
    # for window in [10, 15]:
      df[f'green_red_prop_{window}'] = df['ha_colour'].rolling(window = window).sum() / (window - df['ha_colour'].rolling(window = window).sum() + 1)    
      df[f'green_red_prop_{window}'] = df[f'green_red_prop_{window}'].shift(1)         
                    
    df['pct_diff_150_30'] = df['pct_sum_150'] - df['pct_sum_30']
    df['pct_diff_30_5'] = df['pct_sum_30'] - df['pct_sum_5'] 
    df['pct_diff_150_5'] = df['pct_sum_150'] - df['pct_sum_5'] 
   
  
    df['max_5'] = df['open'].rolling(window = 5).max()
    df['max_10'] = df['open'].rolling(window = 10).max()
    df['max_500'] = df['open'].rolling(window = 500).max()
    df['max_100'] = df['open'].rolling(window = 100).max()
    df['max_300'] = df['open'].rolling(window = 300).max()
    df['max_1000'] = df['open'].rolling(window = 1000).max()
    df['min_5'] = df['open'].rolling(window = 5).min()
    df['min_10'] = df['open'].rolling(window = 10).min()
    df['min_500'] = df['open'].rolling(window = 500).min()
    df['min_100'] = df['open'].rolling(window = 100).min()
    df['min_300'] = df['open'].rolling(window = 300).min()
    df['min_1000'] = df['open'].rolling(window = 1000).min()
    df['maxmin_5']  = (df['open'] - df['min_5']) / (df['max_5'] - df['min_5'])
    df['maxmin_10']  = (df['open'] - df['min_10']) / (df['max_10'] - df['min_10'])
    df['maxmin_500']  = (df['open'] - df['min_500']) / (df['max_500'] - df['min_500'])
    df['maxmin_100']  = (df['open'] - df['min_100']) / (df['max_100'] - df['min_100'])
    df['maxmin_300']  = (df['open'] - df['min_300']) / (df['max_300'] - df['min_300'])
    df['maxmin_1000']  = (df['open'] - df['min_1000']) / (df['max_1000'] - df['min_1000'])

    df['maxmin_500_diff'] = df['maxmin_500'].diff()
    df['maxmin_500_avg_3'] = df['maxmin_500_diff'].rolling(window = 3).mean()
    df['maxmin_500_avg_5'] = df['maxmin_500_diff'].rolling(window = 5).mean()

    df['maxmin_1000_diff'] = df['maxmin_1000'].diff()
    df['maxmin_1000_avg_3'] = df['maxmin_1000_diff'].rolling(window = 3).mean()
    df['maxmin_1000_avg_5'] = df['maxmin_1000_diff'].rolling(window = 5).mean()

    df['change'] = df['close'].diff()
    df['gain'] = df.change.mask(df.change < 0, 0.0)
    df['loss'] = -df.change.mask(df.change > 0, -0.0)

    df['avg_gain_5'] = rma(df.gain.to_numpy(), 5)
    df['avg_loss_5'] = rma(df.loss.to_numpy(), 5)
    df['rs_5'] = df.avg_gain_5 / df.avg_loss_5
    df['rsi_5'] = 100 - (100 / (1 + df.rs_5))
    df['rsi_5'] = df['rsi_5'].shift(1)

    df['avg_gain_10'] = rma(df.gain.to_numpy(), 10)
    df['avg_loss_10'] = rma(df.loss.to_numpy(), 10)
    df['rs_10'] = df.avg_gain_10 / df.avg_loss_10
    df['rsi_10'] = 100 - (100 / (1 + df.rs_10))
    df['rsi_10'] = df['rsi_10'].shift(1)

    df['avg_gain_14'] = rma(df.gain.to_numpy(), 14)
    df['avg_loss_14'] = rma(df.loss.to_numpy(), 14)
    df['rs_14'] = df.avg_gain_14 / df.avg_loss_14
    df['rsi_14'] = 100 - (100 / (1 + df.rs_14))
    df['rsi_14'] = df['rsi_14'].shift(1)
    df['rsi_14_diff'] = df['rsi_14'].diff()
    df['rsi_14_avg_3'] = df['rsi_14_diff'].rolling(window = 3).mean()
    df['rsi_14_avg_5'] = df['rsi_14_diff'].rolling(window = 5).mean()


    df['avg_gain_20'] = rma(df.gain.to_numpy(), 20)
    df['avg_loss_20'] = rma(df.loss.to_numpy(), 20)
    df['rs_20'] = df.avg_gain_20 / df.avg_loss_20
    df['rsi_20'] = 100 - (100 / (1 + df.rs_20))
    df['rsi_20'] = df['rsi_20'].shift(1)
    df['rsi_20_diff'] = df['rsi_20'].diff()
    df['rsi_20_avg_3'] = df['rsi_20_diff'].rolling(window = 3).mean()
    df['rsi_20_avg_5'] = df['rsi_20_diff'].rolling(window = 5).mean()


    df['avg_gain_30'] = rma(df.gain.to_numpy(), 30)
    df['avg_loss_30'] = rma(df.loss.to_numpy(), 30)
    df['rs_30'] = df.avg_gain_30 / df.avg_loss_30
    df['rsi_30'] = 100 - (100 / (1 + df.rs_30))
    df['rsi_30'] = df['rsi_30'].shift(1)

    df['avg_5'] = df['pct_1'].rolling(window=5).mean()
    df['avg_10'] = df['pct_1'].rolling(window=10).mean()
    df['avg_20'] = df['pct_1'].rolling(window=20).mean()
    df['avg_30'] = df['pct_1'].rolling(window=30).mean()
    df['avg_50'] = df['pct_1'].rolling(window=50).mean()
    df['ewm_0.01'] = df['pct_1'].ewm(com=0.01, min_periods=10).mean()
    df['ewm_0.05'] = df['pct_1'].ewm(com=0.05, min_periods=10).mean()
    df['ewm_0.1'] = df['pct_1'].ewm(com=0.1, min_periods=10).mean()
    # df['pct_2'] = df['pct_1'].shift(1)
    # df['ha_pct_1'] = df['ha_pct'].shift(1)

    # df['avg_50_10'] = df['avg_50'] / (df['avg_10'] + 0.1)

    # Strategy = ta.Strategy(
    #       name="EMAs, BBs, and MACD",
    #       description="My strategy",
    #       ta=[
    #           {"kind": "stochrsi"},
    #           {"kind": "bias"},
    #           {"kind": "bop"},
    #           {"kind": "roc"},
    #           {"kind": "stochrsi"},
    #           {"kind": "inertia"},
    #           {"kind": "ao"},
    #           {"kind": "apo"},
    #       ]
    #   )

    # df.ta.strategy(Strategy)  

    # shiftcolumns = ['STOCHRSIk_14_14_3_3',
    #   'STOCHRSId_14_14_3_3', 'BIAS_SMA_26', 'BOP',
    #   'ROC_10', 'INERTIA_20_14', 'AO_5_34',
    #   'APO_12_26']
    # for column in shiftcolumns:
    #     df[column] = df[column].shift(1)

    # columns = list(set(df.columns) - set(['close', 'gain', 'ha_c', 'open', 'high', 'low', 'win_long', 'win_short', 'result', 'change', 'ha_o','loss', 'pct', 'ha_colour']))
    dropcolumns = ['gain', 'ha_c',  'win_long', 'win_short', 'change', 'ha_o','loss', 'pct', 'ha_colour',  \
                    'avg_gain_5',  'avg_gain_10',  'avg_gain_14', 'avg_gain_30',
                     'rs_5', 'rs_10', 'rs_14', 'rs_30', 'avg_loss_5', 'avg_loss_10', 'avg_loss_14', 'avg_loss_30',  'pct_sum_1', 'ha_pct_sum_1', 'ha_pct', 
                     'max_5', 'max_10', 'max_100', 'max_300', 'max_500',  'min_5', 'min_10', 'min_100', 'min_300', 'min_500']
    
    
    for column in dropcolumns:
       try:
          df.drop(columns = [column], inplace = True)
       except:
          pass
    
    df.dropna(inplace=True)
    
    X = df.drop(columns = ['open', 'low', 'high', 'close'])
    # X = (X - X.mean()) / X.std()

    X['open'] = df['open']
    X['low'] = df['low']
    X['high'] = df['high']
    X['close'] = df['close']
    
    X = X.reindex(sorted(X.columns), axis=1)

    return X

import numpy as np


def vwap_slope_from_open(df: pd.DataFrame, window=5, verbose=False) -> float:
    # 1. Определяем текущий день
    day = df.index[-1].date()

    # 2. Фильтруем только строки текущего дня
    df_today = df[df.index.date == day]

    # 3. Определяем время открытия рынка (09:30 NY)
    market_open = pd.Timestamp.combine(day, pd.Timestamp("09:30").time())
    market_open = market_open.tz_localize("America/New_York")

    # 4. Фильтруем только строки после открытия рынка
    df_today = df_today[df_today.index >= market_open]
    
    slope = (df_today['vwap'].iloc[-1] / df_today['open'].iloc[0] - 1) * 100
    
    return slope

def vwap_slope(df: pd.DataFrame, window=5, verbose=False) -> float:
    # 1. Определяем текущий день
    day = df.index[-1].date()

    # 2. Фильтруем только строки текущего дня
    df_today = df[df.index.date == day]

    # 3. Определяем время открытия рынка (09:30 NY)
    market_open = pd.Timestamp.combine(day, pd.Timestamp("09:30").time())
    market_open = market_open.tz_localize("America/New_York")

    # 4. Фильтруем только строки после открытия рынка
    df_today = df_today[df_today.index >= market_open]

    # 5. Если данных меньше чем window — slope = 0
    if len(df_today) < window:
        return 0.0

    # 6. Берём последние window значений VWAP
    y = df_today['vwap'].iloc[-window:]

    # 7. Вычисляем наклон
    x = np.arange(len(y))
    slope, _ = np.polyfit(x, y, 1)

    if verbose:
        print("VWAP window:", y.values)
        print("Slope:", slope)

    return slope

def zigzag(df: pd.DataFrame, deviation=5, depth=5, backstep=3) -> pd.Series:
    """
    ZigZag using High/Low prices.
    
    df must contain: df["high"], df["low"]
    deviation: percent threshold (e.g., 5 = 5%)
    depth: minimum bars between pivots
    backstep: pivot cleanup distance
    """
    high = df["high"]
    low = df["low"]

    deviation /= 100.0
    n = len(df)

    pivots = np.zeros(n)
    trend = 0  # 1 = uptrend, -1 = downtrend

    # Start from first bar
    last_pivot_idx = 0
    last_pivot_price = low.iloc[0]  # start as if downtrend

    for i in range(1, n):

        # Current highs/lows
        hi = high.iloc[i]
        lo = low.iloc[i]

        # No trend yet
        if trend == 0:
            # Uptrend start
            if (hi - last_pivot_price) / last_pivot_price >= deviation:
                trend = 1
                last_pivot_idx = i
                last_pivot_price = hi
                pivots[i] = hi

            # Downtrend start
            elif (last_pivot_price - lo) / last_pivot_price >= deviation:
                trend = -1
                last_pivot_idx = i
                last_pivot_price = lo
                pivots[i] = lo

        # Uptrend logic
        elif trend == 1:
            # New higher high
            if hi > last_pivot_price:
                last_pivot_price = hi
                last_pivot_idx = i

            # Reversal to downtrend
            elif (last_pivot_price - lo) / last_pivot_price >= deviation:
                if i - last_pivot_idx >= depth:
                    pivots[last_pivot_idx] = last_pivot_price
                    trend = -1
                    last_pivot_price = lo
                    last_pivot_idx = i

        # Downtrend logic
        elif trend == -1:
            # New lower low
            if lo < last_pivot_price:
                last_pivot_price = lo
                last_pivot_idx = i

            # Reversal to uptrend
            elif (hi - last_pivot_price) / last_pivot_price >= deviation:
                if i - last_pivot_idx >= depth:
                    pivots[last_pivot_idx] = last_pivot_price
                    trend = 1
                    last_pivot_price = hi
                    last_pivot_idx = i

        # Backstep cleanup
        if pivots[i] != 0:
            for j in range(1, backstep + 1):
                if i - j >= 0:
                    pivots[i - j] = 0

    # Final pivot
    pivots[last_pivot_idx] = last_pivot_price
    
    return pd.Series(pivots, index=df.index)

def zigzag_buy_criteria_old(zz: pd.Series, current_price: float) -> bool:
    pass
    # pivots = zz[zz != 0]
    
    # # down trend
    # if pivots.iloc[-1] < pivots.iloc[-3] and pivots.iloc[-2] < pivots.iloc[-4]: 
        
    #     # H -> L -> H -> L (-1 is min, -2 is max, -3 is min, -4 is max)
    #     if pivots.iloc[-1] < pivots.iloc[-2]:
    #         if current_price > pivots.iloc[-1] + (pivots.iloc[-2] - pivots.iloc[-1]) * 0.7: return True
    #         else: return False

    #     # L -> H -> L -> H (-1 is max, -2 is min, -3 is max, -4 is min)
    #     if pivots.iloc[-1] > pivots.iloc[-2]:
    #         return True
        
    # # up trend
    # if pivots.iloc[-1] > pivots.iloc[-3] and pivots.iloc[-2] > pivots.iloc[-4]:  
        
    #     # H -> L -> H -> L (-1 is min, -2 is max, -3 is min, -4 is max)
    #     if current_price > pivots.iloc[-1] + (pivots.iloc[-2] - pivots.iloc[-1]) * 0.3 \
    #         and current_price < pivots.iloc[-2] + (pivots.iloc[-2] - pivots.iloc[-2]) * 1.618:
    #         return True
        
    #     # L -> H -> L -> H (-1 is max, -2 is min, -3 is max, -4 is min) 
    #     if current_price < pivots.iloc[-3] + (pivots.iloc[-3] - pivots.iloc[-2]) * 1.618 \
    #         and current_price > pivots.iloc[-1] - (pivots.iloc[-1] - pivots.iloc[-2]) * 0.3:
    #         return True
    #     else:
    #         return False
               
    # return False

def zigzag_buy_criteria(zz: pd.Series, current_price: float) -> bool:
    pivots = zz[zz != 0]

    if len(pivots) < 4:
        return False

    p1 = pivots.iloc[-1]   # последний pivot
    p2 = pivots.iloc[-2]
    p3 = pivots.iloc[-3]
    p4 = pivots.iloc[-4]

    # ----- DOWN TREND (Lower High + Lower Low) -----
    if p1 < p3 and p2 < p4:

        # Pattern: H → L → H → L (p1 < p2)
        if p1 < p2:
            leg = p2 - p1
            fib_070 = p1 + leg * 0.7
            return current_price >= fib_070

        # Pattern: L → H → L → H (p1 > p2)
        if p1 > p2:
            return current_price >= p1

    # ----- UP TREND (Higher High + Higher Low) -----
    if p1 > p3 and p2 > p4:

        # Pattern: L → H → L → H (p1 > p2)
        if p1 > p2:
            leg = p1 - p2
            leg_1618 = p3 - p2
            fib_neg030 = p1 - leg * 0.3
            fib_1618 = p1 + leg_1618 * 1.618
            return fib_neg030 <= current_price <= fib_1618

        # Pattern: H → L → H → L (p1 < p2)
        if p1 < p2:
            leg = p2 - p1
            fib_030 = p1 + leg * 0.3
            fib_1618 = p2 + leg * 1.618
            return fib_030 <= current_price <= fib_1618

    return False

def zigzag_buy_criteria_test(zz: pd.Series) -> bool:
    pivots = zz[zz != 0]
    
    y = pivots.iloc[-8:]
    x = np.arange(len(y))
    slope, _ = np.polyfit(x, y, 1)     
     
    if slope > 0:
        return True
    if slope < 0:  
        return False

def zigzag_buy_criteria_test_2(zz: pd.Series, df: pd.DataFrame, current_price, ticker) -> bool:
        
    pivots = zz[zz != 0]
    
    if len(pivots) < 8:
        return False
    
    y = pivots.iloc[-8:]
    x = np.arange(len(y))
    slope, _ = np.polyfit(x, y, 1)    
    # print(f'Ticker {ticker} has slope {slope:.3f}')
    
    # last pivots
    p1 = pivots.iloc[-1]
    p2 = pivots.iloc[-2]
    p3 = pivots.iloc[-3]
    p4 = pivots.iloc[-4]
    p5 = pivots.iloc[-5]
    
    p1idx = pivots.index[-1]
    p2idx = pivots.index[-2]
    p3idx = pivots.index[-3]
    p4idx = pivots.index[-4]
    p5idx = pivots.index[-5]
    
    
    max_value_after_p1 = df['high'].loc[p1idx:].max()
    if df['high'].loc[p1idx] != df['high'].iloc[-1]:
        min_value_after_p1 = df['low'].loc[p1idx:].min()
    else:
        min_value_after_p1 = df['low'].iloc[-1]   
    
        
    max_value_after_p2 = df['high'].loc[p2idx:].max()
    if df['low'].loc[p2idx] != df['low'].iloc[-1]:
        min_value_after_p2 = df['low'].loc[p2idx:].min()   
    else:
        min_value_after_p2 = df['low'].iloc[-1]
    
    if slope < 0:  
        return False 
    
    # trend goes down
    if p1 < p3:
        return False
    
    # we at local maximum, 
    # want that price drop below top - fib 0.618 and after you can buy
    if p2 > p3:
        leg = p2 - p3
        # we have local trend down
        if p3 < p5:
            if min_value_after_p2 < p2 - leg * 0.764:  # 0.764 -> 0.382
                return True
        else: # p3 > p5, we have local trend up 
            if min_value_after_p2 < p2 - leg * 0.382:  # 0.764 -> 0.382
                return True
        
    # we at local minimum,
    # want that price rise above bottom + fib 0.309 and after you can buy
    if p2 < p3: 
        leg = p3 - p2
        # we have new p1 point (my alg), but price below personal fib 0.309 level
        if p1 < p3:
            # trend goes down, we don't want to buy if price rise above p2 + fib 0.309 level, beacuse it could go down soon
            if p3 < p5:
                if current_price < p2 + leg * 0.309:
                    return True
            else: 
            # trend goes up, we can but in longer range from p3 + fib 0.309 level, because price can go up more
                if current_price < p3 + leg * 0.309:
                    return True
        # p1 above p3; we expecting price going down, and wait when price below personal p1 - fib 0.764 level
        else:
            leg = p1 - p2
            if min_value_after_p1 < p1 - leg * 0.382: # 0.764 -> 0.382
                return True
    return False

def zigzag_buy_criteria_with_fib(df: pd.DataFrame, zz: pd.Series) -> bool:
    """
    df: DataFrame with columns ['high','low','close']
    zz: pandas Series with ZigZag pivots (0 for non-pivot)
    """

    # --- 1. Базовый ZigZag критерий (как у тебя) ---
    buy_signal = False
    pivots = zz[zz != 0]

    # last value is a rise maximum:
    if zz.iloc[-1] > zz.iloc[-2] \
        and zz.iloc[-1] > zz.iloc[-3]:
        buy_signal = True
        return buy_signal

    if len(pivots) < 2:
        return False

    A = pivots.iloc[-2]
    B = pivots.iloc[-1]
    A_idx = pivots.index[-2]
    B_idx = pivots.index[-1]

    # --- 3. Определяем направление волны ---
    if B > A:
        # Восходящая волна: A = минимум, B = максимум
        fib_130_up = B + (B - A) * 0.3
    else:
        # Нисходящая волна: A = максимум, B = минимум
        fib_130_down = B - (A - B) * 0.3

    # --- 4. Проверяем касание уровня 130% ---
    price_now = df['close'].iloc[-1]
    # cond_touch_before = (df['close'].loc)
    
    if  A > B and price_now <= fib_130_down:
        buy_signal = True

    # --- 5. Итоговый сигнал ---
    return buy_signal

def zigzag_buy_criteria_test_3(zz: pd.Series, df: pd.DataFrame, fib_loc_min=0.764, fib_loc_max=1.3,
                               fib_loc_max_p1_less_p3 = 0.764, fib_low_border=0.236, break_classic_type=1) -> bool:
    """
    df: DataFrame with 'close'
    zz: ZigZag Series (0 for non-pivot)
    """
    buy_downtrend = False
    buy_uptrend = False
    # --- 1. Берём последние 5 пивотов ---
    pivots = zz[zz != 0]
    if len(pivots) < 5:
        return False

    # Твой порядок:
    p1 = pivots.iloc[-1]   # самый новый pivot
    p2 = pivots.iloc[-2]
    p3 = pivots.iloc[-3]
    p4 = pivots.iloc[-4]
    p5 = pivots.iloc[-5]   # самый старый pivot

    price_now = df["close"].iloc[-1]
    
    y = pivots.iloc[-5:]
    x = np.arange(len(y))
    slope, _ = np.polyfit(x, y, 1)

    # --- 1. Восходящий тренд (HH + HL) ---
    # trend_up =  (p1 > p3 > p5) and (p2 > p4)
    trend_up =  (p1 > p3) and (p2 > p4)

    # --- 2. Нисходящий тренд (LH + LL) ---
    # trend_down = (
    #     (p1 < p3)
    #     or ((pct_ratio(p3, p5) < 0.3) and (p1 < p3))
    # ) and (p4 > p2)
    trend_down = (p1 < p3) and (p2 < p4)


    if not trend_up and not trend_down:
        trend_up = (slope > 0)
        # trend_down = slope < 0
        # print(f'Ticker {ticker} has no clear trend, trend direction define by slope: {slope:.3f}')
    
    # --- Пробой классического нисходящего тренда ---
    break_classic = False
    # if break_classic_type == 1:
    #     if p1 > p2:
    #         break_classic = price_now > p4   
    # elif break_classic_type == 2:
    #     break_classic = price_now > min(p3, p5)
    if p1 > p2:
        # break_classic = price_now > p4
        # leg = abs(p3 - p2)
        # price_now_less_0p382 = price_now < p2 + leg * 0.382
        break_classic = False
    else:
        break_classic = price_now > min(p3, p5)
        leg = abs(p2 - p1)
        price_now_less_0p382 = price_now < p1 + leg * 0.382

    # limit price rise
    
    # --- Пробой при маленькой коррекции p3 (p3 ≈ p5) ---
    break_small_pullback = (pct_ratio(p3, p5) < 0.3) and (price_now > p5)

    # --- Итоговый пробой нисходящего тренда ---
    buy_downtrend = trend_down and (break_classic or break_small_pullback) and price_now_less_0p382

    # --- 7. BUY при продолжении восходящего тренда ---
    # Case 1: p1 lower p2 (local minimum), buy after p1 drop below fib 0.764 of (p2-p3) leg
    if trend_up:
        if p1 < p2:
            if (price_now > p1 + (p2 - p1) * 0.236 # price above fib 0.236 level of 
                and price_now < p3 + (p2 - p3) * 0.764): # price below fib 0.764 level of p3
                    return True
            if p2 > p3: # p1 < p2 > p3
                leg = p2 - p3
                if p1 > p3: # we have real up trend
                    if p1 < p2 - leg * fib_loc_min \
                     and price_now < p2:# price below fib 0.764 level of p2
                        buy_uptrend = True
                else: # p1 < p2 and p1 < p3 - we have local down trend, but slope > 0; we break p3 level;
                    if (p3 < p4) and (p4 > p5) and (p1 > p5):
                        leg2 = abs(p4 - p5) # p1 < p2, p2 > p3, p 
                        if p1 < p4  - leg2 * fib_loc_min \
                            and price_now < p2:# price below fib 0.764 level of p4; want to buy after price drop below p4 - fib 0.764 level, because we have local trend up, and price can go up more after that
                            buy_uptrend = True
            else: # p1 < p2 < p3 - not zigzag pattern, but we can have local trend up, and want to buy after price drop below p2 - fib 0.382 level
                pass 
            
        # Case 2: p1 higher p2 (local maximum), we can buy until price reach fib 130% level of p, because it can go up more, but reverse after
        else: # p1 > p2
            leg = p3 - p2 # p1 > p2 then p2 < p3
            if p1 > p3:
                if price_now < p2 + leg * fib_loc_max \
                    and price_now > p2 + leg * fib_low_border: # price below fib 130% level of p2
                    buy_uptrend = True  # when stop buy in this case???
            else: #p1 < p3
                    if price_now < p2 + leg * fib_loc_max_p1_less_p3 \
                    and price_now > p2 + leg * fib_low_border: # price below fib 130% level of p2
                        buy_uptrend = True  # when stop buy in this case???
        
    # if buy_downtrend:
    #     print(f'BUY signal on downtrend')
    
    # if buy_uptrend:
    #     print(f'BUY signal on uptrend')

    # --- 8. Итог ---
    return buy_downtrend or buy_uptrend


def _line(x1, y1, x2, y2):
    slope = (y2 - y1) / (x2 - x1)
    return slope, (x1, y1, x2, y2)


def _best_valid_high_line(xs, ys):
    candidates = []

    for i in range(3):
        for j in range(i+1, 3):
            slope, (x1, y1, x2, y2) = _line(xs[i], ys[i], xs[j], ys[j])
            ok = True
            for k in range(3):
                y_line = slope * (xs[k] - x1) + y1
                if ys[k] > y_line:  # точка выше линии → линия плохая
                    ok = False
                    break
            if ok:
                candidates.append((slope, (x1, y1, x2, y2)))

    if not candidates:
        # fallback: линия через крайние точки
        return _line(xs[0], ys[0], xs[2], ys[2])

    # минимальный наклон вверх → самая плотная линия
    return min(candidates, key=lambda x: x[0])


def _best_valid_low_line(xs, ys):
    candidates = []

    for i in range(3):
        for j in range(i+1, 3):
            slope, (x1, y1, x2, y2) = _line(xs[i], ys[i], xs[j], ys[j])
            ok = True
            for k in range(3):
                y_line = slope * (xs[k] - x1) + y1
                if ys[k] < y_line:  # точка ниже линии → линия плохая
                    ok = False
                    break
            if ok:
                candidates.append((slope, (x1, y1, x2, y2)))

    if not candidates:
        return _line(xs[0], ys[0], xs[2], ys[2])

    # максимальный наклон вниз → самая плотная линия
    return max(candidates, key=lambda x: x[0])


def best_trend_line(points: pd.Series, mode: str, df_index):
    """
    points: 3 ZigZag точки (high или low)
    mode: "high" или "low"
    df_index: исходный df.index
    """

    xs = df_index.get_indexer(points.index)
    ys = points.values

    x1, x2, x3 = xs
    y1, y2, y3 = ys

    # ==========================
    #   HIGH LINE
    # ==========================
    if mode == "high":

        # CASE 1 — средняя ниже обеих → линия через крайние
        if y2 < y1 and y2 < y3:
            return _line(x1, y1, x3, y3)

        # CASE 2 — средняя выше обеих → доминирует
        if y2 > y1 and y2 > y3:
                return _line(x2, y2, x3, y3)

        # CASE 3 — средняя между → ищем валидную линию
        return _best_valid_high_line(xs, ys)

    # ==========================
    #   LOW LINE
    # ==========================
    if mode == "low":

        # CASE 1 — средняя выше обеих → линия через крайние
        if y2 > y1 and y2 > y3:
            return _line(x1, y1, x3, y3)

        # CASE 2 — средняя ниже обеих → доминирует
        if y2 < y1 and y2 < y3:
                return _line(x2, y2, x3, y3)

        # CASE 3 — средняя между → ищем валидную линию
        return _best_valid_low_line(xs, ys)


# ==========================
#   КОНУСНЫЙ АНАЛИЗ
# ==========================

def zigzag_cone_position(zz: pd.Series, df: pd.DataFrame):

    pivots = zz[zz != 0].iloc[:-1]
    if len(pivots) < 6:
        return None

    highs = pivots[pivots > pivots.shift(1)][-3:]
    lows  = pivots[pivots < pivots.shift(1)][-3:]

    slope_high, (xh1, yh1, xh2, yh2) = best_trend_line(highs, "high", df.index)
    slope_low,  (xl1, yl1, xl2, yl2) = best_trend_line(lows, "low", df.index)

    def high_line(x):
        return slope_high * (x - xh1) + yh1

    def low_line(x):
        return slope_low * (x - xl1) + yl1

    x_now = len(df) - 1
    price_now = df['close'].iloc[-1]

    upper_line_now = high_line(x_now)
    lower_line_now = low_line(x_now)

    if upper_line_now == lower_line_now:
        price_pct = 0.0
    else:
        price_pct = (price_now - lower_line_now) / (upper_line_now - lower_line_now) * 100

    if lower_line_now <= upper_line_now:
        # нормальный конус
        if price_now < lower_line_now:
            position = "below"
        elif price_now > upper_line_now:
            position = "above"
        else:
            position = "inside"
    else:
        # линии пересеклись — конус перевёрнут
        # цена между линиями, если она между значениями
        if upper_line_now <= price_now <= lower_line_now:
            position = "inside_cross"
        elif price_now > lower_line_now:
            position = "above_cross"
        else:
            position = "below_cross"


    if slope_high > 0 and slope_low > 0:
        cone_type = "both_up"
    elif slope_high < 0 and slope_low < 0:
        cone_type = "both_down"
    elif slope_high > 0 and slope_low < 0:
        cone_type = "high_up_low_down"
    elif slope_high < 0 and slope_low > 0:
        cone_type = "high_down_low_up"
    else:
        cone_type = "flat_mixed"

    return {
        "upper_line_now": upper_line_now,
        "lower_line_now": lower_line_now,
        "price": price_now,
        "price_pct": price_pct,
        "position": position,
        "cone_type": cone_type,
        "slope_high": slope_high,
        "slope_low": slope_low,
        "high_points": [yh1, yh2],
        "low_points": [yl1, yl2],
    }

def zigzag_has_fib_drop(zz: pd.Series, df: pd.DataFrame, drop_value=0.7):
    pivots = zz[zz != 0]
    if len(pivots) < 5:
        return False

    # Твой порядок:
    p1 = pivots.iloc[-1]   # самый новый pivot
    p2 = pivots.iloc[-2]
    p3 = pivots.iloc[-3]
    p4 = pivots.iloc[-4]
    p5 = pivots.iloc[-5]   # самый старый pivot

    price_now = df["close"].iloc[-1]
    
    if p1 < p3:
        return False
    
    if p1 < p2:        
        
        leg = abs(p2 - p3)
        if min(p1, price_now) < p2 - leg * drop_value:
            return True
    
    else: # p1 > p2 and p1 > p3
        leg = (p1 - p2)
        if price_now < p1 - leg * drop_value:
            return True
    
    return False
            

def zigzag_buy_criteria_with_bullish_candels(zz: pd.Series, df: pd.DataFrame) -> bool:
    """
    df: DataFrame with 'close'
    zz: ZigZag Series (0 for non-pivot)
    """
    buy_downtrend = False
    buy_uptrend = False
    # --- 1. Берём последние 5 пивотов ---
    pivots = zz[zz != 0]
    if len(pivots) < 5:
        return False

    # Твой порядок:
    p1 = pivots.iloc[-1]   # самый новый pivot
    p2 = pivots.iloc[-2]
    p3 = pivots.iloc[-3]
    p4 = pivots.iloc[-4]
    p5 = pivots.iloc[-5]   # самый старый pivot

    price_now = df["close"].iloc[-1]
    
    y = pivots.iloc[-5:]
    x = np.arange(len(y))
    slope, _ = np.polyfit(x, y, 1)

    # --- 1. Восходящий тренд (HH + HL) ---
    trend_up =  (p1 > p3) and (p2 > p4)
    
    if trend_up and p1 < p2 and p2 > p3:
        return True
    else:
        return False

    

def zigzag_sell_criteria(zz: pd.Series, current_price: float) -> bool:
    # allow to sell if p1 > p2 and p1 more than fib_level 1.618 level (overbought reason)
    pivots = zz[zz != 0]

    if len(pivots) < 4:
        return False

    p1 = pivots.iloc[-1]
    p2 = pivots.iloc[-2]
    p3 = pivots.iloc[-3]
    p4 = pivots.iloc[-4]

    # p1 must be a HIGH (p1 > p2)
    if p1 <= p2:
        return False

    fib_level = p3 + (p3 - p2) * 0.618

    # p1 must exceed fib level
    if p1 <= fib_level:
        return False

    # current price must be below p1 (start of reversal)
    if current_price >= p1:
        return False

    return True

def zigzag_sell_criteria_break_l3_or_l5(zz: pd.Series, current_price: float) -> bool:  
    # sell if price broke p3 or p5 level
    pivots = zz[zz != 0] 

    if len(pivots) < 4:
        return False

    p1 = pivots.iloc[-1]
    p2 = pivots.iloc[-2]
    p3 = pivots.iloc[-3]
    p4 = pivots.iloc[-4]
    if len(pivots) >= 5:
        p5 = pivots.iloc[-5]
    else :
        p5 = p4
    
    if p1 > p2: # local maximum
        if p1 > p3: # trend was up, but we have local maximum, and want to sell after price break below p3 level or p5 level
            if current_price < max(p3, p4):
                return True
        if p1 > p5 and p3 < p5:
            if current_price < p5:
                return True        

    return False

def zigzag_sell_criteria_down_trend_or_breakdown1618(zz: pd.Series, current_price: float) -> bool:
    # condition for 1m and and deviation 0.5%
    # sell if trend down or breakdown1618
    pivots = zz[zz != 0] 

    if len(pivots) < 4:
        return False

    p1 = pivots.iloc[-1]
    p2 = pivots.iloc[-2]
    p3 = pivots.iloc[-3]
    p4 = pivots.iloc[-4]
    p5 = pivots.iloc[-5] if len(pivots) >= 5 else p4
    
    trend_down = (
        (p5 > p3 > p1)
        or ((pct_ratio(p3, p5) < 0.3) and (p1 < p3))
    ) and (p4 > p2)
    if trend_down:
        return True
    
    # p1 — HIGH pivot
    if p1 > p2 and p1 > p3:

        # длина импульса
        leg = abs(p3 - p2)

        # уровень расширения 1.618
        fib_1618 = min(p2, p3) + leg * 1.618

        # условие ложного пробоя и разворота
        if (p1 > fib_1618) and (current_price < fib_1618):
            return True

    return False


def zigzag_sell_criteria_down_trend_or_breakdown1618_without_current_price(zz: pd.Series, current_price: float) -> bool:
    # condition for 1m and and deviation 1%
    pivots = zz[zz != 0] 

    if len(pivots) < 4:
        return False, 2

    p1 = pivots.iloc[-1]
    p2 = pivots.iloc[-2]
    p3 = pivots.iloc[-3]
    p4 = pivots.iloc[-4]
    p5 = pivots.iloc[-5] if len(pivots) >= 5 else p4
    
    trend_down = (
        (p5 > p3 > p1)
        or ((pct_ratio(p3, p5) < 0.3) and (p1 < p3))
    ) and (p4 > p2)
    if trend_down:
        trailing_ratio = 0.1
        return True, trailing_ratio
    
    # p1 — HIGH pivot
    if p1 > p2 and p1 > p3:

        # длина импульса
        leg = abs(p3 - p2)

        # уровень расширения 1.618
        fib_1618 = min(p2, p3) + leg * 1.618

        # условие ложного пробоя и разворота
        if (p1 > fib_1618):
            option1 = 0.236 * 100 * abs(p3 - p2) / current_price
            option2 = 0.118 * 100 * abs(p1 - p2) / current_price
            trailing_ratio = min(option1, option2, 0.1)
            return True, trailing_ratio

    return False, 2


def find_key_levels(zz: pd.Series, number_points, tolerance=0.005, min_distance=0.003):
    """
    min_distance: минимальная относительная дистанция между соседними уровнями (0.003 = 0.3%)
    """

    pivots = zz[zz != 0].iloc[-number_points:].values
    if len(pivots) == 0:
        return {"levels": [], "zones": []}

    pivots = sorted(map(float, pivots))

    min_pivot = pivots[0]
    max_pivot = pivots[-1]

    # 2) Кластеризация
    clusters = []
    current_cluster = [pivots[0]]

    for price in pivots[1:]:
        mean_price = np.mean(current_cluster)
        if abs(price - mean_price) / mean_price <= tolerance:
            current_cluster.append(price)
        else:
            clusters.append(current_cluster)
            current_cluster = [price]

    clusters.append(current_cluster)

    # 3) Кластеры → уровни
    raw_levels = []
    for cluster in clusters:
        level_price = float(np.mean(cluster))
        strength = len(cluster)
        raw_levels.append((level_price, strength))

    # 4) Добавляем min и max
    raw_levels.append((min_pivot, 1))
    raw_levels.append((max_pivot, 1))

    raw_levels = list({round(l[0], 6): l for l in raw_levels}.values())
    raw_levels = sorted(raw_levels, key=lambda x: x[0])

    # Если уровней ≤4 — просто берём их
    if len(raw_levels) <= 4:
        top = sorted(raw_levels, key=lambda x: x[0])
    else:
        # maximin с фильтром по минимальной дистанции
        import itertools

        best_combo = None
        best_score = -1

        # всегда включаем min и max, выбираем ещё 2 из остальных
        candidates = raw_levels[1:-1]

        for a, b in itertools.combinations(candidates, 2):
            combo = sorted([raw_levels[0], a, b, raw_levels[-1]], key=lambda x: x[0])

            # проверяем минимальную дистанцию между соседними уровнями
            ok = True
            for i in range(len(combo) - 1):
                p1, p2 = combo[i][0], combo[i+1][0]
                if abs(p2 - p1) / ((p1 + p2) / 2) < min_distance:
                    ok = False
                    break

            if not ok:
                continue

            # считаем maximin: минимальный интервал между уровнями
            intervals = [combo[i+1][0] - combo[i][0] for i in range(3)]
            score = min(intervals)

            if score > best_score:
                best_score = score
                best_combo = combo

        # если не нашли ни одной валидной комбинации → fallback: старый maximin без фильтра
        if best_combo is None:
            best_combo = []
            best_score = -1
            for a, b in itertools.combinations(candidates, 2):
                combo = sorted([raw_levels[0], a, b, raw_levels[-1]], key=lambda x: x[0])
                intervals = [combo[i+1][0] - combo[i][0] for i in range(3)]
                score = min(intervals)
                if score > best_score:
                    best_score = score
                    best_combo = combo

        top = best_combo

    # Зоны
    zones = [(top[i][0], top[i+1][0]) for i in range(len(top) - 1)]

    return {
        "levels": top,
        "zones": zones
    }



def count_bounces(zz: pd.Series, level, threshold=0.0075, max_gap_pivots=3):
    """
    zz: zigzag series
    level: уровень
    threshold: допустимое отклонение от уровня (0.75%)
    max_gap_pivots: максимальное количество pivot-ов между касаниями уровня
                    если больше — цепочка прерывается
    """

    pivots = zz[zz != 0]

    # выбираем pivots, которые близко к уровню
    close_pivots = pivots[
        (pivots < level * (1 + threshold)) &
        (pivots > level * (1 - threshold))
    ]

    if len(close_pivots) == 0:
        return 0

    # индексы всех касаний уровня
    touch_indices = close_pivots.index.to_list()

    bounce_count = 1  # первое касание уже есть
    last_touch = touch_indices[0]

    for idx in touch_indices[1:]:
        # считаем количество pivot-ов между касаниями
        gap = pivots.index.get_loc(idx) - pivots.index.get_loc(last_touch)

        # если касания идут подряд (gap <= max_gap_pivots)
        if gap <= max_gap_pivots:
            bounce_count += 1
            last_touch = idx
        else:
            # цепочка прерывается — p12 не считается
            break

    return bounce_count



def should_buy_from_levels(levels, zones, current_price: float, zz: pd.Series, verbose=False) -> bool:
    """
    levels: [(level_price, strength), ...]
    zones: [(low, high), (low, high), (low, high)]
    current_price: текущая цена
    zz: zigzag серия для подсчёта отскоков
    """
    pivots = zz[zz != 0]
    
    # Определяем зону по цене
    zone = None
    for i, (low, high) in enumerate(zones):
        if low <= current_price <= high:
            if i == 0:
                zone = "low"
            elif i == 1:
                zone = "middle"
            elif i == 2:
                zone = "top"
            break

    if zone is None:
        return False

    # -----------------------------
    # 1) ВЕРХНЯЯ ЗОНА
    # -----------------------------
    if zone == "top":
        low, high = zones[2]
        fib_level = levels[2][0] + (levels[2][0] - levels[1][0]) * 0.618  # уровень 1.3 fib level 2 - level 1
        fib_level_zone0 = low + (high - low) * 0.768  # уровень 1.3 fib level 2 - level 1
        
        if current_price > fib_level:
            if verbose:
                print(f"Price {current_price:.2f} is top zone above fib level {fib_level:.2f} in top zone. Not buying.")
            return False
        if current_price > fib_level_zone0:
            if verbose:
                print(f"Price {current_price:.2f} is top zone above fib zone {fib_level_zone0:.2f} in top zone. Not buying.")
            return False

    # -----------------------------
    # 2) СРЕДНЯЯ ЗОНА
    # -----------------------------
    if zone == "middle":
        low, high = zones[1]
        zone_width = (high - low)
        
        bounces_num = count_bounces(zz, high)
        
        if bounces_num <= 2:
            if current_price > low + zone_width * 0.5:
                if verbose:
                    print(f"Price {current_price:.2f} is in middle zone but has only {bounces_num} bounces. Price above 0.5. Not buying.")     
                return False
        else:
            if current_price > low + zone_width * 0.768:
                if verbose:
                    print(f"Price {current_price:.2f} is in middle zone with {bounces_num} bounces. Price above 0.768. Not buying.")     
                return False
            
    # -----------------------------
    # 3) НИЖНЯЯ ЗОНА
    # -----------------------------
    if zone == "low":
        low, high = zones[0]
 
        bounces_num = count_bounces(zz, low)

        if bounces_num < 2:
            if verbose:
                print(f"Price {current_price:.2f} is in low zone but has only {bounces_num} bounces. Not buying.")     
            return False
        
    if verbose:
        print(f"Price {current_price:.2f} is in {zone} zone. Permitted to buy for other conditions.")

    return True


def analyze_key_levels(levels, zones, current_price):
    # Определяем зону
    zone_index = None
    for i, (low, high) in enumerate(zones):
        if low <= current_price <= high:
            zone_index = i
            break

    # Определяем поддержку и сопротивление
    if zone_index is not None:
        support, support_count = levels[zone_index]
        resistance, resistance_count = levels[zone_index + 1]
    elif current_price < zones[0][0]:
        support, support_count = levels[0]
        resistance, resistance_count = levels[1]
    else:
        support, support_count = levels[-2]
        resistance, resistance_count = levels[-1]

    # Расстояния
    dist_support = current_price - support
    dist_resistance = resistance - current_price

    # Сила уровней
    def strength(count):
        if count <= 2: return "fresh"
        if count <= 4: return "strong"
        return "weakening"

    return {
        "zone": zone_index,
        "support": support,
        "support_count": support_count,
        "support_strength": strength(support_count),
        "resistance": resistance,
        "resistance_count": resistance_count,
        "resistance_strength": strength(resistance_count),
        "dist_support": dist_support,
        "dist_resistance": dist_resistance,
    }


# Support functions for candlestick patterns
def body(o, c): return abs(c - o)
def upper(o, h, c): return h - max(o, c)
def lower(o, l, c): return min(o, c) - l

# 1. Hanging Man
def is_hanging_man(df):
    o, h, l, c = df.iloc[-1][['open','high','low','close']]
    b = body(o,c)
    return (
        c < o and
        lower(o,l,c) >= b * 2 and
        upper(o,h,c) <= b * 0.3
    )

# 2. Shooting Star
def is_shooting_star(df):
    o, h, l, c = df.iloc[-1][['open','high','low','close']]
    b = body(o,c)
    return (
        c < o and
        upper(o,h,c) >= b * 2 and
        lower(o,l,c) <= b * 0.3
    )

# 3. Gravestone Doji
def is_gravestone_doji(df):
    o, h, l, c = df.iloc[-1][['open','high','low','close']]
    return (
        body(o,c) <= (h-l) * 0.05 and
        upper(o,h,c) >= (h-l) * 0.6 and
        lower(o,l,c) <= (h-l) * 0.1
    )

# 4. Bearish Engulfing
def is_bearish_engulfing(df):
    o1, c1 = df.iloc[-2][['open','close']]
    o2, c2 = df.iloc[-1][['open','close']]
    return (
        c1 > o1 and
        c2 < o2 and
        o2 > c1 and
        c2 < o1
    )

# 5. Dark Cloud Cover (exldue from combine funtion, because it can be false signal in uptrend)
def is_dark_cloud(df):
    o1, c1 = df.iloc[-2][['open','close']]
    o2, c2 = df.iloc[-1][['open','close']]
    mid = (o1 + c1) / 2
    return (
        c1 > o1 and
        o2 > c1 and
        c2 < mid and
        c2 > o1
    )

# 6. Bearish Harami
def is_bearish_harami(df):
    o1, c1 = df.iloc[-2][['open','close']]
    o2, c2 = df.iloc[-1][['open','close']]
    return (
        c1 > o1 and
        o2 < c1 and o2 > o1 and
        c2 < c1 and c2 > o1
    )

# 7. Evening Star
def is_evening_star(df):
    o1, c1 = df.iloc[-3][['open','close']]
    o2, c2 = df.iloc[-2][['open','close']]
    o3, c3 = df.iloc[-1][['open','close']]
    return (
        c1 > o1 and
        abs(c2 - o2) < (c1 - o1) * 0.3 and
        c3 < o3 and
        c3 < (o1 + c1) / 2
    )

# 8. Falling Three Methods
def is_falling_three(df):
    o0, c0 = df.iloc[-5][['open','close']]
    o1, c1 = df.iloc[-4][['open','close']]
    o2, c2 = df.iloc[-3][['open','close']]
    o3, c3 = df.iloc[-2][['open','close']]
    o4, c4 = df.iloc[-1][['open','close']]
    return (
        c0 < o0 and
        c1 > o1 and c2 > o2 and c3 > o3 and
        o1 > c0 and c3 < o0 and
        c4 < o4 and c4 < c0
    )

# 9. Three Black Crows
def is_three_black_crows(df):
    o1, c1 = df.iloc[-3][['open','close']]
    o2, c2 = df.iloc[-2][['open','close']]
    o3, c3 = df.iloc[-1][['open','close']]
    return (
        c1 < o1 and c2 < o2 and c3 < o3 and
        o2 < o1 and o3 < o2
    )

# 10. Bearish Marubozu
def is_bearish_marubozu(df, tolerance=0.1):
    o, h, l, c = df.iloc[-1][['open','high','low','close']]
    b = body(o, c)
    total = h - l
    
    if c >= o:
        return False  # свеча должна быть красной

    return (
        upper(o,h,c) <= total * tolerance and
        lower(o,l,c) <= total * tolerance and
        b >= total * (1 - tolerance)
    )

def breaks_min_close_last5(df):
    # нужно минимум 6 свечей: текущая + 5 предыдущих
    if len(df) < 6:
        return False

    current_close = df.iloc[-1]['сlose']
    last5_min_close = df.iloc[-6:-1]['сlose'].min()

    return current_close < last5_min_close


# -----------------------------
# Объединённая функция
# -----------------------------
def is_any_bearish_pattern(df):
    # Проверка длины, чтобы не ловить ошибки
    n = len(df)

    # 1-свечные паттерны
    if is_hanging_man(df): 
        return True

    if is_shooting_star(df): 
        return True

    if is_gravestone_doji(df): 
        return True

    if is_bearish_marubozu(df):
        return True

    # 2-свечные паттерны
    if n >= 2 and is_bearish_engulfing(df): 
        return True

    if n >= 2 and is_bearish_harami(df): 
        return True

    # 3-свечные паттерны
    if n >= 3 and is_evening_star(df): 
        return True

    if n >= 3 and is_three_black_crows(df): 
        return True

    # 5-свечные паттерны
    if n >= 5 and is_falling_three(df): 
        return True
    
    # 6-свечные паттерны
    if n >= 6 and breaks_min_close_last5(df):
        return True

    return False


def bearish_last_7_candles(df):
    # Если меньше 7 свечей — проверяем сколько есть
    n = len(df)
    lookback = min(7, n)

    # Берём последние N свечей
    for i in range(1, lookback + 1):
        sub_df = df.iloc[-i-5 : -i]  # срез от -i до конца
        if is_any_bearish_pattern(sub_df):
            return True

    return False

def bearish_last_k_candles(df, k=5):
    # Если меньше 7 свечей — проверяем сколько есть
    n = len(df)
    lookback = min(k, n)

    # Берём последние N свечей
    for i in range(1, lookback + 1):
        sub_df = df.iloc[-i-5 : -i]  # срез от -i до конца
        if is_any_bearish_pattern(sub_df):
            return True

    return False

#--------------------------------------------------------------
# Function for buying based on 1 hours handes patterns
#--------------------------------------------------------------
#--------------------------------------------------------------
# 1. Bullish Engulfing (бычье поглощение)
def is_bullish_engulfing(df):
    o1, h1, l1, c1 = df.iloc[-1][['open','high','low','close']]  # текущая
    o2, h2, l2, c2 = df.iloc[-2][['open','high','low','close']]  # предыдущая

    return (
        c2 < o2 and      # предыдущая свеча красная
        c1 > o1 and      # текущая зелёная
        o1 <= c2 and     # тело текущей открывается не выше закрытия предыдущей
        c1 >= o2         # и закрывается не ниже открытия предыдущей (поглощение)
    )


# 2. Hammer (молот)
def is_hammer(df):
    o, h, l, c = df.iloc[-1][['open','high','low','close']]
    b = body(o, c)

    return (
        lower(o, l, c) >= b * 2 and   # длинная нижняя тень
        upper(o, h, c) <= b * 0.3 and # маленькая верхняя
        c > o                         # желательно зелёная
    )


# 3. Morning Star (утренняя звезда)
def is_morning_star(df):
    o3, h3, l3, c3 = df.iloc[-3][['open','high','low','close']]  # старая (1-я)
    o2, h2, l2, c2 = df.iloc[-2][['open','high','low','close']]  # середина
    o1, h1, l1, c1 = df.iloc[-1][['open','high','low','close']]  # текущая (3-я)

    return (
        c3 < o3 and                              # 1-я свеча красная
        body(o2, c2) <= body(o3, c3) * 0.5 and   # маленькая 2-я свеча
        c1 > o1 and                              # 3-я зелёная
        c1 >= (o3 + c3) / 2                      # закрытие выше середины 1-й
    )


# 4. Piercing Line (просвет в облаках)
# 
def is_piercing_line(df):
    o1, h1, l1, c1 = df.iloc[-1][['open','high','low','close']]  # текущая
    o2, h2, l2, c2 = df.iloc[-2][['open','high','low','close']]  # предыдущая

    mid = (o2 + c2) / 2

    return (
        c2 < o2 and      # предыдущая свеча красная
        o1 < l2 and      # гэп вниз на открытии текущей
        c1 > mid and     # закрытие выше середины тела предыдущей
        c1 < o2          # но не выше её открытия
    )


# 5. Bullish Harami (бычье харами)
# Small body (current) is contained within the previous large body, and previous is red, current is green
def is_bullish_harami(df):
    o1, h1, l1, c1 = df.iloc[-1][['open','high','low','close']]  # текущая (малая)
    o2, h2, l2, c2 = df.iloc[-2][['open','high','low','close']]  # предыдущая (большая)

    return (
        c2 < o2 and                              # предыдущая свеча красная
        body(o1, c1) < body(o2, c2) * 0.5 and    # маленькое тело текущей
        o1 >= c2 and o1 <= o2 and
        c1 >= c2 and c1 <= o2                   # полностью внутри тела предыдущей
    )


def is_any_bullish_pattern(df, ticker='ticker_name', verbal=False):
    
    n = len(df)
    
    if n >= 1:
        if is_hammer(df): 
            if verbal: green.print(f'  Hammer detected for {ticker}')
            return True
    
    if n > 2:
        if is_bullish_engulfing(df): 
            if verbal: green.print(f'  Bullish Engulfing detected for {ticker}')
            return True

        if is_piercing_line(df): 
            if verbal: green.print(f'  Piercing Line detected for {ticker}')
            return True

        if is_bullish_harami(df): 
            if verbal: green.print(f'  Bullish Harami detected for {ticker}')
            return True
    
    if n > 3:
        if is_morning_star(df): 
            if verbal: green.print(f'  Morning Star detected for {ticker}')
            return True

def bullish_pattern_last_5_candles(df, ticker='ticker_name', verbal=False):
    # Если меньше 5 свечей — проверяем сколько есть
    n = len(df)
    lookback = min(5, n)

    # Берём последние N свечей
    for i in range(1, lookback + 1):
        sub_df = df.iloc[-i-5 : -i]  # срез от -i до конца
        if is_any_bullish_pattern(sub_df, ticker=ticker, verbal=verbal):
            return True

    return False