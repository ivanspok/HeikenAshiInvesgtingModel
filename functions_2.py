import numpy as np
# import pandas_ta as ta

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