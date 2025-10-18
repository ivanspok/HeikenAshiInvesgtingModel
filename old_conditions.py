# version 1 1230 conditon:
if False:
    def stock_buy_condition_1230(df, df_1m):
        i_1m = df_1m.shape[0] - 1
        ha_cond = df_1m['ha_colour'].iloc[-1] == 'green' and df_1m['ha_colour'].iloc[-2] == 'green' \
        and number_red_candles(df_1m, i_1m, k=11) > 5

        condition = False
        if df.index[-1].hour == 13 and df.index[-2].hour == 12:
        # moomoo time: -1: 14:30; -2: 13:30: -3: 12:30; -4: 11:30; -5: 10:30
        # yahho time: 
        # candle is green
        # candle pct more than 0 
        # more taht close 9:30
        # sum pct 10:30 - 12:30  more than 0 
        # 9:30 less than -0l77
        # improve ha_cond for this case!!!!!!!! 
            if ha_cond \
                and df['close'].iloc[-2] > df['open'].iloc[-2] \
                and df['pct'].iloc[-2] > 0 \
                and df['close'].iloc[-2] > df['close'].iloc[-5] \
                and df['pct'].iloc[-4:-1].sum() > 0\
                and (df['pct'].iloc[-5] < -0.58 \
                    or df['close'].iloc[-6] / df['close'].iloc[-5] > 1.006):
                condition = True
        c.green_red_print(condition, 'buy_condition_1230')
        return condition

def stock_buy_condition_1230(df, df_1m, display=False):
  i_1m = df_1m.shape[0] - 1
  ha_cond = df_1m['ha_colour'].iloc[-1] == 'green' and df_1m['ha_colour'].iloc[-2] == 'green' \
    and number_red_candles(df_1m, i_1m, k=11) > 5

  condition = False

  safe_cond_1 = df_1m['close'].iloc[-1] / df['low'].iloc[-2] < 1.005 # the price hasn't risen more that 0.3%

  if df.index[-1].hour == 12 and df.index[-2].hour == 11:
    # dropping during opening/closing and assuming rising after 12:30
    # moomoo time: -1: 13:30; -2: 12:30: -3: 11:30; -4: 10:30; -5: 9:30
    # yahoo time: -1: 12:30; -2: 11:30; -3: 10:30; -4: 9:30; -5: 15:30
    # previuos candle is green
    # previous candle pct more than 0 
    # previous candle more than close 10:30
    # candle pct more than 0.15
    # 9:30 or 15:30 less than -0.58 
    cond_1 = df['close'].iloc[-2] > df['open'].iloc[-2]
    cond_2 = df['pct'].iloc[-2] > 0
    cond_3 = df['close'].iloc[-2] > df['close'].iloc[-3]
    cond_4 = df['pct'].iloc[-1] > 0.15
    cond_5 = (df['pct'].iloc[-4] < -0.58 or df['close'].iloc[-5] / df['close'].iloc[-4] > 1.006)
    cond_6 = (df['pct'].iloc[-5] < -0.58 or df['close'].iloc[-6] / df['close'].iloc[-5] > 1.006)

    if display:
      warning.print('1230 conditions parameters:')
      c.green_red_print(cond_1, 'cond_1')
      c.green_red_print(cond_2, 'cond_2')
      c.green_red_print(cond_3, 'cond_3')
      c.green_red_print(cond_4, 'cond_4')
      c.green_red_print(cond_5, 'cond_5')
      c.green_red_print(cond_6, 'cond_6')

    if ha_cond \
      and safe_cond_1 \
      and df['close'].iloc[-2] > df['open'].iloc[-2] \
      and df['pct'].iloc[-2] > 0 \
      and df['close'].iloc[-2] > df['close'].iloc[-3] \
      and df['pct'].iloc[-1] > 0.15 \
      and (cond_5 or cond_6):
      condition = True

  if df.index[-1].hour == 13 and df.index[-2].hour == 12:
    # dropping during opening/closing and assuming rising after 13:30
    # moomoo time: -1: 13:30; -2: 12:30: -3: 11:30; -4: 10:30; -5: 9:30
    # yahoo time: -1: 12:30; -2: 11:30; -3: 10:30; -4: 9:30; -5: 15:30
    # previuos candle is green
    # previous candle pct more than 0 
    # previous candle more than close 9:30
    # candle sum pct 9:30 - 11:30 more than 0
    # 9:30 or 15:30 less than -0.58 
    if ha_cond \
      and safe_cond_1 \
      and df['close'].iloc[-2] > df['open'].iloc[-2] \
      and df['pct'].iloc[-2] > 0 \
      and df['close'].iloc[-2] > df['close'].iloc[-5] \
      and df['pct'].iloc[-4:-1].sum() > 0\
      and (df['pct'].iloc[-5] < -0.58 \
          or df['close'].iloc[-6] / df['close'].iloc[-5] > 1.006):
      condition = True
  if condition:
    c.green_red_print(condition, 'buy_condition_1230')
  return condition

def stock_buy_condition_930_47(df):
    '''
        Parameters: df, interval 1m

        Returns:
        condition
    '''
    condition = False
    buy_price = 0

    current_minute = datetime.now().astimezone().minute 
    market_opening = datetime.now().astimezone().hour == 23 \
        and current_minute > 30 and current_minute <= 47

    i = df.shape[0] - 1

    max = maximum(df, 20)
    min = minimum(df, 20)
    cond_1 = max / df['close'].iloc[-1] > 1.0085
    cond_2 = (df['ha_colour'].iloc[-4 : -2] == 'red').all()
    cond_3 = df['ha_colour'].iloc[-2] == 'green'
    cond_4 = df['ha_colour'].iloc[-1] == 'green'
    cond_5 = (df['close'].iloc[-20 : -1].min() - min) / (max - min + 0.0001) < 0.4

    if market_opening:
        warning.print('930_47 conditions parameters:')
        c.green_red_print(market_opening, 'market_opening')
        c.green_red_print(cond_1, 'cond_1')
        c.green_red_print(cond_2, 'cond_2')
        c.green_red_print(cond_3, 'cond_3')
        c.green_red_print(cond_4, 'cond_4')
        c.green_red_print(cond_5, 'cond_5')

    if market_opening \
        and cond_1 and cond_2 and cond_3 and cond_4 and cond_5:
        condition = True

    c.green_red_print(condition, 'buy_condition_930_47')
    return condition

def stock_buy_condition_speed_norm100(df, df_1m, df_stats, display=False):
  i_1m = df_1m.shape[0] - 1
  i = df.shape[0] - 1
  ha_cond = df_1m['ha_colour'].iloc[-1] == 'green' and df_1m['ha_colour'].iloc[-2] == 'green' \
    and number_red_candles(df_1m, i_1m, k=11) > 5

  condition = False

  cond_value_1 = df_stats[ticker]['speed100_avg30'].iloc[-1]
  cond_value_2 = df['close'].iloc[-2] - df['close'].iloc[-50 : -1].min()
  cond_value_3 = df_stats[ticker]['norm100'].iloc[-1]
  cond_value_4 = maximum(df, i, 500) / df['close'].iloc[-2]
  cond_value_5 = maximum(df, i, 7) / df['close'].iloc[-2]

  cond_1 = cond_value_1 >= 0.7 and cond_value_1 <= 1.1
  cond_2 = cond_value_2 <= 0   
  cond_3 = cond_value_3 >= 0.7
  cond_4 = cond_value_4 > 1.2
  cond_5 = cond_value_5 < 1.035
  
  if display:
    warning.print('speed_norm100 conditions parameters:')
    c.green_red_print(cond_1, f'cond_1 speed100_avg30: {cond_value_1:.2f}')
    c.green_red_print(cond_2, f'cond_2 {cond_value_2:.2f}')
    c.green_red_print(cond_3, f'cond_3 {cond_value_3:.2f}')
    c.green_red_print(cond_4, f'cond_4 {cond_value_4:.2f}')
    c.green_red_print(cond_5, f'cond_5 {cond_value_5:.2f}')

  if ha_cond \
    and cond_1 \
    and cond_2 \
    and cond_3 \
    and cond_4 \
    and cond_5:

    condition = True
  if condition:          
    c.green_red_print(condition, 'buy_condition_speed_norm100')
  return condition
# version from 13/03/2025
def stock_buy_condition_before_market_open_old(df, df_1m, df_stats, display=False):
  i_1m = df_1m.shape[0] - 1
  i = df.shape[0] - 1
  ha_cond = df_1m['ha_colour'].iloc[-1] == 'green' and df_1m['ha_colour'].iloc[-2] == 'green' \
    and number_red_candles(df_1m, i_1m, k=11) > 5

  condition = False
  condition_type = 'No cond'

  if (df.index[i - 1].hour in [14, 15, 16]):

    cond_value_1 = df_stats[ticker]['speed100_avg30'].iloc[-1]
    cond_value_2 = cond_value_1
    # cond_value_3 = df['close'].iloc[-2] - df['close'].iloc[-5 : -1].min()
    cond_value_4 = df_stats[ticker]['norm100'].iloc[-1]
    cond_value_5 = maximum(df, i, 500) / df['close'].iloc[-2]
    cond_value_6 = maximum(df, i, 7) / df['close'].iloc[-2]

    cond_1 = cond_value_1 >= 0.7
    cond_2 = cond_value_2 <= 1.1
    # cond_3 = cond_value_3 <= 0
    cond_3 = True
    cond_4 = cond_value_4 >= 0.7
    
    # Buy with -0.4%
    cond_51 = cond_value_5 > 1.17
    cond_61 = cond_value_6 < 1.02
    # Buy with -0.7% 
    cond_52 = cond_value_5 > 1.05
    cond_62 = cond_value_6 < 1.03

    if display:
      warning.print('before_market_open conditions parameters:')
      c.green_red_print(cond_1, f'cond_1 {cond_value_1:.2f}')
      c.green_red_print(cond_2, f'cond_2 {cond_value_2:.2f}')
      c.green_red_print(cond_4, f'cond_4 {cond_value_4:.2f}')
      c.green_red_print(cond_51, f'cond_51 {cond_value_5:.2f}')
      c.green_red_print(cond_61, f'cond_61 {cond_value_6:.2f}')
      c.green_red_print(cond_52, f'cond_52 {cond_value_5:.2f}')
      c.green_red_print(cond_62, f'cond_62 {cond_value_6:.2f}')

    if cond_1 \
      and cond_2 \
      and cond_3 \
      and cond_4:

      if cond_52 \
        and cond_62:
        condition_type = 'before_market_open_2' # Buy with -0.7% 
        condition = True

      # More strict condition goes last
      if cond_51 \
        and cond_61:
        condition_type = 'before_market_open_1' # Buy with -0.4%
        condition = True
  if condition: 
    c.green_red_print(condition, condition_type)
  return condition, condition_type

def stock_buy_condition_speed_norm100(df, df_1m, df_stats, display=False):
  i_1m = df_1m.shape[0] - 1
  i = df.shape[0] - 1
  ha_cond = df_1m['ha_colour'].iloc[-1] == 'green' and df_1m['ha_colour'].iloc[-2] == 'green' \
    and number_red_candles(df_1m, i_1m, k=11) > 5

  condition = False

  cond_value_1 = df_stats[ticker]['speed100_avg30'].iloc[-1]
  cond_value_2 = df['close'].iloc[-2] - df['close'].iloc[-50 : -1].min()
  cond_value_3 = df_stats[ticker]['norm100'].iloc[-1]
  cond_value_4 = maximum(df, i, 500) / df['close'].iloc[-2]
  cond_value_5 = maximum(df, i, 7) / df['close'].iloc[-2]

  cond_1 = cond_value_1 >= 0.7 and cond_value_1 <= 1.1
  cond_2 = cond_value_2 <= 0   
  cond_3 = cond_value_3 >= 0.7
  cond_4 = cond_value_4 > 1.2
  cond_5 = cond_value_5 < 1.035
  
  if display:
    warning.print('speed_norm100 conditions parameters:')
    c.green_red_print(cond_1, f'cond_1 speed100_avg30: {cond_value_1:.2f}')
    c.green_red_print(cond_2, f'cond_2 {cond_value_2:.2f}')
    c.green_red_print(cond_3, f'cond_3 {cond_value_3:.2f}')
    c.green_red_print(cond_4, f'cond_4 {cond_value_4:.2f}')
    c.green_red_print(cond_5, f'cond_5 {cond_value_5:.2f}')

  if ha_cond \
    and cond_1 \
    and cond_2 \
    and cond_3 \
    and cond_4 \
    and cond_5:

    condition = True
  if condition:          
    c.green_red_print(condition, 'buy_condition_speed_norm100')
  return condition

def stock_buy_condition_MA50_MA5_old_2(df, df_1m, df_stats, display=False):
  '''
  if gradient MA50 1hour > 0 \\
  and MA5 1hour - local minimum \\
  and rise no more than 0.55% \\
  and ha_cond
  buy_price: to have MA5[-1] > MA5[-2]: \\
           df['close'].iloc[-1] should more than df['close'].iloc[-6]\\ 
           and below this value on 0.5%
  sell_orders: 2 stop_limit loss orders, trailing order with modification \\ 
  order life time: 1hour \\
  profit value: trailing order with modification \\ 
  lose value:  -1% and -2% limit, and stop market if below -2.2%\\
  buy price: df_1m['close'].iloc[-1] + 0.05% during normal hours
  after-hours: NOT BUY
  '''
  condition = False
  condition_type = 'MA50_MA5'

  i_1m = df_1m.shape[0] - 1
  i = df.shape[0] - 1
  ha_cond = df_1m['ha_colour'].iloc[-1] == 'green' and df_1m['ha_colour'].iloc[-2] == 'green' \
    and number_red_candles(df_1m, i_1m, k=11) > 5

  cond_1 = (df['MA50'].iloc[-1] > [df['MA50'].iloc[-75], 
                                   df['MA50'].iloc[-50],
                                   df['MA50'].iloc[-25],
                                   df['MA50'].iloc[-5],
                                  ]).all()
  
  cond_2 = df_1m['close'].iloc[-1] >= df['close'].iloc[-6] * 0.995 \
            and df['MA5'].iloc[-2] <= df['MA5'].iloc[-3] \
            and df['MA5'].iloc[-3] <= df['MA5'].iloc[-4] \
            and df['MA5'].iloc[-4] <= df['MA5'].iloc[-5] \
            and df['MA5'].iloc[-6] <= df['MA5'].iloc[-7] \
            and df['MA5'].iloc[-7] <= df['MA5'].iloc[-8]
                                               
  # cond_2 = df['MA5'].iloc[-1] >= df['MA5'].iloc[-2] \
  #           and (df['MA5'].iloc[-2] < [df['MA5'].iloc[-3],
  #                                      df['MA5'].iloc[-4],
  #                                      df['MA5'].iloc[-5]
  #                                     ]).all()
  rise_value = df_1m['close'].iloc[-1] / df['close'].iloc[-3].min() # first version df_1m['close'].iloc[-1] / df['low'].iloc[-3].min()
  cond_3 = rise_value < 1.0055
  cond_4 = df_1m['close'].iloc[-1] / df['close'].iloc[-6] < 1.0055

  if display:
    warning.print('MA50_MA5 conditions parameters:')
    c.green_red_print(cond_1, f'cond_1')
    c.green_red_print(cond_2, f'cond_2')
    c.green_red_print(cond_3, f'cond_3')
    c.green_red_print(cond_4, f'cond_4')
    c.green_red_print(ha_cond, f'ha_cond')

  if cond_1 \
    and cond_2 \
    and cond_3 \
    and cond_4 \
    and ha_cond:
    condition = True

  if condition: 
    c.green_red_print(condition, condition_type)

  return condition

def stock_buy_condition_MA50_MA5_old(df, df_1m, df_stats, ticker, display=False):
  '''
  if gradient MA5 1hour > 0 \\
  and MA5 1hour - MA50 1 hour > 0%
  and MA5 1hour - MA50 1 hour < 0.3%

  buy_price: to have MA5[-1] > MA5[-2]: \\
           df['close'].iloc[-1] should more than df['close'].iloc[-6]\\ 
           and below this value on 0.5%
  sell_orders: 2 stop_limit loss orders, trailing order with modification \\ 
  order life time: 1hour \\
  profit value: trailing order with modification \\ 
  lose value:  -1% and -2% limit, and stop market if below -2.2%\\
  buy price: df_1m['close'].iloc[-1] + 0.05% during normal hours
  after-hours: NOT BUY
  '''
  global market_value, total_market_value ,total_market_value_2m , \
    total_market_value_5m, total_market_value_30m, total_market_value_60m, \
    total_market_direction_60m, total_market_direction_10m
  
  condition = False
  condition_type = 'MA50_MA5'

  i_1m = df_1m.shape[0] - 1
  i = df.shape[0] - 1
  ha_cond = df_1m['ha_colour'].iloc[-1] == 'green' and df_1m['ha_colour'].iloc[-2] == 'green' \
    and number_red_candles(df_1m, i_1m, k=11) > 5
  
  prediction_rise = 1.005
  grad_MA5 = df_1m['close'].iloc[-1] * prediction_rise >= df['close'].iloc[-6] \
            and df['MA5'].iloc[-2] >= df['MA5'].iloc[-3]

  deltaMA5_MA50 = df['MA5'].iloc[-1] / df['MA50'].iloc[-1]    
  deltaMA5_MA50_b3 = df['MA5'].iloc[-3] / df['MA50'].iloc[-3]    

  # deltaMA5_MA50_prediction = df['MA5'].iloc[-2] + (df_1m['close'].iloc[-1] * prediction_rise - df_1m['close'].iloc[-6]) / 5
  MA5_prediction = df['MA5'].iloc[-2] + (df_1m['close'].iloc[-1] * prediction_rise - df['close'].iloc[-6]) / 5
  MA50_prdediction = df['MA50'].iloc[-2] + (df_1m['close'].iloc[-1] * prediction_rise - df['close'].iloc[-51]) / 50
  deltaMA5_MA50_prediction = MA5_prediction / MA50_prdediction

  # cond_1 = deltaMA5_MA50 >= 1.000 and deltaMA5_MA50 <= 1.003 # first version
  cond_1 = deltaMA5_MA50_prediction >= 1.000 and deltaMA5_MA50_prediction <= 1.003
  cond_value_1 = deltaMA5_MA50_prediction
  cond_2 = deltaMA5_MA50_b3 < 0.999
  cond_value_2 = deltaMA5_MA50_b3

  MA50_max120_i = 120 - np.argmax(df['MA50'].iloc[-120:]) #0 the first and 120 is the last
  cond_3 = MA50_max120_i < 5 or MA50_max120_i > 60
  cond_value_3 = MA50_max120_i
  # np.amax([df['close'].iloc[0 : k].max(), df['open'].iloc[0 :k].max(), df['close'].iloc[i], df['open'].iloc[i]])

  cond_value_4 = df_1m['close'].iloc[-1] / df['MA50'].iloc[-1] 
  cond_4 = cond_value_4 < 1.005
  cond_5 = df_1m['MA5'].iloc[-1] > df_1m['MA20'].iloc[-1]
  cond_value_6 = df_1m['close'].iloc[-1] / minimum(df, i, 24)
  cond_6 = cond_value_6 < 1.03 # current price hasn't risen more than 3% last 24 hours

  if deltaMA5_MA50 < 0.99 \
    or deltaMA5_MA50 > 1.01 \
    or not cond_2 \
    or (MA50_max120_i > 7 and  MA50_max120_i < 57):
    # exclude company from _list optimal list
    warning.print(f'{ticker} is excluding from current optimal stock list')
    exclude_time_dist[ticker] = datetime.now()
    stock_name_list_opt.remove(ticker)
    warning.print(f'Optimal stock name list len is {len(stock_name_list_opt)}')
    market_value = -1 # start offset
    total_market_value = -99  
    # Calculate market direction
    total_market_value_2m = 0
    total_market_value_5m = 0
    total_market_value_30m = 0
    total_market_value_60m = 0
    total_market_direction_10m = 0
    total_market_direction_60m = 0

  if display:
    warning.print('MA50_MA5 conditions parameters:')
    c.green_red_print(grad_MA5, f'grad_MA5')
    c.green_red_print(cond_1, f'cond_1 (deltaMA5_M50 prediction), {cond_value_1:.3f}')
    c.green_red_print(cond_2, f'cond_2 (deltaMA5_M50 3 hours before), {cond_value_2:.3f}')
    c.green_red_print(cond_3, f'cond_3 (distanse from 120 max), {cond_value_3:.3f}')
    c.green_red_print(cond_4, f'cond_4 (percantage above 1h MA50), {cond_value_4:.3f}')
    c.green_red_print(cond_5, f'cond_5 (current price more than 1m MA20)')
    c.green_red_print(cond_6, f'cond_6 (percantage above last 24 hours minimum), {cond_value_6:.3f}')
    c.green_red_print(ha_cond, f'ha_cond')

  if cond_1 \
    and grad_MA5 \
    and cond_2 \
    and cond_3 \
    and cond_4 \
    and cond_5:
    condition = True

  if condition: 
    c.green_red_print(condition, condition_type)

  return condition

def stock_buy_condition_MA50_MA5_old_3(df, df_1m, df_stats, ticker, display=False):
  '''
  buy condition:
  if gradient MA50 1hour > 0 \\
  and gradient MA5 1hour > 0 \\
  and MA5 1hour > MA50 1hour \\
  and Square MA5 - MA50 interval 30 is negative
  and MA5 1hour - MA50 1 hour < 1%
  and 
  (change in gradient MA50 \\
    or MA5 crossing MA50) \\
  
  buy_price: to have MA5[-1] > MA5[-2]: \\
           df['close'].iloc[-1] should more than df['close'].iloc[-6]\\ 
           and below this value on 0.5%
  sell_orders: 2 stop_limit loss orders, trailing order with modification \\ 
  order life time: 1hour \\
  profit value: trailing order with modification \\ 
  lose value:  -1% and -2% limit, and stop market if below -2.2%\\
  buy price: df_1m['close'].iloc[-1] + 0.05% during normal hours
  after-hours: NOT BUY
  
  '''
  global market_value, total_market_value ,total_market_value_2m , \
    total_market_value_5m, total_market_value_30m, total_market_value_60m, \
    total_market_direction_60m, total_market_direction_10m
  
  condition = False
  condition_type = 'MA50_MA5'

  i_1m = df_1m.shape[0] - 1
  i = df.shape[0] - 1
  
  # prediction_rise = 1.005
  # grad_MA5 = df_1m['close'].iloc[-1] * prediction_rise >= df['close'].iloc[-6] \
  #           and df['MA5'].iloc[-2] >= df['MA5'].iloc[-3]
  grad_MA5 = df_1m['close'].iloc[-1] >= df['close'].iloc[-6] \
            and df['MA5'].iloc[-2] >= df['MA5'].iloc[-3] \
            # and df['MA5'].iloc[-3] >= df['MA5'].iloc[-4] # new from 22/05/2025
  
  grad_MA5_1m = df_1m['close'].iloc[-1] >= df_1m['close'].iloc[-6] \
            and df_1m['MA5'].iloc[-2] >= df_1m['MA5'].iloc[-3]
            
  grad_MA50 = df_1m['close'].iloc[-1] >= df['close'].iloc[-51] \
              and df['MA50'].iloc[-2] >= df['MA50'].iloc[-3]
              
  grad_MA50_1m = df_1m['close'].iloc[-1] >= df_1m['close'].iloc[-51] \
                  and df_1m['MA50'].iloc[-2] >= df_1m['MA50'].iloc[-3] \
                  and df_1m['MA50'].iloc[-3] >= df_1m['MA50'].iloc[-4] \
                    
              
  grad_MA50_prev = df['MA50'].iloc[-4] <= df['MA50'].iloc[-5] \
                and df['MA50'].iloc[-5] <= df['MA50'].iloc[-6] \

  deltaMA5_MA50 = df['MA5'].iloc[-1] / df['MA50'].iloc[-1]    
  deltaMA5_MA50_b2 = df['MA5'].iloc[-2] / df['MA50'].iloc[-2]    
  deltaMA5_MA50_b3 = df['MA5'].iloc[-3] / df['MA50'].iloc[-3]    
  
  deltaMA5_MA50_1min = df_1m['MA5'].iloc[-1] / df_1m['MA50'].iloc[-1]
  deltaMA5_MA50_1min_b3 = df_1m['MA5'].iloc[-3] / df_1m['MA50'].iloc[-3]
  
  square_MA5_MA50_30 = (df['MA5'].iloc[-31:] / df['MA50'].iloc[-31:]).sum() - 31
  
  cond_value_1 = grad_MA50
  cond_1 = grad_MA50
  cond_value_2 = grad_MA5
  cond_2 = grad_MA5
  cond_value_3 = deltaMA5_MA50
  cond_3 = cond_value_3 > 1.000 and cond_value_3 < 1.02
  cond_value_4 = square_MA5_MA50_30
  cond_4 = cond_value_4 < - 0.175
  cond_5 = grad_MA50 > 0 and grad_MA50_prev < 0 # change in the gradient of MA50
  cond_6 = deltaMA5_MA50_b2 < 0.999 and (df['pct'].iloc[-3:] < 1).all() # MA5 crossing MA50
  cond_value_7 = df['MA5'].iloc[-1] / df['MA50'].iloc[-7:].min()
  cond_7 = cond_value_7 >= 1 and cond_value_7 < 1.004 and cond_4 # close to local minimum
  cond_8 =  df['MA5'].iloc[-1] / df['MA50'].iloc[-1] >= 1.02 \
            and number_red_candles(df, i, k=7) <= 3 \
            and df['ha_colour'].iloc[-1] == 'green' \
            and df['ha_colour'].iloc[-2] == 'green' \
            and df['ha_colour'].iloc[-3] == 'red'
    
  cond_9 = df['MA120'].iloc[-1] >= df['MA120'].iloc[-2] \
          and df['MA120'].iloc[-2] >= df['MA120'].iloc[-3] \
          and df['MA120'].iloc[-3] >= df['MA120'].iloc[-4] \
  # or good local minimum MA5: more than 5 red ha candles last 12 candles and grad MA5 > 0          
  cond_10 = number_red_candles(df, i, k=8) > 5 \
            and grad_MA5 \
            and df_1m['close'].iloc[-1] / df['low'].iloc[-5:].min() < 1.0085 \
            and grad_MA5_1m
            
  # MA50_1h > 0 and MA120_1h > 0 and MA5_1h > 0 and MA5_1m crossing MA50_1m and MA50_1m > 0
  cond_11 = deltaMA5_MA50_1min_b3 < 0.999 \
            and deltaMA5_MA50_1min > 1.0015 \
            and grad_MA50_1m 
            
  cond_12 = (df_1m['pct'].iloc[-7:] < 1).all()
  # cond_value_12 = df_1m['close'].iloc[-1] / mimimum(df, i, 24)
  # cond_12 = cond_value_12 < 1.03 # current price hasn't risen more than 3% last 24 hours
  
  MA5_MA120_1m_120 = (df_1m['MA5'] / df_1m['MA120'] - 1).rolling(window=120).sum()
  sum_before_crossing = 0
  for j in range(MA5_MA120_1m_120.shape[0] - 1, 1, -1):
    if MA5_MA120_1m_120.iloc[j] < 0:
      sum_before_crossing += MA5_MA120_1m_120.iloc[j]
    if MA5_MA120_1m_120.iloc[j - 1] >= 0 \
      and MA5_MA120_1m_120.iloc[j] <= 0:
      break 
  MA5_MA120_120 = (df['MA5'] / df['MA120'] - 1).rolling(window=120).sum()
  sum_MA5_MA120_120 = MA5_MA120_120.rolling(window=30).sum()
  
  
  cond_13 = MA5_MA120_1m_120.iloc[-1] > 0 \
           and MA5_MA120_1m_120.iloc[-2] <= 0 \
           and MA5_MA120_1m_120.iloc[-3] <= 0 \
           and MA5_MA120_1m_120.iloc[-10] <  0 \
           and sum_before_crossing < -50 \
           and grad_MA50_1m > 0 \
           and grad_MA5_1m > 0 
           
  cond_14 = df['close'].iloc[-2] > df['open'].iloc[-2] \
            or df['close'].iloc[-1] > df['high'].iloc[-2] # new from 10/06/2025
 
  dist_from_MA5_1m = df_1m['close'].iloc[-1] / df['MA5'].iloc[-1]
           
  # if square_MA5_MA50_30 < -50 \
  #   or square_MA5_MA50_30 > 80 \
  #   or deltaMA5_MA50 < 0.97:
  # if not cond_9:
  if sum_MA5_MA120_120.iloc[-1] > 10:
    
    # exclude company from _list optimal list
    warning.print(f'{ticker} is excluding from current optimal stock list')
    exclude_time_dist[ticker] = datetime.now()
    stock_name_list_opt.remove(ticker)
    warning.print(f'Optimal stock name list len is {len(stock_name_list_opt)}')
    market_value = -1 # start offset
    total_market_value = -99  
    # Calculate market direction
    total_market_value_2m = 0
    total_market_value_5m = 0
    total_market_value_30m = 0
    total_market_value_60m = 0
    total_market_direction_10m = 0
    total_market_direction_60m = 0

#   if display:
#     warning.print('MA50_MA5 conditions parameters:')
#     c.print('AND CONDITIONS:', color='yellow')
#     c.green_red_print(cond_9, f'cond_9 (grad_MA120 > 0)')
#     c.green_red_print(cond_1, f'cond_1 (grad_MA50 > 0), {cond_value_1:.3f}')
#     c.green_red_print(cond_2, f'cond_2 (grad_MA5 > 0), {cond_value_2:.3f}')
#     c.green_red_print(grad_MA5_1m, f'(grad_MA5_1m > 0), {int(grad_MA5_1m):.3f}')
#     c.green_red_print(cond_3, f'cond_3 (deltaMA5_MA50 > 1 and deltaMA5_MA50 < 1.02), {cond_value_3:.3f}')
#     # c.green_red_print(cond_4, f'cond_4 (square_MA5_MA50_30 < 0), {cond_value_4:.3f}')
#     c.print('OR CONDITIONS:', color='yellow')
#     c.green_red_print(cond_5, f'cond_5 (grad_MA50 > 0 and grad_MA50_prev < 0)')
#     c.green_red_print(cond_6, f'cond_6 (deltaMA5_MA50_b2 < 0.999), {deltaMA5_MA50_b2:.3f}') # crossing MA5 and MA50
#     c.green_red_print(cond_7, f'cond_7 (close to local minimum, no more than 0.4%), {cond_value_7:.3f}') # other condition for crossing MA5 and MA50
#     # c.green_red_print(cond_10, f'cond_10 (number ha red candles condition), {number_red_candles(df, i, k=8)}')
#     c.green_red_print(cond_11, f'cond_11 (deltaMA5_MA50_1min_b3 ({deltaMA5_MA50_1min_b3:.4f}) < 0.999 \
# and deltaMA5_MA50_1min ({deltaMA5_MA50_1min:.4f}) > 1.0015)')
#     c.green_red_print(cond_13, f'cond_13 (MA5_MA50_1m_120 ({MA5_MA120_1m_120.iloc[-1]:.4f})')
 
#  and dist_from_MA5_1m < 1.005
  # if cond_1 \
  #   and cond_2 \
  #   and cond_3 \
  #   and cond_9 \
  #   and cond_12 \
  #   and grad_MA5_1m \
  #   and cond_14 \
  #   and (cond_6 and not(cond_7)):  # changed from 10/06/2025 from (cond_6 or cond_7)
  #     condition = True 
      
  if cond_13 \
    and cond_1 \
    and cond_9 \
    and cond_2 \
    and cond_12:
      condition = True
      
  if display:
    warning.print('MA50_MA5 conditions parameters:')
    c.print('AND CONDITIONS:', color='yellow')
    c.green_red_print(grad_MA5, f'cond_2 (grad_MA5 > 0), {cond_value_2:.3f}')
    c.green_red_print(grad_MA50_1m, f'(grad_MA50_1m > 0), {int(grad_MA50_1m):.3f}')
    c.green_red_print(grad_MA5_1m, f'(grad_MA5_1m > 0), {int(grad_MA5_1m):.3f}')
    c.green_red_print(cond_3, f'cond_3 (deltaMA5_MA50 > 1 and deltaMA5_MA50 < 1.02), {cond_value_3:.3f}')
    c.green_red_print(cond_12, f'cond_12, {cond_12:.3f}')
    c.green_red_print(sum_MA5_MA120_120.iloc[-1] < 0, f'sum_MA5_MA120_1m_120.iloc[-1], {sum_MA5_MA120_120.iloc[-1]:.3f}')
    c.green_red_print(sum_MA5_MA120_120.iloc[-1] >= sum_MA5_MA120_120.iloc[-2], f'sum_MA5_MA120_1m_120.iloc[-2], {sum_MA5_MA120_120.iloc[-2]:.3f}')
    c.green_red_print(sum_MA5_MA120_120.iloc[-2] >= sum_MA5_MA120_120.iloc[-3], f'sum_MA5_MA120_1m_120.iloc[-3], {sum_MA5_MA120_120.iloc[-3]:.3f}')
    
    # c.green_red_print(cond_4, f'cond_4 (square_MA5_MA50_30 < 0), {cond_value_4:.3f}')
    c.print('OR CONDITIONS:', color='yellow')
    c.green_red_print(deltaMA5_MA50_b2 < 0.999, f'deltaMA5_MA50_b2 < 0.999, {deltaMA5_MA50_b2:.3f}')
    c.green_red_print(deltaMA5_MA50_b3 < 0.999, f'deltaMA5_MA50_b3 < 0.999, {deltaMA5_MA50_b3:.3f}')

      
  if grad_MA5 \
    and grad_MA50_1m \
    and grad_MA5_1m \
    and cond_12 \
    and cond_3 \
    and sum_MA5_MA120_120.iloc[-1] < 0 \
    and sum_MA5_MA120_120.iloc[-1] >= sum_MA5_MA120_120.iloc[-2] \
    and sum_MA5_MA120_120.iloc[-2] >= sum_MA5_MA120_120.iloc[-3] \
    and (deltaMA5_MA50_b2 < 0.999 or deltaMA5_MA50_b3 < 0.999):
      condition = True   
     
  if condition: 
    c.green_red_print(condition, condition_type)
    
  conditions_info =  f'''c1: {int(cond_1)}, c2: {int(cond_2)}, c3: {int(cond_3)},\
  c5: {int(cond_5)}, c6: {int(cond_6)}, c7: {int(cond_7)},\
  c8: {int(cond_8)}, c9: {int(cond_9)}, c10: {int(cond_10)},\
  c11: {int(cond_11)}, c12: {int(cond_12)}, c13: {int(cond_13)}'''

  return condition, conditions_info

def stock_buy_condition_MA5_MA120_DS(df, df_1m, df_stats, ticker, display=False):
  '''
  buy condition:

  grad MA120 > 0
  grad MA120_1m > 0
  gram MA5 1m > 0 
  grad MA50 1m >0
  SD grad>0 an SD is negrative and less than some value

  sell condition MA5 1m cromssim MA120 1m
  if profit small trailing = 1%
  if more than 0.3% 
  if MA5 more than MA120 trailing = 1.5%
  ''' 
  condition = False
  condition_type = 'MA5_MA120_DS'

  i_1m = df_1m.shape[0] - 1
  i = df.shape[0] - 1
        
  grad_MA120 = df['MA120'].iloc[-1] >= df['MA120'].iloc[-2] \
          and df['MA120'].iloc[-2] >= df['MA120'].iloc[-3] \
          and df['MA120'].iloc[-3] >= df['MA120'].iloc[-4] \
            
  grad_MA120_1m = df_1m['MA120'].iloc[-1] >= df_1m['MA120'].iloc[-2] \
           and df_1m['MA120'].iloc[-2] >= df_1m['MA120'].iloc[-3] \
           and df_1m['MA120'].iloc[-3] >= df_1m['MA120'].iloc[-4]
  
  grad_MA120_1m_speed = df_1m['MA120'].iloc[-1] / df_1m['MA120'].iloc[-2]
       
  grad_MA5_1m = df_1m['close'].iloc[-1] >= df_1m['close'].iloc[-6] \
            and df_1m['MA5'].iloc[-2] >= df_1m['MA5'].iloc[-3]
            
  grad_MA5 = df_1m['close'].iloc[-1] >= df['close'].iloc[-6] \
            and df['MA5'].iloc[-2] >= df['MA5'].iloc[-3] \
            # and df['MA5'].iloc[-3] >= df['MA5'].iloc[-4] # new from 22/05/2025
            
  grad_MA50 = df_1m['close'].iloc[-1] >= df['close'].iloc[-51] \
              and df['MA50'].iloc[-2] >= df['MA50'].iloc[-3]
              
  grad_MA50_1m = df_1m['close'].iloc[-1] >= df_1m['close'].iloc[-51] \
                  and df_1m['MA50'].iloc[-2] >= df_1m['MA50'].iloc[-3] \
                  and df_1m['MA50'].iloc[-3] >= df_1m['MA50'].iloc[-4] \
  
  MA5_MA120_1m_120 = (df_1m['MA5'] / df_1m['MA120'] - 1).rolling(window=120).sum()
  
  grad_MA5_MA120_1m_120 = MA5_MA120_1m_120.iloc[-1] >= MA5_MA120_1m_120.iloc[-2] \
                      and MA5_MA120_1m_120.iloc[-2] >= MA5_MA120_1m_120.iloc[-3] \
                      and MA5_MA120_1m_120.iloc[-1] > MA5_MA120_1m_120.iloc[-5]
  dist_from_MA50_1m = df_1m['close'].iloc[-1] / df_1m['MA50'].iloc[-1] 
  
  sum_before_crossing = 0
  for j in range(MA5_MA120_1m_120.shape[0] - 1, 1, -1):
    if MA5_MA120_1m_120.iloc[j] < 0:
      sum_before_crossing += MA5_MA120_1m_120.iloc[j]
    if MA5_MA120_1m_120.iloc[j - 1] >= 0 \
      and MA5_MA120_1m_120.iloc[j] <= 0:
      break 
  
  warning.print(f'MA5_MA120_1m_120 is {MA5_MA120_1m_120.iloc[-1]:.4f} for {ticker}')
  
  if display:
    warning.print('MA5_MA120_DS conditions parameters:')
    c.print('AND CONDITIONS:', color='yellow')
    c.green_red_print(grad_MA120, f'grad_MA120')
    c.green_red_print(grad_MA120_1m, f'grad_MA120_1m')
    c.green_red_print(grad_MA5_1m, f'grad_MA5_1m')
    c.green_red_print(grad_MA50_1m, f'grad_MA50_1m')
    c.green_red_print(grad_MA5_MA120_1m_120, f'grad_MA5_MA120_1m_120')
    c.green_red_print(MA5_MA120_1m_120.iloc[-1], f'MA5_MA120_1m_120 is {MA5_MA120_1m_120.iloc[-1]:.4f}')
    c.green_red_print(sum_before_crossing < -50, f'sum_before_crossing is {sum_before_crossing:.4f}')

  if grad_MA120 \
    and grad_MA5 \
    and grad_MA50 \
    and (
         grad_MA120_1m \
         or grad_MA120_1m_speed > 0.99995 
        ) \
    and grad_MA5_1m \
    and grad_MA50_1m \
    and grad_MA5_MA120_1m_120 \
    and sum_before_crossing < -30 \
    and dist_from_MA50_1m < 1.0035 \
    and MA5_MA120_1m_120.iloc[-1] < 0:
      condition = True

  if condition: 
    c.green_red_print(condition, condition_type)

  return condition

def modify_trailing_stop_limit_MA50_MA5_order_old(df, order, current_price, stock_df, stock_df_1m) -> Tuple[pd.DataFrame, pd.DataFrame]:
  try:
    order_id = order['trailing_stop_limit_order_id']
    current_gain = current_price / order['buy_price']
    if order['buy_condition_type'] == 'MA50_MA5' \
      and not(order_id in [None, '', 'FAXXXX'] 
          or isNaN(order_id)):
      trail_spread = order['buy_price'] * trail_spread_coef
      trailing_ratio = order['trailing_ratio']

      
      grad_MA5 = stock_df_1m['close'].iloc[-1] >= stock_df['close'].iloc[-6] \
            and stock_df['MA5'].iloc[-2] >= stock_df['MA5'].iloc[-3]
            
      grad_MA50 = stock_df_1m['close'].iloc[-1] > stock_df['close'].iloc[-51] \
                  and stock_df['MA50'].iloc[-2] >= stock_df['MA50'].iloc[-3]
                  
      grad_MA50_prev = stock_df['MA50'].iloc[-4] <= stock_df['MA50'].iloc[-5] \
                    and stock_df['MA50'].iloc[-5] <= stock_df['MA50'].iloc[-6] \

      deltaMA5_MA50 = stock_df['MA5'].iloc[-1] / stock_df['MA50'].iloc[-1]    
      deltaMA5_MA50_b3 = stock_df['MA5'].iloc[-3] / stock_df['MA50'].iloc[-3]    
      
      deltaMA5_MA120_1m = stock_df_1m['MA5'].iloc[-1] / stock_df_1m['MA120'].iloc[-1]    
      deltaMA5_MA120_1m_b3 = stock_df_1m['MA5'].iloc[-3] / stock_df_1m['MA120'].iloc[-3]   
      
      cond_1 = deltaMA5_MA50_b3 > 1.001  and deltaMA5_MA50 < 1
      cond_2 = not grad_MA5
      cond_3 = stock_df['MA50'].iloc[-1] <= stock_df['MA50'].iloc[-2] \
               and stock_df['MA50'].iloc[-2] <= stock_df['MA50'].iloc[-3]
      
      # MA5 1m crossing MA120 1m ???
      cond_4 = deltaMA5_MA120_1m_b3 > 1 and deltaMA5_MA120_1m < 1
      
      # Double MA5 1hour is negative and red ha
      cond_5 = stock_df['MA5'].iloc[-1] <=  stock_df['MA5'].iloc[-2] \
               and stock_df['MA5'].iloc[-2] <=  stock_df['MA5'].iloc[-3] \
               and stock_df['ha_colour'].iloc[-1] == 'red'
               
      # If Negative MA50 1m
      stock_df_1m_MA50_max = np.maximum(stock_df_1m['MA50'].iloc[-50:-1].max(), stock_df_1m['MA50'].iloc[-1])
      cond_6 =  stock_df_1m['MA50'].iloc[-1] / stock_df_1m_MA50_max < 0.9995 \
                and stock_df_1m['MA5'].iloc[-1] <= stock_df_1m['MA5'].iloc[-2] \
                and stock_df_1m['MA5'].iloc[-2] <= stock_df_1m['MA5'].iloc[-3] 
      
      if (cond_1 and cond_2) \
        or (cond_3 and cond_2) \
        or deltaMA5_MA50 < 0.998 \
        and order['trailing_ratio'] > 0.3:
        trailing_ratio = 0.3  
        
      if (cond_5 or cond_6) \
         and order['trailing_ratio'] > 0.05:
        trailing_ratio = 0.05
        
      if stock_df['ha_colour'].iloc[-1] == 'green' \
         and stock_df['MA5'].iloc[-1] >= stock_df['MA5'].iloc[-2] \
         and order['trailing_ratio'] == 0.05:           
           trailing_ratio = default.trailing_ratio_MA50_MA5
           
      if current_gain <= 0.997 \
        and order['trailing_ratio'] > 0.01:
        trailing_ratio = 0.01
           
      # 0 -- 0.2 -- 0.5 -- 1 
      #   0.99  0.7    0.5
      
      # if current_gain > 1.07 \
      #   and cond_2 \
      #   and order['trailing_ratio'] > 0.3:
      #     trailing_ratio = 0.3
      
      # safe condition 
      # if current_gain < 1.01 and order['trailing_ratio'] > 0.99:
      #   trailing_ratio = 0.99

      # if current_gain >= 1.002 and order['trailing_ratio'] == 0.99:
      #   trailing_ratio = 0.8
      # if current_gain > 1.002 and current_gain < 1.005 \
      #   and order['trailing_ratio'] > 0.7:
      #   trailing_ratio = 0.7
      # if current_gain >= 1.005 and current_gain < 1.0075 \
      #   and order['trailing_ratio'] > 0.55:
      #   trailing_ratio = 0.55
      # if current_gain >= 1.0075 and current_gain < 1.01 \
      #   and order['trailing_ratio'] > 0.35:
      #   trailing_ratio = 0.35
      # if current_gain >= 1.01 and current_gain < 1.015 \
      #   and order['trailing_ratio'] > 0.32:
      #   trailing_ratio = 0.32
      # if current_gain >= 1.015 \
      #   and order['trailing_ratio'] > 0.3:
      #   trailing_ratio = 0.3
  
      if order['trailing_ratio'] != trailing_ratio:
        try:
          info = f'time {datetime.now()}:Ticker {order['ticker']}, cond1: {cond_1}, cond2: {cond_2}, cond3: {cond_3}, cond4: {cond_4} \
          cond5: {cond_5}, cond6: {cond_6}, grad_MA5: {grad_MA5}, grad_MA50: {grad_MA50}, grad_MA50_prev: {grad_MA50_prev}, \
          deltaMA5_MA50: {deltaMA5_MA50:.4f}, deltaMA5_MA50_b3: {deltaMA5_MA50_b3:.4f},  trailing_ratio: {trailing_ratio:.2f}'
          print(info)
          logger.info(info)
        except Exception as e:
          alarm.print(traceback.format_exc())
        order_id = ma.modify_trailing_stop_limit_order(order=order,
                                            trail_value=trailing_ratio,
                                            trail_spread=trail_spread)  
        if order_id != order['trailing_stop_limit_order_id']:
          order['trailing_stop_limit_order_id'] = order_id
        order['trailing_ratio'] = trailing_ratio
        df = ti.update_order(df, order)
  except Exception as e:
    alarm.print(traceback.format_exc())
  return df, order   

def modify_trailing_stop_limit_1230_order(df, order, current_price) -> Tuple[pd.DataFrame, pd.DataFrame]:
  # Modify order with buy_codition_type = 12:30
  try:
    order_id = order['trailing_stop_limit_order_id']
    current_gain = current_price / order['buy_price']
    if order['buy_condition_type'] == '1230' \
      and not(order_id in [None, '', 'FAXXXX'] 
          or isNaN(order_id)):
      trail_spread = order['buy_price'] * trail_spread_coef

      trailing_ratio = order['trailing_ratio']
      if current_gain >= 1.005:
        trailing_ratio = 0.15

      if current_gain <= 1.005 and order['trailing_ratio'] != 0.15:
        trailing_ratio = 0.3

      if ((datetime.now().astimezone(tzinfo_ny).hour == 16 and datetime.now().astimezone(tzinfo_ny).minute > 55) \
        or (datetime.now().astimezone(tzinfo_ny).hour == 17 and datetime.now().astimezone(tzinfo_ny).minute > 55)) \
          and current_gain >= 0.998:
        trailing_ratio = 0.02
  
      if order['trailing_ratio'] != trailing_ratio:
        order_id = ma.modify_trailing_stop_limit_order(order=order,
                                            trail_value=trailing_ratio,
                                            trail_spread=trail_spread)  
        if order_id != order['trailing_stop_limit_order_id']:
          order['trailing_stop_limit_order_id'] = order_id
        order['trailing_ratio'] = trailing_ratio
        df = ti.update_order(df, order)
  except Exception as e:
    alarm.print(traceback.format_exc())
  return df, order   

def modify_trailing_stop_limit_MA5_MA120_DS_order(df, order, current_price, stock_df, stock_df_1m) -> Tuple[pd.DataFrame, pd.DataFrame]:
  try:
    order_id = order['trailing_stop_limit_order_id']
    current_gain = current_price / order['buy_price']
    
    deltaMA5_MA120_1m = stock_df_1m['MA5'].iloc[-1] / stock_df_1m['MA120'].iloc[-1]    
    deltaMA5_MA120_1m_b3 = stock_df_1m['MA5'].iloc[-3] / stock_df_1m['MA120'].iloc[-3]   
    deltaMA5_MA50_1m = stock_df_1m['MA5'].iloc[-1] / stock_df_1m['MA50'].iloc[-1]    
    deltaMA5_MA50_1m_b3 = stock_df_1m['MA5'].iloc[-3] / stock_df_1m['MA50'].iloc[-3]  
    
    if deltaMA5_MA50_1m < 1 and deltaMA5_MA120_1m < 1:
          df, order = check_sell_order_has_been_placed(df, order, ticker, order_type='trailing_stop_limit')
              
    if order['buy_condition_type'] == 'MA5_MA120_DS' \
      and not(order_id in [None, '', 'FAXXXX'] 
          or isNaN(order_id)):
      trail_spread = order['buy_price'] * trail_spread_coef
      trailing_ratio = order['trailing_ratio']

      # MA5 1m crossing MA120 1m 
      cond_1 = deltaMA5_MA120_1m_b3 >= 1 and deltaMA5_MA120_1m < 1
      cond_2 = deltaMA5_MA50_1m_b3 >= 1 and deltaMA5_MA50_1m < 1
        
      if (cond_1 or cond_2):
        if current_gain <= 1.0007:
          if order['trailing_ratio'] > 0.3:
            trailing_ratio = 0.3
        else:
            trailing_ratio = 0.05
    
      if deltaMA5_MA50_1m < 1 and deltaMA5_MA120_1m < 1 \
        and order['trailing_ratio'] > 0.05:
          trailing_ratio = 0.05
          
      if deltaMA5_MA120_1m >= 1.005 \
        or deltaMA5_MA50_1m >= 1.0005:
         trailing_ratio = default.trailing_ratio_MA5_MA120_DS
      
      if order['trailing_ratio'] != trailing_ratio:
        order_id = ma.modify_trailing_stop_limit_order(order=order,
                                            trail_value=trailing_ratio,
                                            trail_spread=trail_spread)  
        if order_id != order['trailing_stop_limit_order_id']:
          order['trailing_stop_limit_order_id'] = order_id
        order['trailing_ratio'] = trailing_ratio
        df = ti.update_order(df, order)
  except Exception as e:
    alarm.print(traceback.format_exc())
  return df, order   