# 2. Check that for all bought stocks placed LIMIT_IF_TOUCHED, STOP orders, trailing_LIT_order if needed
if False:
  if order['limit_if_touched_order_id'] in [None, ''] \
    and not (ticker in limit_if_touched_sell_orders_list):
    price = order['buy_price'] * order['gain_coef']  # buy price should be taken from the trade platform
    order_id = ma.place_limit_if_touched_order(ticker, price, qty)
    if not (order_id is None):
      order['limit_if_touched_order_id'] = order_id
      df = ti.record_order(df, order)
  else:
    sell_order = limit_if_touched_sell_orders.loc[limit_if_touched_sell_orders['order_id'] == order['limit_if_touched_order_id']]
    if sell_order['order_status'].values[0] != ft.OrderStatus.SUBMITTED \
      and sell_order['order_status'].values[0]  != ft.OrderStatus.SUBMITTING \
      and sell_order['order_status'].values[0]  != ft.OrderStatus.WAITING_SUBMIT:
      alarm.print(f'{ticker} limit if touched order has not been sumbitted')
  # Checking stop_order
  if order['stop_order_id'] is None \
    and not (ticker in stop_sell_orders_list):
    price = order['buy_price'] * order['lose_coef']  # buy price should be taken from the trade platform
    order_id = ma.place_stop_order(ticker, price, qty)
    if not (order_id is None):
      order['stop_order_id'] = order_id
      df = ti.record_order(df, order)
  else:
    sell_order = stop_sell_orders.loc[stop_sell_orders['order_id'] == order['stop_order_id']]
    if sell_order['order_status'].values[0] != ft.OrderStatus.SUBMITTED \
        and sell_order['order_status'].values[0] != ft.OrderStatus.SUBMITTING \
        and sell_order['order_status'].values[0]  != ft.OrderStatus.WAITING_SUBMIT:
      alarm.print(f'{ticker} stop order has not been sumbitted')
  # Checing trailing_LIT_order
  if order['gain_coef'] > 1.005:
    if order['trailing_LIT_order_id'] is None \
      and not (ticker in trailing_LIT_orders_list):
      price = order['buy_price'] * order['trailing_LIT_gain_coef']  # buy price should be taken from the trade platform
      order_id = ma.place_limit_if_touched_order(ticker, price, qty, aux_price_coef = 1.0005, remark = 'trailing_LIT')
      if not (order_id is None):
        order['trailing_LIT_order_id'] = order_id
        df = ti.record_order(df, order)
    else:
      sell_order = trailing_LIT_orders.loc[trailing_LIT_orders['order_id'] == order['trailing_LIT_order_id']]
      if sell_order['order_status'].values[0] != ft.OrderStatus.SUBMITTED \
        and sell_order['order_status'].values[0]  != ft.OrderStatus.SUBMITTING \
        and sell_order['order_status'].values[0]  != ft.OrderStatus.WAITING_SUBMIT:
        alarm.print(f'{ticker} trailing limit if touched order has not been sumbitted')

def load_orders_from_csv():
  
  # FUNCTION TO UPDATE times from csv files to df with correct time format
  df = pd.read_csv('db/real_trade_db.csv', index_col='Unnamed: 0')
  df['buy_time'] = pd.to_datetime(df['buy_time'], dayfirst=False)
  df['sell_time'] = pd.to_datetime(df['sell_time'], dayfirst=False)
  # tzinfo = pytz.timezone('Australia/Melbourne')
  # buy_times_list = []
  # sell_times_list = []
  # for index, row in df2.iterrows():
  #   if type(row['buy_time']) == str and'+' in row['buy_time']:
  #     dt = datetime.strptime(row['buy_time'].split('+')[0], '%Y-%m-%d %H:%M:%S.%f')
  #     buy_times_list.append(np.datetime64(dt))
  #   else:
  #     dt = datetime.strptime('1971-01-01 00:00:00.000000', '%Y-%m-%d %H:%M:%S.%f')
  #     buy_times_list.append(np.datetime64(dt))
  #   if type(row['sell_time']) == str and '+' in row['sell_time']:
  #     dt = datetime.strptime(row['sell_time'].split('+')[0], '%Y-%m-%d %H:%M:%S.%f')
  #     sell_times_list.append(np.datetime64(dt))
  #   else:
  #     dt = datetime.strptime('1971-01-01 00:00:00.000000', '%Y-%m-%d %H:%M:%S.%f')
  #     sell_times_list.append(np.datetime64(dt))
  # df2['buy_time'] = buy_times_list
  # df2['sell_time'] = sell_times_list
  # df = pd.concat([df, df2])
  return df


  # order ={
  #     'ticker': 'AVGO',
  #     'buy_price': 100,
  #     'stocks_number': 1
  # }
  # ticker = 'AVGO'
  # if ticker == 'AVGO':
  #   trail_spread = 166 * 0.0003
  #   order_id = ma.place_trailing_stop_limit_order(ticker='AVGO', price=200, qty=1, trail_value=0.15, trail_spread=trail_spread)
  #   order['trailing_stop_limit_order_id'] = order_id
  #   trail_spread = 146 * 0.0003
  #   order_id = ma.modify_trailing_stop_limit_order(order=order,
  #                                                 trail_value=0.111,
  #                                                 trail_spread=trail_spread)
    
  #   ma.cancel_order(order, order_type='trailing_stop_limit')



# Manual updates to DB

# for index, row in df.iterrows():
#   ti.update_order(df, row)

# Manualy change the values
# df.iloc[0] = [0,'DE', datetime.now(), 374.06, 751, 0, None,0,0,0,2,'bought',1.005, 0.95,1.007,0,
# 'FA1956E877FC84A000', 'FA1956E75EBF44A000', 'FA1956E877FC84A000', None] 
# df.loc[] = [0,'DE', datetime.now(), 374.06, 751, 0, None,0,0,0,2,'bought',1.005, 0.95,1.007,0,
# 'FA1956DF73E03B2000', 'FA1956E75EBF44A000', 'FA1956E877FC84A000', 'FA1956EEE2AC3B2000'] 
# df._set_value(0, 'trailing_LIT_order_id' ,'FA1956EEE2AC3B2000')
# df.drop(index=1, inplace=True)


# BUY TEST
# order = ti.buy_order(ticker='CWPE', buy_price=1.8, buy_sum=4)
# order['gain_coef'] = 1.05
# order['lose_coef'] = 0.98
# order = update_buy_order_based_on_platform_data(order)
# df = ti.record_order(order)
# ma.cancel_order(order, type='buy')

# SELL TEST
# historical_orders = ma.get_history_orders()
# historical_order = historical_orders.iloc[0]
# order = df.iloc[0]
# order = ti.sell_order(order, sell_price=order['buy_price']*1.05, historical_order = historical_order)



# Version 1.0
# stock_name_list_opt = [
#   'GOOG', 'JPM', 'XOM', 'UNH', 'AVGO', 'LLY', 'COST',
#   'CRM', 'TMO', 'NFLX', 'TXN', 'INTU', 'NKE', 'QCOM',
#   'BA', 'AMGN', 'MDT', 'PLD', 'MS', 'GS', 'LMT', 'ADI', 'TJX', 'ELV', 'C', 'CVS', 'VRTX', 'SCHW', 'LRCX',
#   'TMUS', 'ETN', 'ZTS', 'CI', 'FI', 'EQIX', 'DUK', 'MU',
#   'AON', 'ITW', 'SNPS', 'KLAC', 'CL', 'WM', 'HCA', 'MMM',
#   'CMG', 'EW', 'GM', 'MCK', 'NSC', 'PH', 'MPC', 'ROP', 
#   'MCHP', 'USB', 'CCI', 'MAR', 'MSI', 'GD', 'JCI', 'PSX', 
#   'SRE', 'ADSK', 'AJG', 'TEL', 'TT', 'PCAR', 'OXY', 'CARR',
#   'IDXX', 'GIS', 'CTAS', 'AIG', 'ANET', 'BIIB', 'SPG', 'MSCI', 'DHI'
# ]

# Version 2.0
# stock_name_list_opt = [
# 'BA', 'ON', 'MCHP', 'ADI', 'PANW', 'DHI', 'ANET', 'AMD', 'LRCX', 'LLY',
# 'MU', 'TXN', 'AIG', 'WMB', 'BSX', 'NKE', 'OXY', 'TT', 'AMAT', 'ETN', 'DE',
# 'EL', 'FDX', 'MAR', 'GE', 'NFLX', 'NUE', 'GOOG', 'ECL', 'AVGO', 'CAT', 'SPG',
# 'ADSK', 'INTU', 'SLB', 'F', 'WMT', 'SBUX', 'SNPS', 'AJG', 'TMUS', 'KLAC', 'CI',
# 'JCI', 'GILD', 'QCOM', 'ROP', 'MO', 'WM', 'HON', 'ITW', 'GS', 'HCA', 'TJX', 'ICE',
# 'DXCM', 'IDXX', 'ABBV', 'CDNS', 'CMCSA', 'JNJ', 'EQIX', 'MDLZ', 'NXPI', 'MSI', 'TEL',
# 'LMT', 'USB', 'MRK', 'HLT', 'APD', 'CTAS', 'MNST', 'NOW', 'AMT', 'PH', 'HUM', 'ADM',
# 'TDG', 'EMR', 'GM', 'ADP', 'CMG', 'SCHW', 'MSCI', 'EOG', 'UNP', 'INTC', 'CME',
# 'MA', 'CVS', 'XOM', 'CSCO', 'WELL', 'TMO', 'MRNA', 'PLD', 'APH', 'PEP', 'CRM', 'MMM',
# 'MMC', 'LIN', 'GIS', 'COST', 'CSX', 'IQV', 'FI', 'MCD', 'VRTX'
# ]

# settings for buy condition Version 1.0
# is_near_global_max_prt = 80
# distance_from_last_top  = 0
# last_top_ratio = 1
# RIV  = 0.25
# buy_ratio_border = 9
# bull_trend_coef = 1.12
# number_tries_to_submit_order = {}
#
# settings for buy condition Version 2.0
# is_near_global_max_prt = 96
# distance_from_last_top  = 0
# last_top_ratio = 1
# RIV  = 0.15
# buy_ratio_border = 0
# bull_trend_coef = 1.12
# number_tries_to_submit_order = {}
#

def version_before():
  pass
  # for ticker in positions_list:
  #   try:
  #     if ticker in bought_stocks_list:
  #       order = bought_stocks.loc[bought_stocks['ticker'] == ticker].sort_values('buy_time').iloc[-1] 
  #       if order['buy_condition_type'] in ['before_market_open_1', 'before_market_open_2', 'before_market_open_3', 'MA50_MA5', 'MA5_MA120_DS']:
  #         stock_df_1m = get_historical_df(ticker = ticker, period='max', interval='1m', prepost=True)
  #         if not stock_df_1m.empty:
  #           current_price = stock_df_1m['close'].iloc[-1]  
  #           current_gain = current_price / order['buy_price']
  #         else:
  #           current_price = 0
  #           current_gain = 0
  #         if order['buy_condition_type'] in ['before_market_open_1', 'before_market_open_2', 'before_market_open_3']:
  #           # Place limit if touched sell order 
  #           df, order = check_sell_order_has_been_placed(df, order, ticker, order_type='limit_if_touched')
  #           # Place stop limit sell order if gain more that 2%:
  #           if current_gain >= 1.01:  # v1. >=1.02 v2.1.005
  #             df, order = check_sell_order_has_been_placed(df, order, ticker, order_type='stop_limit')
  #           # Modify stop limit sell order based on current gain and sell if fast rise 3%
  #           df, order = modify_stop_limit_before_market_open_order(df, order)
  #         if order['buy_condition_type'] == 'MA50_MA5':
  #           # Place limit if touched sell order 1
  #           df, order = check_sell_order_has_been_placed(df, order, ticker, order_type='limit_if_touched')
  #           df, order = check_sell_order_has_been_placed(df, order, ticker, order_type='stop_limit')
  #           if current_gain >= 1.001:  
  #             df, order = check_sell_order_has_been_placed(df, order, ticker, order_type='trailing_stop_limit')
  #           # Modify trailing stop limit sell order based on current gain   
  #           df, order = modify_trailing_stop_limit_MA50_MA5_order(df, order)  
  #   except Exception as e:
  #     alarm.print(traceback.format_exc())

  # def prev_version():
  #         for ticker2 in positions_list:
  #           try:
  #             if ticker2 in bought_stocks_list:
  #               stock_df_1m = get_historical_df(ticker = ticker2, period='max', interval='1m', prepost=True)
  #               current_price = stock_df_1m['close'].iloc[-1]  
  #               order = bought_stocks.loc[bought_stocks['ticker'] == ticker2].sort_values('buy_time').iloc[-1]  

  #               if not order['buy_condition_type'] in ['1230', 'before_market_open_1', 'before_market_open_2', 'before_market_open_3']:
  #               # Checking trailing_stop_limit_order
  #                 df, order = check_sell_order_has_been_placed(df, order, ticker2, order_type='trailing_stop_limit')
  #                 # Modification of trailing stop limit order based on current gain
  #                 df, order= trailing_stop_limit_order_trailing_ratio_modification(df, order, current_price)

  #               if order['buy_condition_type'] == '1230':
  #                 df, order = place_traililing_stop_limit_order_at_the_end_of_trading_day(df, order, ticker2, current_price) 
  #                 df, order = modify_trailing_stop_limit_1230_order(df, order, current_price)
                
  #               if order['buy_condition_type'] in ['before_market_open_1', 'before_market_open_2', 'before_market_open_3']:
  #                 # Place limit if touched sell order 
  #                 df, order = check_sell_order_has_been_placed(df, order, ticker, order_type='limit_if_touched')
  #                 # Place stop limit sell order if gain more that 2%:
  #                 if current_gain >= 1.02:
  #                   df, order = check_sell_order_has_been_placed(df, order, ticker, order_type='stop_limit')
  #                 if order['buy_condition_type'] == 'before_market_open_3' \
  #                   and current_gain > 1.003:
  #                   df, order = check_sell_order_has_been_placed(df, order, ticker, order_type='stop_limit')
  #                 # Modify stop limit sell order based on current gain and sell if fast rise 3%\
  #                 df, order = modify_stop_limit_before_market_open_order(df, order, current_price) 
                
  #               if order['buy_condition_type'] == 'MA50_MA5':
  #                 # Place limit if touched sell order 1
  #                 df, order = check_sell_order_has_been_placed(df, order, ticker2, order_type='limit_if_touched')
  #                 df, order = check_sell_order_has_been_placed(df, order, ticker2, order_type='stop_limit')
  #                 if current_gain >= 1.001:  
  #                   df, order = check_sell_order_has_been_placed(df, order, ticker2, order_type='trailing_stop_limit')
  #                 # Modify trailing stop limit sell order based on current gain   
  #                 df, order = modify_trailing_stop_limit_MA50_MA5_order(df, order, current_price)

  #               # Recalculate Trailing LIT gain coefficient:
  #               df, order = recalculate_trailing_LIT_gain_coef(df, order, current_price)
  #             else:
  #                   alarm.print(f'{ticker2} is in positional list but not in DB!')      
  #           except Exception as e:
  #             alarm.print(traceback.format_exc())   
  
    # buy when MACD_hist is positive and increasing and not more 8 greens MACD_hist in a row:
  MACD_hist_max_amp = df['MACD_hist'].iloc[-50:-2].abs().max()
  cond_a1  =  df['MACD_hist'].iloc[-1] > 0 \
            and df['MACD_hist'].iloc[-1] < MACD_hist_max_amp\
            and df['MACD_hist'].iloc[-1] > df['MACD_hist'].iloc[-2] \
            and df['MACD_hist'].iloc[-2] >= df['MACD_hist'].iloc[-3] \
            and not (df['MACD_hist'].iloc[-8:] > 0).all()

  # buy when MACD_hist is negative and increasing:
  cond_a2 = df['MACD_hist'].iloc[-1] < 0 \
           and df['MACD_hist'].iloc[-1] > df['MACD_hist'].iloc[-2] \
           and df['MACD_hist'].iloc[-2] > df['MACD_hist'].iloc[-3] \
          #  and df['MACD_hist'].iloc[-3] > df['MACD_hist'].iloc[-4]   # comment from 12/07/2025        
  
  # buy when MACD_hist is negative and heiken ashi is green after red:
  cond_value_a3 = number_red_candles(df, i, k=10)
  cond_a3 = df['MACD_hist'].iloc[-1] < 0 \
            and df['MACD_hist'].iloc[-1] > df['MACD_hist'].iloc[-2] \
            and df['ha_colour'].iloc[-1] == 'green' \
            and cond_value_a3 > 5
            
  # buy when MACD_hist is positive and increasing after local minimum:
  local_min_index = np.argmin(df['MACD_hist'].iloc[-9:]) # 0 is the last candle, 8 is the oldest
  local_min = df['MACD_hist'].iloc[-9:].iloc[local_min_index]
  value_before_local_min = df['MACD_hist'].iloc[-9:].iloc[local_min_index - 1] if local_min_index > 0 else local_min
  cond_a4 = df['MACD_hist'].iloc[-1] > 0 \
            and df['MACD_hist'].iloc[-1] > df['MACD_hist'].iloc[-2] \
            and df['MACD_hist'].iloc[-1] > local_min \
            and value_before_local_min > local_min \
            and (df['MACD_hist'].iloc[-9:] > 0).all() \
            and local_min_index < 6 # local minimum should be not more than 6 candles ago
            
    # Sum MACD 1 min last 20 cangles should be negative
  MACD_1m_sum20 = df_1m['MACD'].iloc[-20:].sum()
  cond_MACD_1m_sum20 = MACD_1m_sum20 < 0
  

  MACD_hist_1m = df_1m['MACD_hist'].iloc[-1]
  cond_MACD_hist_1m = MACD_hist_1m < 0 \
    or ( not (df_1m['MACD_hist'].iloc[-7:] > 0).all() \
        and df_1m['MACD_hist'].iloc[-1] >= df_1m['MACD_hist'].iloc[-2] \
        and df_1m['MACD_hist'].iloc[-2] >= df_1m['MACD_hist'].iloc[-3])
  
  # cond_9: MACD 1 min is positive and increasing, MACD_hist is negative and increasing
  cond_positive_MACD_1m = df_1m['MACD'].iloc[-1] > 0.01 \
           and df_1m['MACD'].iloc[-1] >= df_1m['MACD'].iloc[-2] \
           and df_1m['MACD_hist'].iloc[-1] < 0 \
           and df_1m['MACD_hist'].iloc[-1] > df_1m['MACD_hist'].iloc[-2]

  cond2_positive_MACD_1m  = df_1m['MACD'].iloc[-1] > 0.01 \
            and df_1m['MACD_hist'].iloc[-1] > 0 \
            and df_1m['MACD_hist'].iloc[-1] > df_1m['MACD_hist'].iloc[-2] \
            and df_1m['MACD_hist'].iloc[-2] >= df_1m['MACD_hist'].iloc[-3] \
            and not (df_1m['MACD_hist'].iloc[-30:] > 0).all()     
            
  if False:
    grad_MA5 = df_1m['close'].iloc[-1] >= df['close'].iloc[-6] \
              and df['MA5'].iloc[-2] >= df['MA5'].iloc[-3] \
              # and df['MA5'].iloc[-3] >= df['MA5'].iloc[-4] # new from 22/05/2025
    grad_MA5_1m = df_1m['close'].iloc[-1] >= df_1m['close'].iloc[-6] \
              and df_1m['MA5'].iloc[-2] >= df_1m['MA5'].iloc[-3]   
    grad_MA50 = df_1m['close'].iloc[-1] >= df['close'].iloc[-51] \
                and df['MA50'].iloc[-2] >= df['MA50'].iloc[-3]             
    grad_MA50_1m = df_1m['close'].iloc[-1] >= df_1m['close'].iloc[-51] \
                    and df_1m['MA50'].iloc[-2] >= df_1m['MA50'].iloc[-3] \
                    and df_1m['MA50'].iloc[-3] >= df_1m['MA50'].iloc[-4]         
    grad_MA50_prev = df['MA50'].iloc[-4] <= df['MA50'].iloc[-5] \
                  and df['MA50'].iloc[-5] <= df['MA50'].iloc[-6] \

    deltaMA5_MA50 = df['MA5'].iloc[-1] / df['MA50'].iloc[-1]    
    deltaMA5_MA50_b2 = df['MA5'].iloc[-2] / df['MA50'].iloc[-2]    
    deltaMA5_MA50_b3 = df['MA5'].iloc[-3] / df['MA50'].iloc[-3]    
    deltaMA5_MA50_1min = df_1m['MA5'].iloc[-1] / df_1m['MA50'].iloc[-1]
    deltaMA5_MA50_1min_b3 = df_1m['MA5'].iloc[-3] / df_1m['MA50'].iloc[-3]
    square_MA5_MA50_30 = (df['MA5'].iloc[-31:] / df['MA50'].iloc[-31:]).sum() - 31
    
  MACD_hist_max_amp_1m = df_1m['MACD_hist'].iloc[-50:-2].abs().max()
  cond_MACD_hist_max_amp_1m =  df_1m['MACD_hist'].iloc[-1] < 0 \
    or df_1m['MACD_hist'].iloc[-1] < MACD_hist_max_amp_1m * 0.5
  
  # cond 8 new from 02/07/2025 !!!
  if (cond_a1 or cond_a2 or cond_a3 or cond_a4) \
      and cond_RSI \
      and cond_grad_MACD \
      and cond_MACD_hist_speed \
      and cond_delta_MA50_M120 \
      and cond_grad_MACD_1m \
      and cond_MA50_M120_1m_120 \
      and cond_grad_MA5_1m \
      and cond_RSI_1m \
      and cond_MACD_hist_max_amp_1m:
      # condition = True
      tt = 1
      
  time_cond = datetime.now().astimezone(tzinfo_ny).hour == 9  \
              and datetime.now().astimezone(tzinfo_ny).minute > 28 \
              and datetime.now().astimezone(tzinfo_ny).minute < 45       
              

  delta_MA50_MA120_1m_sum180 = delta_MA50_MA120_1m.iloc[-180:].sum()
  cond_sum_delta_MA50_MA120_1m = delta_MA50_MA120_1m_sum180 < 0
  
    if display and False:
    warning.print('MA50_MA5 conditions parameters:')
    c.print('AND CONDITIONS:', color='yellow')
    c.green_red_print(cond_RSI, f'cond_RSI (MA30_RSI10), {cond_value_RSI:.3f}')
    c.green_red_print(cond_grad_MACD, f'cond_grad_MACD, {MACD:.3f}')
    c.green_red_print(cond_delta_MA50_M120, f'cond_delta_MA50_M120, {delta_MA50_M120.iloc[-1]:.3f}')
    c.green_red_print(cond_MACD_hist_speed, f'(MACD_hist speed more than 1.05), {MACD_hist_speed:.3f}, MACD_hist: {MACD_hist:.3f}')
    c.green_red_print(cond_grad_MACD_1m, f'grad_MACD_1m > 0, {MACD_1m:.3f}')
    c.green_red_print(cond_MACD_hist_1m, f'(MACD_hist 1m < 0), {MACD_hist_1m:.3f}')
    c.green_red_print(cond_RSI_1m, f'cond_RSI_1m, {df_1m["MA30_RSI10"].iloc[-1]:.3f}')
    c.green_red_print(cond_MA50_M120_1m_120, f'cond_MA50_M120_1m_120, {delta_MA50_MA120_1m.iloc[-1]:.3f}')
    c.green_red_print(cond_grad_MA5_1m, f'cond_grad_MA5_1m')
    c.green_red_print(cond_MACD_hist_max_amp_1m, f'cond_MACD_hist_max_amp_1m, {MACD_hist_max_amp_1m:.3f}')
    c.print('OR CONDITIONS:', color='yellow')
    c.green_red_print(cond_MACD_1m_sum20, f'MACD 1m sum last 20 < 0, {MACD_1m_sum20:.3f}')
    c.green_red_print(cond_positive_MACD_1m, f'cond_positive_MACD_1m')
    c.green_red_print(cond2_positive_MACD_1m, f'cond2_positive_MACD_1m')
    c.print('OR CONDITIONS:', color='yellow')
    c.green_red_print(cond_a1, f'cond_a1 (MACD_hist > 0), {MACD_hist:.3f}')
    c.green_red_print(cond_a2, f'cond_a2 (MACD_hist < 0), {MACD_hist:.3f}')
    c.green_red_print(cond_a3, f'cond_a3 (number red 1h ha candles > 5), {cond_value_a3:.3f}')
    c.green_red_print(cond_a4, f'cond_a4 (MACD_hist > 0), {MACD_hist:.3f}')


# settings for buy condition Version 3.0
# is_near_global_max_prt = 120
last_top_ratio = 1
RIV  = 0.15
buy_ratio_border = 0
bull_trend_coef = 1.07
number_tries_to_submit_order = {}

# orders settings
lose_coef_1m = 0.995
lose_coef_1h = 0.995



and cond_grad_MA50_1m \ +++
and cond_grad_MACD_1m \ +++
and cond_grad_MA5_1m \
and cond_grad_MA20_1m \
and cond_RSI_1m \
and cond_VR_1m \  # +++
and cond_MA5_delta_MA5_MA20_1m:
