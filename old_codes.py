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