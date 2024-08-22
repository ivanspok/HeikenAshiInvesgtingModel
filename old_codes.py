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