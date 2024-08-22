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