if False:
  def stock_buy_conditions(df, ticker):
    '''
      Parameters:

      Returns:
        condition
    '''
    condition = False
    buy_price = 0

    last_top = df['close'].iloc[0]
    last_top_time = df['close'].index[0]
    last_top_i = 0
      
    range_ = range(df.shape[0] - 200, df.shape[0])

    for i in range_:
    # last top and reverse flag
      if df['ha_colour'].iloc[i - 1] == 'red' \
        and df['ha_colour'].iloc[i - 2] == 'green'\
        and df['ha_colour'].iloc[i - 3] == 'green'\
        and df['ha_pct'].iloc[i - 2] > 0.1 \
        and df['ha_pct'].iloc[i - 3] > 0.1:
    
        last_top = df['high'].iloc[i - 1]
        last_top_i = i - 1

    i = df.shape[0] - 1

    # changed from i to i-1 as i is dynamic price!!!
    buy_ratio = float(last_top -  df['open'].iloc[i]) / float(df['ha_pct'].iloc[i])
    if df['ha_pct'].iloc[i] > RIV \
      and last_top / df['open'].iloc[i] > last_top_ratio \
      and i - last_top_i > distance_from_last_top \
      and buy_ratio > buy_ratio_border \
      and not(is_near_global_max_prt(df, i, k=400, prt=is_near_global_max_prt)) \
      and number_red_candles(df, i) > 6:
      
      buy_price = float(df['close'].iloc[i])
      condition = True
    
    return condition, buy_price
