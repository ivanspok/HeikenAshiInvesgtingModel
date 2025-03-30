import time
import moomoo as ft
from forex_python.converter import CurrencyRates
from currency_converter import CurrencyConverter
import pandas as pd
from colog.colog import colog
from datetime import datetime, timedelta
c = colog()
warning = colog(TextColor='orange')
alarm = colog(TextColor='red')

# Moomoo settings
ip = '127.0.0.1'
port = 11111
unlock_pwd = '771991'
MARKET = 'US.'
trd_env = ft.TrdEnv.SIMULATE

order = {}
ticker = 'test'

# Global
number_attemps_to_get_data = 0

class Moomoo_API():
    
    def __init__(self, ip, port, trd_env, acc_id):
        self.ip = ip
        self.port = port
        self.trd_env = trd_env
        self.acc_id = acc_id
    
    def stock_is_bought(self, ticker):
        try:
            trd_ctx = ft.OpenSecTradeContext(filter_trdmarket=ft.TrdMarket.US, host=self.ip, port=self.port, security_firm=ft.SecurityFirm.FUTUAU)
            ret, data = trd_ctx.position_list_query()
            if ret == ft.RET_OK:
                print(data)
                if data.shape[0] > 0:  # If the position list is not empty
                    bought_stocks = data['code'].to_list()
                    print(data['stock_name'][0])  # Get the first stock name of the holding position
                    print(data['stock_name'].values.tolist())  # Convert to list
            else:
                alarm.print('position_list_query error: ', data)
            trd_ctx.close()  # Close the current connection
        except Exception as e:
            alarm.print(e)
        return ticker in bought_stocks

    def place_market_buy_order(self, ticker, price, qty):
        order_id = None
        try:
            trd_ctx  = ft.OpenSecTradeContext(filter_trdmarket=ft.TrdMarket.US, host=self.ip, port=self.port, security_firm=ft.SecurityFirm.FUTUAU)
            self.unlock_trade()
            stock_code = MARKET + ticker
            ret, data = trd_ctx.place_order(
                                      price=0.1,
                                      qty=qty,
                                      code=stock_code,
                                      trd_side=ft.TrdSide.BUY,
                                      trd_env=self.trd_env,
                                      order_type=ft.OrderType.MARKET
                                      )
            if ret == ft.RET_OK:
                order_id = data['order_id'].values[0]
                print(f'Placing market buy order for {stock_code}')
            else:
                alarm.print(data)
            trd_ctx.close()
        except Exception as e:
            alarm.print(e)
            trd_ctx.close()
        return data, order_id
    
    def place_buy_limit_order(self, ticker, price, qty, fill_outside_rth=False):
        order_id = None
        order = None
        try:
            trd_ctx  = ft.OpenSecTradeContext(filter_trdmarket=ft.TrdMarket.US, host=self.ip, port=self.port, security_firm=ft.SecurityFirm.FUTUAU)
            self.unlock_trade()
            stock_code = MARKET + ticker
            ret, data = trd_ctx.place_order(
                                    price=price,
                                    qty=qty,
                                    code=stock_code,
                                    trd_side=ft.TrdSide.BUY,
                                    trd_env=self.trd_env,
                                    order_type=ft.OrderType.NORMAL,
                                    adjust_limit=0.01,
                                    fill_outside_rth=fill_outside_rth)
            print(f'Placing BUY limit order for {stock_code}')
            print(f'Market response is {ret}, data is {data}')
            if ret == ft.RET_OK:
                order_id = data['order_id'].values[0]
                order = data
            else:
                alarm.print(data)
            trd_ctx.close()
        except Exception as e:
            alarm.print(e)
            trd_ctx.close()
        return order, order_id

    def place_stop_limit_buy_order(self, ticker, price, qty):
        order_id = None
        order = None
        try:
            trd_ctx  = ft.OpenSecTradeContext(filter_trdmarket=ft.TrdMarket.US, host=self.ip, port=self.port, security_firm=ft.SecurityFirm.FUTUAU)
            self.unlock_trade()
            stock_code = MARKET + ticker
            ret, data = trd_ctx.place_order(
                                    price=price,
                                    qty=qty,
                                    code=stock_code,
                                    trd_side=ft.TrdSide.BUY,
                                    trd_env=self.trd_env,
                                    order_type=ft.OrderType.STOP_LIMIT,
                                    adjust_limit=0.01,
                                    aux_price=price * 1.0002,
                                    fill_outside_rth=True)
            print(f'Placing BUY stop limit order for {stock_code}')
            print(f'Market response is {ret}, data is {data}')
            if ret == ft.RET_OK:
                order_id = data['order_id'].values[0]
                order = data
            else:
                alarm.print(data)
            trd_ctx.close()
        except Exception as e:
            alarm.print(e)
            trd_ctx.close()
        return order, order_id

    def place_stop_limit_sell_order(self, ticker, price, qty):
        order_id = None
        order = None
        try:
            trd_ctx  = ft.OpenSecTradeContext(filter_trdmarket=ft.TrdMarket.US, host=self.ip, port=self.port, security_firm=ft.SecurityFirm.FUTUAU)
            self.unlock_trade()
            stock_code = MARKET + ticker
            ret, data = trd_ctx.place_order(
                                    price=price * 0.999,
                                    qty=qty,
                                    code=stock_code,
                                    trd_side=ft.TrdSide.SELL,
                                    trd_env=self.trd_env,
                                    order_type=ft.OrderType.STOP_LIMIT,
                                    time_in_force=ft.TimeInForce.GTC,
                                    aux_price=price * 0.9997,
                                    adjust_limit=0.01,
                                    fill_outside_rth=True)
            print(f'Placing stop limit sell order for {stock_code}')
            print(f'Market response is {ret}, data is {data}')
            if ret == ft.RET_OK:
                order_id = data['order_id'].values[0]
                order = data
            else:
                alarm.print(data)
            trd_ctx.close()
        except Exception as e:
            alarm.print(e)
            trd_ctx.close()
        return order_id

    def place_buy_limit_if_touched_order(self, ticker, price, qty, fill_outside_rth=True):
        order_id = None
        order = None
        try:
            trd_ctx  = ft.OpenSecTradeContext(filter_trdmarket=ft.TrdMarket.US, host=self.ip, port=self.port, security_firm=ft.SecurityFirm.FUTUAU)
            self.unlock_trade()
            stock_code = MARKET + ticker
            ret, data = trd_ctx.place_order(
                                    price=price * 1.00015,
                                    qty=qty,
                                    code=stock_code,
                                    trd_side=ft.TrdSide.BUY,
                                    trd_env=self.trd_env,
                                    time_in_force=ft.TimeInForce.GTC,
                                    adjust_limit=0.01,
                                    order_type=ft.OrderType.LIMIT_IF_TOUCHED,
                                    aux_price=price,
                                    fill_outside_rth=fill_outside_rth)
            print(f'Placing BUY limit if touched order for {stock_code}')
            print(f'Market response is {ret}, data is {data}')
            if ret == ft.RET_OK:
                order_id = data['order_id'].values[0]
                order = data
            else:
                alarm.print(data)
                order = data
            trd_ctx.close()
        except Exception as e:
            alarm.print(e)
            trd_ctx.close()
        return order, order_id

    def place_limit_if_touched_order(self, ticker, price, qty, aux_price_coef = 1.0001, remark=''):
        order_id = None
        try:
            trd_ctx  = ft.OpenSecTradeContext(filter_trdmarket=ft.TrdMarket.US, host=self.ip, port=self.port, security_firm=ft.SecurityFirm.FUTUAU)
            self.unlock_trade()
            stock_code = MARKET + ticker
            ret, data = trd_ctx.place_order(
                                    price=price,
                                    qty=qty,
                                    code=stock_code,
                                    trd_side=ft.TrdSide.SELL,
                                    trd_env=self.trd_env,
                                    time_in_force=ft.TimeInForce.GTC,
                                    adjust_limit=0.01,
                                    order_type=ft.OrderType.LIMIT_IF_TOUCHED,
                                    aux_price=price * aux_price_coef,
                                    remark=remark,
                                    fill_outside_rth=True)
            print(f'Placing limit if touched order for {stock_code}')
            print(f'Market response is {ret}, data is {data}')
            if ret == ft.RET_OK:
                order_id = data['order_id'].values[0]
            else:
                alarm.print(data)
            trd_ctx.close()
        except Exception as e:
            alarm.print(e)
            trd_ctx.close()
        return order_id
    
    def place_trailing_stop_limit_order(self, ticker, price, qty, trail_value, trail_spread, remark=''):
        order_id = None
        try:
            trd_ctx  = ft.OpenSecTradeContext(filter_trdmarket=ft.TrdMarket.US, host=self.ip, port=self.port, security_firm=ft.SecurityFirm.FUTUAU)
            self.unlock_trade()
            stock_code = MARKET + ticker
            ret, data = trd_ctx.place_order(
                                    price=price,
                                    qty=qty,
                                    code=stock_code,
                                    trd_side=ft.TrdSide.SELL,
                                    trd_env=self.trd_env,
                                    time_in_force=ft.TimeInForce.GTC,
                                    adjust_limit=0.01,
                                    order_type=ft.OrderType.TRAILING_STOP_LIMIT,
                                    trail_type= ft.TrailType.RATIO,
                                    trail_value=trail_value,
                                    trail_spread=trail_spread,
                                    remark=remark,
                                    fill_outside_rth=True)
            print(f'Placing trailing stop limit order for {stock_code}')
            print(f'Market response is {ret}, data is {data}')
            if ret == ft.RET_OK:
                order_id = data['order_id'].values[0]
            else:
                alarm.print(data)
            trd_ctx.close()
        except Exception as e:
            alarm.print(e)
            trd_ctx.close()
        return order_id
    
    def place_stop_order(self, ticker, price, qty):
        '''
        Note: for Market order price can be passed any value
        '''
        order_id = None
        try:
            trd_ctx  = ft.OpenSecTradeContext(filter_trdmarket=ft.TrdMarket.US, host=self.ip, port=self.port, security_firm=ft.SecurityFirm.FUTUAU)
            self.unlock_trade()
            stock_code = MARKET + ticker
            ret, data = trd_ctx.place_order(
                                    price=price,
                                    qty=qty,
                                    code=stock_code,
                                    trd_side=ft.TrdSide.SELL,
                                    trd_env=self.trd_env,
                                    time_in_force=ft.TimeInForce.GTC,
                                    order_type=ft.OrderType.STOP,
                                    aux_price=price * 0.9998)
            print(f'Placing stop order for {stock_code}')
            print(f'Market response is {ret}, data is {data}')
            if ret == ft.RET_OK:
                order_id = data['order_id'].values[0]
            else:
                alarm.print(data)
            trd_ctx.close()
        except Exception as e:
            alarm.print(e)
            trd_ctx.close()
        return order_id
    
    def cancel_order(self, order, order_type):
        '''
        Cancel order uy type:
         - order_type: buy | limit_if_touch | stop | trailing_LIT | trailing_stop_limit
        '''
        status = False        
        ticker = order['ticker']
        order_id_type = order_type + '_order_id'
        order_id = order[order_id_type]
        try:
            trd_ctx  = ft.OpenSecTradeContext(filter_trdmarket=ft.TrdMarket.US, host=self.ip, port=self.port, security_firm=ft.SecurityFirm.FUTUAU)
            self.unlock_trade()
            ret, data = trd_ctx.modify_order(
                                        modify_order_op=ft.ModifyOrderOp.CANCEL,
                                        order_id=order_id,
                                        price=0,
                                        qty=0
                                        )
            warning.print(f'Canceling {type} order for {ticker}, {order_id}')
            print(f'Market response is {ret}, data is {data}')
            if ret == ft.RET_OK:
                status = True
            else:
                alarm.print(data)
            trd_ctx.close()
        except Exception as e:
            alarm.print(e)
            trd_ctx.close()
        return status
    
    def modify_trailing_stop_limit_order(self, order, trail_value, trail_spread):
        '''
        Note: for Market order price can be passed any value
        '''
        order_id = order['trailing_stop_limit_order_id']
        price =  order['buy_price'] * 1.1
        aux_price =  order['buy_price'] * 0.9
        qty = order['stocks_number']
        ticker = order['ticker']
        try:
            trd_ctx  = ft.OpenSecTradeContext(filter_trdmarket=ft.TrdMarket.US, host=self.ip, port=self.port, security_firm=ft.SecurityFirm.FUTUAU)
            self.unlock_trade()
            ret, data = trd_ctx.modify_order(
                                    modify_order_op=ft.ModifyOrderOp.NORMAL,
                                    order_id=order_id,
                                    price=price,
                                    qty=qty,
                                    trd_env=self.trd_env,
                                    adjust_limit=0.01,
                                    aux_price=aux_price,
                                    trail_type= ft.TrailType.RATIO,
                                    trail_value=trail_value,
                                    trail_spread=trail_spread
                                    )
            print(f'Modifying trailing stop limit order for {ticker}, {order_id}')
            print(f'Market response is {ret}, data is {data}')
            if ret == ft.RET_OK:
                order_id_returned = data['order_id'].values[0]
                if order_id_returned != order_id:
                    print(f'order_id is {order_id}, order_id_returned is {order_id_returned}')
                    order_id = order_id_returned
            else:
                alarm.print(data)
            trd_ctx.close()
        except Exception as e:
            alarm.print(e)
            trd_ctx.close()
        return order_id

    def modify_limit_if_touched_order(self, order, gain_coef, aux_price_coef = 1.0001, order_type='limit_if_touched'):
        '''
        Modify for sell orders: limit_if_touched and trailing_LIT
        '''
        order_id = None
        if order_type == 'limit_if_touched':
            order_id = order['limit_if_touched_order_id']
        if order_type == 'trailing_LIT':
            order_id = order['trailing_LIT_order_id']

        price =  order['buy_price'] * gain_coef
        qty = order['stocks_number']
        ticker = order['ticker']
        try:
            trd_ctx  = ft.OpenSecTradeContext(filter_trdmarket=ft.TrdMarket.US, host=self.ip, port=self.port, security_firm=ft.SecurityFirm.FUTUAU)
            self.unlock_trade()
            ret, data = trd_ctx.modify_order(
                                    modify_order_op=ft.ModifyOrderOp.NORMAL,
                                    order_id=order_id,
                                    price=price,
                                    aux_price=price * aux_price_coef,
                                    qty=qty,
                                    trd_env=self.trd_env,
                                    adjust_limit=0.01,
                                    )
            print(f'Modifying limit if touched order for {ticker}, {order_id}')
            print(f'Market response is {ret}, data is {data}')
            if ret == ft.RET_OK:
                order_id_returned = data['order_id'].values[0]
                if order_id_returned != order_id:
                    print(f'order_id is {order_id}, order_id_returned is {order_id_returned}')
                    order_id = order_id_returned
            else:
                alarm.print(data)
            trd_ctx.close()
        except Exception as e:
            alarm.print(e)
            trd_ctx.close()
        return order_id

    def modify_stop_order(self, order, lose_coef):
        '''
        Note: for Market order price can be passed any value
        '''
        order_id = order['stop_order_id']
        price =  order['buy_price'] * lose_coef
        qty = order['stocks_number']
        ticker = order['ticker']
        try:
            trd_ctx  = ft.OpenSecTradeContext(filter_trdmarket=ft.TrdMarket.US, host=self.ip, port=self.port, security_firm=ft.SecurityFirm.FUTUAU)
            self.unlock_trade()
            ret, data = trd_ctx.modify_order(
                                    modify_order_op=ft.ModifyOrderOp.NORMAL,
                                    order_id=order_id,
                                    price=price,
                                    aux_price=price * 0.9998,
                                    qty=qty,
                                    trd_env=self.trd_env,
                                    adjust_limit=0.01,
                                    )
            print(f'Modifying stop order for {ticker}, {order_id}')
            print(f'Market response is {ret}, data is {data}')
            if ret == ft.RET_OK:
                order_id_returned = data['order_id'].values[0]
                if order_id_returned != order_id:
                    print(f'order_id is {order_id}, order_id_returned is {order_id_returned}')
                    order_id = order_id_returned
            else:
                alarm.print(data)
            trd_ctx.close()
        except Exception as e:
            alarm.print(e)
            trd_ctx.close()
        return order_id

    def modify_stop_limit_order(self, order, lose_coef):
        '''
        Note: for Market order price can be passed any value
        '''
        order_id = order['stop_limit_order_id']
        price =  order['buy_price'] * lose_coef
        qty = order['stocks_number']
        ticker = order['ticker']
        try:
            trd_ctx  = ft.OpenSecTradeContext(filter_trdmarket=ft.TrdMarket.US, host=self.ip, port=self.port, security_firm=ft.SecurityFirm.FUTUAU)
            self.unlock_trade()
            ret, data = trd_ctx.modify_order(
                                    modify_order_op=ft.ModifyOrderOp.NORMAL,
                                    order_id=order_id,
                                    price=price,
                                    aux_price=price * 0.9998,
                                    qty=qty,
                                    trd_env=self.trd_env,
                                    adjust_limit=0.01,
                                    )
            print(f'Modifying stop limit order for {ticker}, {order_id}')
            print(f'Market response is {ret}, data is {data}')
            if ret == ft.RET_OK:
                order_id_returned = data['order_id'].values[0]
                if order_id_returned != order_id:
                    print(f'order_id is {order_id}, order_id_returned is {order_id_returned}')
                    order_id = order_id_returned
            else:
                alarm.print(data)
            trd_ctx.close()
        except Exception as e:
            alarm.print(e)
            trd_ctx.close()
        return order_id    

    def get_history_orders(self):
        data = None
        global number_attemps_to_get_data
        try:
            trd_ctx  = ft.OpenSecTradeContext(filter_trdmarket=ft.TrdMarket.US, host=ip, port=port, security_firm=ft.SecurityFirm.FUTUAU)
            match number_attemps_to_get_data:
                case 0:
                    hours = 24 * 30
                case 1:
                    match datetime.now().weekday():
                        case 6: k = 1
                        case 7: k = 2
                        case 0: k = 3
                        case _: k = 0
                    hours = 24 * (k + 1)
                case 2:
                    hours = 1
                case _:
                    hours = 1
            start = str(datetime.now() - timedelta(hours=hours)).split('.')[0]
            end  = str(datetime.now()).split('.')[0]
            number_attemps_to_get_data += 1
            ret, data = trd_ctx.history_order_list_query(acc_id=self.acc_id, trd_env=self.trd_env, start=start, end=end)
            if ret == ft.RET_OK:
                if data.shape[0] > 0:  # If the order list is not empty
                    print('history orders received successfully')
            else:
                alarm.print('history_order_list_query error: ', data)
            trd_ctx.close()
        except Exception as e:
            alarm.print(e)
            trd_ctx.close()
        if type(data) == str:
            if data  == 'Get Historical Order List request timed out' \
                or data == 'PacketErr.Timeout':
                if number_attemps_to_get_data <=2:
                    alarm.print(f'Number attemps to get data is {number_attemps_to_get_data} ')
                    alarm.print('''1 second sleep timeout due tue 'Get Historical Order List request timed out' ''')
                    time.sleep(1)
                    data = self.get_history_orders()
                else:
                    alarm.print('''30 second sleep timeout due tue 'Get Historical Order List request timed out' ''')
                    time.sleep(30)
        number_attemps_to_get_data = 0
        return data    
    
   
    def get_list_of_trading_accounts(self):
        try:
            trd_ctx  = ft.OpenSecTradeContext(filter_trdmarket=ft.TrdMarket.US, host=ip, port=port, security_firm=ft.SecurityFirm.FUTUAU)
            ret, data = trd_ctx.get_acc_list()
            if ret == ft.RET_OK:
                print(data)
                print(data['acc_id'][0])  # Get the first account ID
                print(data['acc_id'].values.tolist())  # convert to list
            else:
                alarm.print('get_acc_list error: ', data)
            trd_ctx.close()
        except Exception as e:
            alarm.print(e)
            trd_ctx.close()
        return data
    
    def unlock_trade(self, is_unlock=True):
        trd_ctx  = ft.OpenSecTradeContext(filter_trdmarket=ft.TrdMarket.US, host=ip, port=port, security_firm=ft.SecurityFirm.FUTUAU)
        ret, data = trd_ctx.unlock_trade(unlock_pwd, is_unlock=is_unlock)
        if ret == ft.RET_OK:
            warning.print('unlock success!')
        else:
            alarm.print('unlock_trade failed: ', data)
        trd_ctx.close()

    def get_positions(self):
        positions = []
        try:
            trd_ctx  = ft.OpenSecTradeContext(filter_trdmarket=ft.TrdMarket.US, host=ip, port=port, security_firm=ft.SecurityFirm.FUTUAU)
            ret, data = trd_ctx.position_list_query(acc_id=self.acc_id)
            if ret == ft.RET_OK:
                if data.shape[0] > 0:  # If the position list is not empty
                    for index, row in data.iterrows():
                        if row['qty'] > 0:
                            positions.append(row['code'].split('.')[1])  
            else:
                alarm.print('position_list_query error: ', data)
            trd_ctx.close()  # Close the current connection
        except Exception as e:
            alarm.print(e)
            trd_ctx.close()
        return positions

    def get_orders(self):
        'get orders with statuses: SUBMITTED, SUBMITTING, WAITING_SUBMIT'
        limit_if_touched_sell_orders = pd.DataFrame()
        stop_sell_orders = pd.DataFrame()
        limit_if_touched_buy_orders = pd.DataFrame()
        limit_buy_orders = pd.DataFrame()
        trailing_LIT_orders = pd.DataFrame()
        trailing_stop_limit_orders = pd.DataFrame()
        stop_limit_buy_orders = pd.DataFrame()
        stop_limit_sell_orders = pd.DataFrame()
        try:
            trd_ctx  = ft.OpenSecTradeContext(filter_trdmarket=ft.TrdMarket.US, host=ip, port=port, security_firm=ft.SecurityFirm.FUTUAU)
            ret, data = trd_ctx.order_list_query(acc_id=self.acc_id)
            if ret == ft.RET_OK:
                if data.shape[0] > 0:  # If the orderds list is not empty
                    for index, row in data.iterrows():
                        if (row['order_status'] == ft.OrderStatus.SUBMITTED  \
                                 or row['order_status'] == ft.OrderStatus.SUBMITTING \
                                 or row['order_status'] == ft.OrderStatus.WAITING_SUBMIT):
                            
                            if row['trd_side'] == ft.TrdSide.SELL:
                                if row['order_type'] == ft.OrderType.LIMIT_IF_TOUCHED \
                                    and row['remark'] in ['', None]:                                          
                                    limit_if_touched_sell_orders = pd.concat([limit_if_touched_sell_orders, row], axis = 1)
                                if row['order_type'] == ft.OrderType.LIMIT_IF_TOUCHED \
                                       and row['remark'] == 'trailing_LIT':      
                                    trailing_LIT_orders = pd.concat([trailing_LIT_orders, row], axis = 1)
                                if row['order_type'] == ft.OrderType.STOP: 
                                    stop_sell_orders = pd.concat([stop_sell_orders, row], axis = 1)
                                if row['order_type'] == ft.OrderType.TRAILING_STOP_LIMIT:
                                    trailing_stop_limit_orders = pd.concat([trailing_stop_limit_orders, row], axis = 1)     
                                if row['order_type'] == ft.OrderType.STOP_LIMIT: 
                                    stop_limit_sell_orders = pd.concat([stop_limit_sell_orders, row], axis = 1)       
                                  
                            if row['trd_side'] == ft.TrdSide.BUY:
                                 if row['order_type'] == ft.OrderType.LIMIT_IF_TOUCHED: 
                                    limit_if_touched_buy_orders  = pd.concat([limit_if_touched_buy_orders , row], axis = 1)
                                 if row['order_type'] == ft.OrderType.NORMAL: 
                                    limit_buy_orders  = pd.concat([limit_buy_orders , row], axis = 1)        
                                 if row['order_type'] == ft.OrderType.STOP_LIMIT: 
                                    stop_limit_buy_orders  = pd.concat([stop_limit_buy_orders , row], axis = 1)        

                    limit_if_touched_sell_orders = limit_if_touched_sell_orders.transpose()
                    stop_sell_orders = stop_sell_orders.transpose()
                    limit_if_touched_buy_orders = limit_if_touched_buy_orders.transpose()
                    limit_buy_orders = limit_buy_orders.transpose()
                    trailing_LIT_orders = trailing_LIT_orders.transpose()
                    trailing_stop_limit_orders = trailing_stop_limit_orders.transpose()
                    stop_limit_buy_orders = stop_limit_buy_orders.transpose()
                    stop_limit_sell_orders = stop_limit_sell_orders.transpose()
            else:
                alarm.print('order_list_query error: ', data)
            trd_ctx.close()  # Close the current connection
        except Exception as e:
            alarm.print(e)
            trd_ctx.close()
        return  limit_if_touched_sell_orders, stop_sell_orders, limit_buy_orders, \
              limit_if_touched_buy_orders, trailing_LIT_orders, trailing_stop_limit_orders,\
              stop_limit_buy_orders, stop_limit_sell_orders
    
    def get_order_commission(self, order_id):
        commission = None
        try:
            trd_ctx  = ft.OpenSecTradeContext(filter_trdmarket=ft.TrdMarket.US, host=ip, port=port, security_firm=ft.SecurityFirm.FUTUAU)
            ret, data = trd_ctx.order_fee_query(order_id_list=[order_id], acc_id=self.acc_id)
            if ret == ft.RET_OK:
                commission = float(data['fee_amount'].values[0])
            else:
                alarm.print('order_fee_query error: ', data)
            trd_ctx.close()
        except Exception as e:
            alarm.print(e)
            trd_ctx.close()
        return commission

    def get_us_cash(self):
        us_cash = 0
        try:
            trd_ctx  = ft.OpenSecTradeContext(filter_trdmarket=ft.TrdMarket.US, host=ip, port=port, security_firm=ft.SecurityFirm.FUTUAU)
            ret, data = trd_ctx.accinfo_query(acc_id=self.acc_id, refresh_cache=True)
            if ret == ft.RET_OK:
                # cr = CurrencyConverter()
                # rate = cr.convert(1, 'HKD', 'USD')
                # avl_withdrawal_cash = float(data['avl_withdrawal_cash'].values[0] * rate)      
                us_cash = data['us_cash'].values[0] 
            else:
                alarm.print('accinfo_query error: ', data)
            trd_ctx.close()
        except Exception as e:
            alarm.print(e)
            trd_ctx.close()
        return float(us_cash)

if __name__ == "__main__":
    ma = Moomoo_API(ip, port, trd_env=ft.TrdEnv.SIMULATE, acc_id=111887)
    # data=ma.get_history_orders()
    # data.to_csv('sim_trade_hist.csv')
    # df = ma.get_list_of_trading_accounts()
    # df[0]
    # ticker = "AGSS"
    # ma.unlock_trade()
    # ma.place_limit_if_touched_order(ticker, price=0.4, qty=1)
    # ma.unlock_trade(is_unlock=False)
    # ma.place_stop_order('MA', 480, 1)
    
    # Subscription check
    quote_ctx = ft.OpenQuoteContext(host='127.0.0.1', port=11111)
    print('current subscription status :', quote_ctx.query_subscription()) # Query the initial subscription status
    ret_sub, err_message = quote_ctx.subscribe(['US.AAPL'], [ft.SubType.QUOTE, ft.SubType.TICKER], subscribe_push=False)
    # First subscribed to the two types of QUOTE and TICKER. After the subscription is successful, OpenD will continue to receive pushes from the server, False means that there is no need to push to the script temporarily
    if ret_sub == ft.RET_OK: # Subscription successful
        print('subscribe successfully! current subscription status :', quote_ctx.query_subscription()) # Query subscription status after successful subscription
        time.sleep(60) # You can unsubscribe at least 1 minute after subscribing
        ret_unsub, err_message_unsub = quote_ctx.unsubscribe(['US.AAPL'], [ft.SubType.QUOTE])
        if ret_unsub == ft.RET_OK:
            print('unsubscribe successfully! current subscription status:', quote_ctx.query_subscription()) # Query the subscription status after canceling the subscription
        else:
            print('unsubscription failed!', err_message_unsub)
    else:
        print('subscription failed', err_message)
    quote_ctx.close() # After using the connection, remember to close it to prevent the number of connections from running out
    
    #test historical data
    quote_ctx = ft.OpenQuoteContext(host='127.0.0.1', port=11111)
    ret, data, page_req_key = quote_ctx.request_history_kline('US.AAPL', start='2019-09-11', end='2019-09-18', max_count=5) # 5 per page, request the first page
    if ret == ft.RET_OK:
        print(data)
        print(data['code'][0]) # Take the first stock code
        print(data['close'].values.tolist()) # The closing price of the first page is converted to a list
    else:
        print('error:', data)
    while page_req_key != None: # Request all results after
        print('*************************************')
        ret, data, page_req_key = quote_ctx.request_history_kline('US.AAPL', start='2019-09-11', end='2019-09-18', max_count=5,page_req_key=page_req_key) # Request the page after turning data
        if ret == ft.RET_OK:
            print(data)
        else:
            print('error:', data)
    print('All pages are finished!')
    quote_ctx.close() # After using the connection, remember to close it to prevent the number of connections from running out


