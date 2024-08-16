import time
import moomoo as ft
from forex_python.converter import CurrencyRates
from currency_converter import CurrencyConverter
import pandas as pd
from colog.colog import colog
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
            trd_ctx  =  trd_ctx  = ft.OpenSecTradeContext(filter_trdmarket=ft.TrdMarket.US, host=self.ip, port=self.port, security_firm=ft.SecurityFirm.FUTUAU)
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
        return data, order_id

    def place_limit_if_touched_order(self, ticker, price, qty):
        order_id = None
        try:
            trd_ctx  = ft.OpenSecTradeContext(filter_trdmarket=ft.TrdMarket.US, host=self.ip, port=self.port, security_firm=ft.SecurityFirm.FUTUAU)
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
                                    aux_price=price * 1.0001)
            print(f'Placing limit if touched order for {stock_code}')
            print(f'Market response is {ret}, data is {data}')
            if ret == ft.RET_OK:
                order_id = data['order_id'].values[0]
            else:
                alarm.print(data)
        except Exception as e:
            alarm.print(e)
        return order_id

    def place_stop_order(self, ticker, price, qty):
        order_id = None
        try:
            trd_ctx  = ft.OpenSecTradeContext(filter_trdmarket=ft.TrdMarket.US, host=self.ip, port=self.port, security_firm=ft.SecurityFirm.FUTUAU)
            stock_code = MARKET + ticker
            ret, data = trd_ctx.place_order(
                                    price=price,
                                    qty=qty,
                                    code=stock_code,
                                    trd_side=ft.TrdSide.SELL,
                                    trd_env=self.trd_env,
                                    time_in_force=ft.TimeInForce.GTC,
                                    order_type=ft.OrderType.STOP,
                                    aux_price=price * 0.9995)
            print(f'Placing stop order for {stock_code}')
            print(f'Market response is {ret}, data is {data}')
            if ret == ft.RET_OK:
                order_id = data['order_id'].values[0]
            else:
                alarm.print(data)
            trd_ctx.close()
        except Exception as e:
            alarm.print(e)
        return order_id
    
    def cancel_order(self, order_id):
        status = None
        try:
            trd_ctx  = ft.OpenSecTradeContext(filter_trdmarket=ft.TrdMarket.US, host=self.ip, port=self.port, security_firm=ft.SecurityFirm.FUTUAU)
            self.unlock_trade()
            ret, data = trd_ctx.modify_order(ft.ModifyOrderOp.CANCEL, order_id, 0, 0)
            if ret == ft.RET_OK:
                status = 'Canceled'
                warning.print(f'{order_id} has been canceled ')  # Get the order ID of the modified order
            else:
                alarm.print('modify_order error: ', data)
            trd_ctx.close()
        except Exception as e:
            alarm.print(e)
        return status
 
    def get_history_orders(self):
        data = None
        try:
            trd_ctx  = ft.OpenSecTradeContext(filter_trdmarket=ft.TrdMarket.US, host=ip, port=port, security_firm=ft.SecurityFirm.FUTUAU)
            ret, data = trd_ctx.history_order_list_query(acc_id=self.acc_id, trd_env=self.trd_env)
            if ret == ft.RET_OK:
                if data.shape[0] > 0:  # If the order list is not empty
                    print('history orders received successfully')
            else:
                alarm.print('history_order_list_query error: ', data)
            trd_ctx.close()
        except Exception as e:
            alarm.print(e)
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
                        if row['can_sell_qty'] > 0:
                            positions.append(row['code'])  
            else:
                alarm.print('position_list_query error: ', data)
            trd_ctx.close()  # Close the current connection
        except Exception as e:
            alarm.print(e)
        return positions

    def get_sell_orders(self):
        limit_if_touched_sell_orders = pd.DataFrame()
        stop_sell_orders = pd.DataFrame()
        try:
            trd_ctx  = ft.OpenSecTradeContext(filter_trdmarket=ft.TrdMarket.US, host=ip, port=port, security_firm=ft.SecurityFirm.FUTUAU)
            ret, data = trd_ctx.order_list_query(acc_id=self.acc_id)
            if ret == ft.RET_OK:
                if data.shape[0] > 0:  # If the orderds list is not empty
                    for index, row in data.iterrows():
                        if row['trd_side'] == ft.TrdSide.SELL \
                            and (row['order_status'] == ft.OrderStatus.SUBMITTED  \
                                 or row['order_status'] == ft.OrderStatus.SUBMITTING \
                                 or row['order_status'] == ft.OrderStatus.WAITING_SUBMIT):

                            if row['order_type'] == ft.OrderType.LIMIT_IF_TOUCHED:         
                                # limit_if_touched_sell_orders.append(row)  
                                limit_if_touched_sell_orders = pd.concat([limit_if_touched_sell_orders, row], axis = 1)
                            
                            if row['order_type'] == ft.OrderType.STOP: 
                                stop_sell_orders = pd.concat([stop_sell_orders, row], axis = 1)        
                                # stop_sell_orders.append(row)
                    limit_if_touched_sell_orders = limit_if_touched_sell_orders.transpose()
                    stop_sell_orders = stop_sell_orders.transpose()
            else:
                alarm.print('order_list_query error: ', data)
            trd_ctx.close()  # Close the current connection
        except Exception as e:
            alarm.print(e)
        return  limit_if_touched_sell_orders, stop_sell_orders
    
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
        return commission

    def get_avl_withdrawal_cash(self):
        avl_withdrawal_cash = 0
        try:
            trd_ctx  = ft.OpenSecTradeContext(filter_trdmarket=ft.TrdMarket.US, host=ip, port=port, security_firm=ft.SecurityFirm.FUTUAU)
            ret, data = trd_ctx.accinfo_query(acc_id=self.acc_id, refresh_cache=True)
            if ret == ft.RET_OK:
                cr = CurrencyConverter()
                rate = cr.convert(1, 'HKD', 'USD')
                avl_withdrawal_cash = float(data['avl_withdrawal_cash'].values[0] * rate)      
            else:
                alarm.print('accinfo_query error: ', data)
            trd_ctx.close()
        except Exception as e:
            alarm.print(e)
        return avl_withdrawal_cash

if __name__ == "__main__":
    ma = Moomoo_API(ip, port, trd_env=ft.TrdEnv.SIMULATE, acc_id=111887)
    data=ma.get_history_orders()
    data.to_csv('sim_trade_hist.csv')
    # df = ma.get_list_of_trading_accounts()
    # df[0]
    # ticker = "AGSS"
    # ma.unlock_trade()
    # ma.place_limit_if_touched_order(ticker, price=0.4, qty=1)
    # ma.unlock_trade(is_unlock=False)
    # ma.place_stop_order('MA', 480, 1)




