import time
from moomoo import *
# class OrderBookTest(OrderBookHandlerBase):
#     def on_recv_rsp(self, rsp_pb):
#         ret_code, data = super(OrderBookTest,self).on_recv_rsp(rsp_pb)
#         if ret_code != RET_OK:
#             print("OrderBookTest: error, msg: %s"% data)
#             return RET_ERROR, data
#         print("OrderBookTest ", data) # OrderBookTest's own processing logic
#         return RET_OK, data
# quote_ctx = OpenQuoteContext(host='127.0.0.1', port=11111)
# handler = OrderBookTest()
# quote_ctx.set_handler(handler) # Set real-time swing callback
# quote_ctx.subscribe(['HK.00700'], [SubType.ORDER_BOOK]) # Subscribe to the order type, OpenD starts to receive continuous push from the server
# time.sleep(15) # Set the script to receive OpenD push duration to 15 seconds
# quote_ctx.close() # Close the current link, OpenD will automatically cancel the corresponding type of subscription for the corresponding stock after 1 minute


# quote_ctx = OpenQuoteContext(host='127.0.0.1', port=11111)

# quote_ctx.subscribe(['HK.00700'], [SubType.QUOTE])
# ret, data = quote_ctx.query_subscription()
# if ret == RET_OK:
#     print(data)
# else:
#     print('error:', data)
# quote_ctx.close() # After using the connection, remember to close it to prevent the number of connections from running out


# from moomoo import *
# quote_ctx = OpenQuoteContext(host='127.0.0.1', port=11111)

# ret, data = quote_ctx.get_market_snapshot(['US.AMD'])
# if ret == RET_OK:
#     print(data)
#     print(data['code'][0])    # Take the first stock code
#     print(data['code'].values.tolist())   # Convert to list
# else:
#     print('error:', data)
# quote_ctx.close() # After using the connection, remember to close it to prevent the number of connections from running out

import time
from moomoo import *
class OrderBookTest(OrderBookHandlerBase):
    def on_recv_rsp(self, rsp_pb):
        ret_code, data = super(OrderBookTest,self).on_recv_rsp(rsp_pb)
        if ret_code != RET_OK:
            print("OrderBookTest: error, msg: %s"% data)
            return RET_ERROR, data
        print("OrderBookTest ", data) # OrderBookTest's own processing logic
        return RET_OK, data
quote_ctx = OpenQuoteContext(host='127.0.0.1', port=11111)
handler = OrderBookTest()
quote_ctx.set_handler(handler) # Set real-time swing callback
quote_ctx.subscribe(['HK.0005'], [SubType.ORDER_BOOK]) # Subscribe to the order type, OpenD starts to receive continuous push from the server
time.sleep(15) # Set the script to receive OpenD push duration to 15 seconds
quote_ctx.close() # Close the current link, OpenD will automatically cancel the corresponding type of subscription for the corresponding stock after 1 minute
