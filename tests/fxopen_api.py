web_api_address = "https://ttdemowebapi.soft-fx.com:8443";
web_api_id = "1de621ca-e686-4ee2-92a5-45c87b4b3fe5";
web_api_key = "czNhCcnK6ydePCHZ";
web_api_secret = "J6Jxc2xPr8JyNpWtyEaCPYpkpJpsSQ38xb9AZNxBAGdtQrNDhQwf9mkWQygCKd6K";

# Create instance of the TickTrader Web API client
client = TickTraderWebClient(web_api_address, web_api_id, web_api_key, web_api_secret)
# Public trade session status
public_trade_session = client.get_public_trade_session()
print('TickTrader name: {0}'.format(public_trade_session['PlatformName']))
print('TickTrader company: {0}'.format(public_trade_session['PlatformCompany']))
print('TickTrader address: {0}'.format(public_trade_session['PlatformAddress']))
print('TickTrader timezone offset: {0}'.format(public_trade_session['PlatformTimezoneOffset']))
print('TickTrader session status: {0}'.format(public_trade_session['SessionStatus']))

# Public feed ticks
ticks = client.get_public_all_ticks()
for t in ticks:
    print('{0} tick: {1} {2}'.format(t['Symbol'], t['BestBid']['Price'], t['BestAsk']['Price']))

tick = client.get_public_tick(ticks[0]['Symbol'])
print("{0} tick timestamp: {1}".format(tick[0]['Symbol'], tick[0]['Timestamp']))
