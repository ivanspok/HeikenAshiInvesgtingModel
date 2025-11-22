# Version 7.0 (speed_norm100 list)
# stock_name_list_opt = ['HD', 'C', 'SBUX', 'DE', 'CRM', 'SHW', 'CHTR', 'FTNT', 'IBM', 'PNC', 'ORCL', 'SPG',
#                         'BA', 'CB', 'ANET', 'APD', 'ON', 'MRNA', 'WFC', 'COF', 'AMAT', 'MS', 'DHI', 'LOW', 
#                         'QCOM', 'ADI', 'ETN', 'MCO', 'NSC', 'CMCSA', 'PANW', 'CAT', 'IDXX', 'INTC', 'ACN',
#                           'ADBE', 'ADSK', 'APH', 'WMB', 'CSX', 'RTX', 'AIG', 'PCAR', 'XOM', 'TJX', 'CSCO',
#                             'CTAS', 'JPM', 'NOW', 'NFLX', 'MRK', 'SYK', 'GS', 'TDG', 'GOOG', 'CMG', 'HCA', 
#                             'BLK', 'FDX', 'PM', 'VZ', 'BSX', 'GE', 'AON', 'WELL', 'ABT', 'ABBV', 'PH', 'PG', 
#                             'ICE', 'TT', 'ECL', 'HLT', 'ISRG', 'JNJ', 'LIN', 'MCD', 'V', 'DHR', 'COST', 'MA',
#                               'INTU', 'LMT', 'REGN', 'ROK', 'BAC', 'GM', 'PSX', 'UPS', 'MPC', 'CARR', 'AVGO',
#                              'NKE', 'AMD', 'F', 'LRCX', 'KLAC']
# Version 8.0 with exlusion

stock_name_list = []
stock_name_list += ['AAPL', 'GOOG','JPM','XOM','UNH','JNJ','V','AVGO','PG','LLY','MA','HD','CVX','MRK', 
                       'PEP','COST','ABBV','ADBE','KO','CRM','WMT','MCD','CSCO','BAC','PFE','TMO','ACN','NFLX','ABT','AMD','LIN','ORCL','CMCSA',
                       'TXN','DIS','WFC','DHR','PM','NEE','VZ','INTC','RTX','HON','LOW','UPS','INTU','SPGI','NKE','COP','QCOM','BMY','CAT','UNP','BA','ISRG',
                        'GE','IBM','AMGN','AMAT','MDT','SBUX','PLD','NOW','MS','DE','BLK','GS','T','LMT','AXP','SYK','ADI','TJX','ELV','MDLZ','GILD','ADP','MMC',
                        'C','AMT','CVS','VRTX','SCHW','LRCX','MO','TMUS','SLB', 'ETN', 'ZTS', 'CI', 'PYPL']

stock_name_list += ['CB','SO','BSX','EQIX','BDX','PANW','DUK','EOG','MU','AON','ITW','CSX','SNPS','PGR','APD','KLAC','CME','NOC','CDNS','ICE',
                       'CL','SHW','WM','HCA','TGT','FCX','FDX','F','MMM','CMG','EW','GM','MCK','NXPI','MCO','NSC','HUM','EMR','DXCM','PNC','PH','MPC','APH',
                       'ROP','FTNT','MCHP','USB','CCI','MAR','MSI','GD','PSA','JCI','PSX','SRE','ADSK','AZO','TDG','ECL','AJG','KMB','TEL','TT','AEP','EL','PCAR',
                       'OXY','TFC','CARR','D','IDXX','GIS','ON','COF','ADM','MNST','NUE','CTAS','AIG','EXC','VLO','MRNA','ANET','WMB','O','STZ','IQV','HLT','CHTR','WELL',
                       'BIIB','SPG','MSCI','DHI','ROK']
# new stockes from 12/07/2025
stock_name_list += ['ALGN','CDW','ZBRA','VTRS','KHC','CINF','LUMN','BAX','CZR','FANG','HIG','HWM','KEYS','LNT','NDAQ',
                       'NTRS','ODFL','RMD','SEE','SWK','TROW','VFC','WAT','WST','ZION','AAL','ALB','AMCR',
                       'ASML','AVY','CAG','CHKP','CHRW','CNP','CTSH','ETSY',
                       'EXPE', 'FTV', 'GWW', 'HII', 'HWM', 'IPGP', 'JCI', 'KEYS', 'KMX', 'LHX',
                       'MASI', 'MORN', 'MSCI', 'PAYX',
                       'PKG', 'PNR', 'PPG', 'PRGO']

stock_name_list += ['QRVO', 'RHI', 'SEE', 'SWK', 'TROW', 'WAT',
                       'AAL', 'ALB', 'AMCR', 'ASML', 'AVY', 'CAG', 'CHKP', 'CHRW',
                       'CNP', 'CTSH', 'ETSY', 'EXPE', 'FTV', 'GWW', 'HII',
                       'HWM', 'IPGP', 'JCI', 'KEYS', 'KMX']

stock_name_list += ['NVDA', 'LMT', 'AMZN', 'TSLA', 'META', 'GOOGL', 'PYPL', 'ADBE', 'AVGO', 'CSCO', 'NFLX',
                          'INTU', 'AMD', 'NOW', 'ADSK', 'ZM', 'DOCU', 'FSLY', 'CRWD', 'SNOW', 'TEAM', 'OKTA', 'DDOG',
                          'NET', 'PANW', 'FTNT',  'MRVL', 'ANET', 'VRSN', 'CHKP', 'LRCX', 'KLAC',
                          'ASML', 'AMAT', 'MU', 'QCOM', 'TXN']

stock_name_list_opt = list(set(stock_name_list))

if __name__ == "__main__":
    print(f"Total {len(stock_name_list_opt)} stocks in the optimized stock list.")      