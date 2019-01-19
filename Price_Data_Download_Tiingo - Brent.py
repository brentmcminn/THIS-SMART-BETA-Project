# -*- coding: utf-8 -*-
"""
@author: evanf

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests as r
import os

def df_rename(df, column, ticker):
    ''' This function extracts a column from a DataFrame and creates another 
        DataFrame with the values from this column, but the name changed to 
        the ticker parameter specified'''
    return pd.DataFrame(df[column].values,
                           index = stock_data.index,
                           columns = [ticker])

homedir = os.environ['HOMEPATH']
wdir = os.path.join(homedir , 'Dropbox/Data/Equity/US')

constituents = pd.read_csv(wdir+'/Historical Constituents.csv',
                           index_col = 0,
                           parse_dates = True)

constituents.columns

url = 'https://api.tiingo.com/tiingo/daily'
API_key = #Your key here

headers = {'Content-Type': 'application/json',
           'Authorization' : 'Token '+API_key}

parameters = {'startDate':'2009-12-31',
          'endDate':'2018-10-27'}

Close = pd.read_csv(wdir+'/close.csv',
                    index_col = 0,
                    parse_dates = True)

error_stocks = []
for ticker in np.setdiff1d(constituents.columns, Close.columns):
    if ticker == 'SHLD': continue

    requestResponse = r.get(url+'/'+ticker+'/prices?',
                              headers=headers,
                              params = parameters)
    try:
        stock_data = pd.DataFrame(requestResponse.json())
    except ValueError:
        print(ticker)
    try:

        stock_data.set_index('date', inplace = True)
    except KeyError:
        print(ticker)
        error_stocks.append(ticker)
        continue
    
    if 'adjClose' not in locals():
        adjClose = df_rename(stock_data,'adjClose',ticker)
        adjHigh = df_rename(stock_data,'adjHigh',ticker)
        adjLow = df_rename(stock_data,'adjLow',ticker)
        adjOpen = df_rename(stock_data,'adjOpen',ticker)
        adjVolume = df_rename(stock_data,'adjVolume',ticker)
        Close = df_rename(stock_data,'close',ticker)
        divCash = df_rename(stock_data,'divCash',ticker)
        High = df_rename(stock_data,'high',ticker)
        Low = df_rename(stock_data,'low',ticker)
        Open = df_rename(stock_data,'open',ticker)
        splitFactor = df_rename(stock_data,'splitFactor',ticker)
        Volume = df_rename(stock_data,'volume',ticker)
    else:
        adjClose = adjClose.join(df_rename(stock_data,'adjClose',ticker), 
                                 how = 'outer')
        adjHigh = adjHigh.join(df_rename(stock_data,'adjHigh',ticker), 
                               how = 'outer')
        adjLow = adjLow.join(df_rename(stock_data,'adjLow',ticker), 
                             how = 'outer')
        adjOpen = adjOpen.join(df_rename(stock_data,'adjOpen',ticker), 
                               how = 'outer')
        adjVolume = adjVolume.join(df_rename(stock_data,'adjVolume',ticker), 
                                   how = 'outer')
        Close = Close.join(df_rename(stock_data,'close',ticker), 
                           how = 'outer')
        divCash = divCash.join(df_rename(stock_data,'divCash',ticker), 
                               how = 'outer')
        High = High.join(df_rename(stock_data,'high',ticker), 
                         how = 'outer')
        Low = Low.join(df_rename(stock_data,'low',ticker), 
                       how = 'outer')
        Open = Open.join(df_rename(stock_data,'open',ticker), 
                         how = 'outer')
        splitFactor = splitFactor.join(df_rename(stock_data,'splitFactor'
                                                 ,ticker), 
                                       how = 'outer')
        Volume = Volume.join(df_rename(stock_data,'volume',ticker), 
                             how = 'outer')

pd.DataFrame(error_stocks).to_csv('error.csv')
adjClose.to_csv('adjClose.csv')
adjHigh.to_csv('adjHigh.csv')
adjLow.to_csv('adjLow.csv')
adjOpen.to_csv('adjOpen.csv')
adjVolume.to_csv('adjVolume.csv')
Close.to_csv('close.csv')
divCash.to_csv('divCash.csv')
High.to_csv('high.csv')
Low.to_csv('low.csv')
Open.to_csv('open.csv')
splitFactor.to_csv('splitFactor.csv')
Volume.to_csv('volume.csv')

