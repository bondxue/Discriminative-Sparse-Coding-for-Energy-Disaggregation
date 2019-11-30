import pandas as pd
import numpy as np
import datetime
import os
import sys
import pandas as pd
from collections import defaultdict
import glob
import warnings
warnings.filterwarnings("ignore")

def read_data():
    df = pd.DataFrame()
    houses = []
    path = "datasets/*.csv"
    for fname in glob.glob(path):
        house_df = pd.read_csv(fname)
        houseID = int(fname.split('.')[0].split('\\')[1])
        
        house_df['DateTime'] = pd.to_datetime(house_df['DateTime'])
        mask = (house_df['DateTime'] >= '2014-1-1') & (house_df['DateTime'] < '2015-1-1')
        house_df = house_df.loc[mask]
        house_df = house_df.set_index('DateTime')
        house_df = house_df.resample('1H').first()  # resample from 15 mins to 1 hour 
        house_df = house_df.reset_index(drop=False)
        
        if len(house_df) == 8760 and house_df.isnull().sum().sum() <= 100: # one year hours 
            houses.append(houseID)
            house_df.columns = ['localhour', 'use', 'air1', 'furnace1', 'dishwasher1', 'regrigerator1']  
            house_df = house_df.fillna(method='pad') 
            house_df['regrigerator1'] = house_df['regrigerator1'].fillna(method='ffill')
            appliances_sum = house_df[['air1', 'furnace1', 'dishwasher1', 'regrigerator1']].sum(axis=1)
            house_df['other'] = house_df['use'].subtract(appliances_sum)
            house_df['house'] = houseID
            house_df = house_df.set_index('house')
           
            df = pd.concat([df, house_df])
                               
    return df, houses

def format_data(df, houses):
    '''
    Parameters: dataframe of the apppliacnes
    Return: dictionary contains all X^T x m
    '''
    d = {}
    for appliance in df.columns.tolist():
        started = 0
        
        for i in houses:
            if started == 0:
                d[str(appliance)] = df[[str(appliance)]][df[str(appliance)].index == i]
                started = 1
                dfindex = d[str(appliance)].index    
            else:
                d[str(appliance)][str(i)] = pd.Series(df[str(appliance)][df[str(appliance)].index == i].values,index=dfindex)

        d[str(appliance)]=d[str(appliance)].rename(columns = {str(appliance):str(dfindex[0])})
        d[str(appliance)].reset_index(drop=True, inplace=True)
        
    return d

def split(d,portion,timeframe):
    '''
    Parameters: d = dictionary, portion 0.5 - 0.9, timeframe 1-8760
    Return: x_train,x_test dictionarys containing dataframes of all the appliances within the timeframe.
    '''
    x_train = {}
    x_test = {}
    timeframe = range(timeframe)
    train_list  = timeframe[int(len(timeframe) * 0.0):int(len(timeframe) * portion)]
    test_list = timeframe[int(len(timeframe) * portion):int(len(timeframe) * 1.0)]

    for key in d.keys():
        x_train[key] = d[key].loc[train_list,:]
        x_test[key] = d[key].loc[test_list,:]

    return x_train,x_test