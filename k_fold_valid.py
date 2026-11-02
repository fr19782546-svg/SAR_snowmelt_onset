# -*- coding: utf-8 -*-
"""
File:        k_fold_val.py
Description: 
             
Created on:  Mon Nov 02 2026 20:17:01
Author:      ruifang
University:  Southwest Jiaotong
Email:       fr19782546@my.swjtu.edu.cn
"""
#%%
import numpy as np
import pandas as pd
import pathlib as pl
import os
import glob
from sklearn.model_selection import KFold 

#---------import data-----------
station_dir = '/Users/king/Documents/Snowmelt_onset/OBS/raw_db_csv_by_year'
station_list = glob.glob(station_dir + '/*.csv')


smo_obs_file = '/Users/king/Documents/Snowmelt_onset/' \
'OBS/SMO_OBS/SMO_OBS_max.csv'
smo_obs = pd.read_csv(smo_obs_file)



smo_samples = smo_obs[['station','year','SMO_OBS']]

smo_samples = smo_samples.sort_values(
    by=['year', 'station']
).reset_index(drop=True)

ids = np.arange(len(smo_samples)) 
rng = np.random.default_rng(42)  
rng.shuffle(ids)
folds = np.array_split(ids, 10)  
k_values = np.round(np.arange(1, 8.01, 0.1), 2)

fold_results = []
#---------smo_sar calculate-----------
def k_fold_result(station_dir,year,obs,station,k):
    sta = f'{station_dir}/raw_db_{year}.csv'
    data = pd.read_csv(sta)
    db = data[['DOY',station]]
    apr_start = pd.Timestamp(int(year),4,1).day_of_year
    apr_end = pd.Timestamp(int(year),4,30).day_of_year
    apr_db = db.loc[(db['DOY'] >= apr_start) & (db['DOY'] <= apr_end),
                    station].dropna()
    apr_median = np.median(apr_db)
    mad = np.median(np.abs(apr_db - apr_median))
    threshold = apr_median - k * mad       
    db_daily = db.set_index('DOY').loc[:,station]
    db_daily = db_daily.reindex(range(90,241))
    below = db_daily <= threshold
    three_days = below.rolling(3).sum() == 3
    error = np.nan
    if three_days.any():
        smo_sar = three_days[three_days].index[0] - 2
        if 120 <= smo_sar <= 240: 
            error = smo_sar - obs
    return error

#---------k_fold_result-----------
for fold  in range(10): 
    test_idx = folds[fold]
    train_idx = np.setdiff1d(ids, test_idx)
    train_data = smo_samples.loc[train_idx].reset_index(drop=True)
    test_data = smo_samples.loc[test_idx].reset_index(drop=True)
    results = []
    for k in k_values:
        errors = []
        for i in range(len(train_data)):
            station = train_data.loc[i,'station']
            year = train_data.loc[i,'year']
            obs = train_data.loc[i,'SMO_OBS']
            # sta = f'{station_dir}/raw_db_{year}.csv'
            # data = pd.read_csv(sta)
            # db = data[['DOY',station]]
            # apr_start = pd.Timestamp(int(year),4,1).day_of_year
            # apr_end = pd.Timestamp(int(year),4,30).day_of_year
            # apr_db = db.loc[(db['DOY'] >= apr_start) & (db['DOY'] <= apr_end),
            #                 station].dropna()
            # apr_median = np.median(apr_db)
            # mad = np.median(np.abs(apr_db - apr_median))
            # threshold = apr_median - k * mad
            # db_daily = db.set_index('DOY').loc[:,station]
            # db_daily = db_daily.reindex(range(90,241))
            # below = db_daily <= threshold
            # three_days = below.rolling(3).sum() == 3
            # if three_days.any():
            #     smo_sar = three_days[three_days].index[0] - 2
            #     if 120 <= smo_sar <= 240: 
            #         error = smo_sar - obs

            error = k_fold_result(station_dir,year,obs,station,k)
            if np.abs(error) >= 0:
                errors.append(error)
            
            
                
        p = len(errors) / len(train_data)
        if len(errors) > 0:
            errors = np.array(errors)
            rmse = np.sqrt(np.mean(np.array(errors)**2))
        else:
            rmse = np.nan
        results.append([k,p,rmse])

    k_result = pd.DataFrame(results, columns=['k', 'p', 'rmse'])

    min_n = int(np.floor(0.85 * len(train_data) + 0.5))

    valid_k = k_result.loc[
        (k_result['p'] * len(train_data) >= min_n) &
        (k_result['rmse'].notna()),
        ['k', 'rmse']
    ]

    if len(valid_k) == 0:
        fold_results.append([fold + 1, np.nan, np.nan, np.nan])
        continue

    min_rmse = valid_k['rmse'].min()

    tied_k = valid_k.loc[
        np.isclose(valid_k['rmse'], min_rmse, atol=1e-12),
        'k'
    ]

    best_k = np.round(tied_k.median(), 1)

    test_errors = []
    for i in range(len(test_data)):
        station = test_data.loc[i,'station']
        year = test_data.loc[i,'year']
        obs = test_data.loc[i,'SMO_OBS']
        test_error = k_fold_result(station_dir,year,obs,station,best_k)
        if np.abs(test_error) >= 0:
                test_errors.append(test_error)
    test_p = len(test_errors) / len(test_data)
    if len(test_errors) > 0:
        test_errors = np.array(test_errors)
        test_rmse = np.sqrt(np.mean(np.array(test_errors)**2))
    else:
        test_rmse = np.nan
    fold_results.append([fold,best_k,test_p,test_rmse])
fold_results = pd.DataFrame(fold_results,columns=['fold','best_k','test_p','test_rmse'])



#%%
menak = np.mean(np.array(fold_results['best_k']))
std = np.std(np.array(fold_results['best_k']))