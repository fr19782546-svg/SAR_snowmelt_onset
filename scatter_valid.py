#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 00:31:49 2026

@author: king
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('/Users/king/Documents/Rubbish/SMO_three_scripts_configurable_240/max_obs_right_censored/fold_validation_scatter_rmse.csv')

years = data['year'].unique()

mean_adp = np.full(len(years),np.nan)
mean_fix = np.full(len(years),np.nan)
mean_adp_obs = np.full(len(years),np.nan)
mean_fix_obs = np.full(len(years),np.nan)
result = pd.DataFrame({
    'year':years,
    'mean_adp':mean_adp,
    'mean_fix':mean_fix,
    'mean_adp_obs':mean_adp_obs,
    'mean_fix_obs':mean_fix_obs
    }
    )

gray_data = data[
    data['SMO_SAR'].notna()&
    data['SMO_FIXED'].notna()&
    data['SMO_OBS'].notna()
    ]

for i, year in enumerate(years):
    valid_year = data[data['year'] == year]
    valid_adp = valid_year[valid_year['SMO_SAR'].notna()
                        & valid_year['SMO_OBS'].notna()]
    valid_fix = valid_year[valid_year['SMO_FIXED'].notna()
                        & valid_year['SMO_OBS'].notna()]
    adp   = valid_adp['SMO_SAR'].to_numpy()
    fix   = valid_fix['SMO_FIXED'].to_numpy()
    adp_obs   = valid_adp['SMO_OBS'].to_numpy()
    fix_obs   = valid_fix['SMO_OBS'].to_numpy()

    result.loc[i,'mean_adp'] = np.mean(adp)
    result.loc[i,'mean_fix'] = np.mean(fix)
    result.loc[i,'mean_adp_obs'] = np.mean(adp_obs)
    result.loc[i,'mean_fix_obs'] = np.mean(fix_obs)


adp_std = np.std(result['mean_adp']-result['mean_adp_obs'])
fix_std = np.std(result['mean_fix']-result['mean_fix_obs'])
adp_rmse = np.sqrt(np.mean((result['mean_adp']-result['mean_adp_obs'])**2))
fix_rmse = np.sqrt(np.mean((result['mean_fix']-result['mean_fix_obs'])**2))
adp_mae = np.mean(np.abs(result['mean_adp']-result['mean_adp_obs']))
fix_mae = np.mean(np.abs(result['mean_fix']-result['mean_fix_obs']))
adp_mbe = np.mean(result['mean_adp']-result['mean_adp_obs'])
fix_mbe = np.mean(result['mean_fix']-result['mean_fix_obs'])


plt.rcParams['font.family'] = 'Arial' 
fig, ax = plt.subplots(figsize=(8.3/2.54, 8.3/2.54)) 
fig.subplots_adjust(
    left=0.15,     # 
    right=0.97,    # 
    bottom=0.1,   #
    top=0.97      #
)
ax.set_ylim(120,200)
ax.set_xlim(120,200)

ax.set_yticks(np.arange(120,201,20))
ax.set_xticks(np.arange(120,201,20))
ax.scatter(gray_data['SMO_OBS'],gray_data['SMO_SAR'],facecolors='white',edgecolors = 'gray',marker = 'o',s=60,linewidths=0.5)
ax.scatter(gray_data['SMO_OBS'],gray_data['SMO_FIXED'],facecolors='white',edgecolors = 'gray',marker = '^',s=60,linewidths=0.5)
ax.errorbar(result['mean_adp_obs'],result['mean_adp'],
            xerr=adp_std,yerr=adp_std,fmt='none',
            color='#94c99e',linewidth=0.5,capsize=0,zorder=1)
ax.errorbar(result['mean_fix_obs'],result['mean_fix'],
            xerr=fix_std,yerr=fix_std,fmt='none',
            color='#e97e81',linewidth=0.5,capsize=0,zorder=1)
ADP = ax.scatter(result['mean_adp_obs'],
                result['mean_adp'],
                facecolors='#94c99e',
                edgecolors = 'black',
                marker = 'o',s=100,
                linewidths=0.9)
FIX = ax.scatter(result['mean_fix_obs'],
                result['mean_fix'],
                facecolors='#e97e81',
                edgecolors = 'black',
                marker = '^',s=100,
                linewidths=0.9)   

ax.tick_params(axis='both', which='major',direction='out',length=3.5,width=1.1,labelsize=8)
ax.set_xlabel('Observed DOY',fontsize=8,fontname='Arial',labelpad=0.7)
ax.set_ylabel('Predicted DOY',fontsize=8,fontname='Arial',labelpad=0.7)
leg = ax.legend((ADP,FIX),('Adaptive','Fixed'),
            loc='lower right',
            fontsize=8,frameon=True,
            edgecolor='black',
            fancybox=False,
            )
plt.plot([120,200],[120,200],color='black',linestyle='-',linewidth=1.1)
leg.get_frame().set_linewidth(1.2)
ax.spines[:].set_linewidth(1.1)
ax.text(140,195,'      Fix',fontsize=9,fontname='Arial')
ax.text(131,195,'      Adp',fontsize=9,fontname='Arial')
ax.text(123,188,'RMSE   4.6   6.4',fontsize=9,fontname='Arial')
ax.text(123,181,'   MAE   4.2   5.3',fontsize=9,fontname='Arial')
ax.text(123,173,'   MBE   0.1   5.2',fontsize=9,fontname='Arial')

plt.savefig('output.png',
            dpi=300,
            )
plt.show()







