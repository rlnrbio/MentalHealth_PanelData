# -*- coding: utf-8 -*-
"""
Created on Tue May 31 17:58:22 2022

@author: rapha
"""

import pandas as pd
import scipy.stats as st
import statsmodels.api as sm
import statsmodels.formula.api as smf
from matplotlib import pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np
import re

import seaborn as sns

from sklearn import preprocessing
import matplotlib.pyplot as plt



import scipy.stats as st
import statsmodels.api as sm
import statsmodels.graphics.tsaplots as tsap
from statsmodels.compat import lzip
from statsmodels.stats.diagnostic import het_white
import seaborn as sns


from initial_data_loading import calc_ages, load_alc, load_health_long, load_b5, load_biodata, load_wealthdata

b5_heads = ['open', 'cons', 'extra', 'neuro', 'agree']

# load data

mcs_long = load_health_long()
b5, high_low = load_b5()

indiv_data = load_biodata()
wealth_data = load_wealthdata()

alcohol_data = load_alc()


b5_years = [2005,2009]
mcs_years = [2006, 2010]
wealth_years = [2007, 2012]

alcohol_years = [2006, 2010]

total_combined = pd.DataFrame()

b5 = pd.merge(b5, high_low)

for i in range(len(b5_years)):
    b5y = b5_years[i]
    mcsy = mcs_years[i]
    wy = wealth_years[i]
    alc = alcohol_years[i]
    
    b5_year  = b5[b5["syear"] == b5y]
    
    
    indiv_year = indiv_data
    indiv_year.loc[:,"age"] = calc_ages(indiv_year, b5y)
    
    mcs_year = mcs_long[mcs_long["syear"] == mcsy].loc[:,["pid", "mcs_std"]]
    
    wealth_year = wealth_data[wealth_data["syear"] == wy].loc[:,["pid", "wealth_std"]]
    
    alc_year = alcohol_data[alcohol_data["syear"] == alc].loc[:,["pid", "alcohol_combined"]]
    
    combined = pd.merge(b5_year, indiv_year, on=['pid','pid'])
    combined = pd.merge(combined, mcs_year, on=['pid','pid'])
    combined = pd.merge(combined, wealth_year, on=['pid','pid'])
    combined = pd.merge(combined, alc_year, on=['pid','pid'])

    total_combined = total_combined.append(combined)
    
#Define the y and X variable names
y_var_name = 'mcs_std'
X_var_names =  ["sex", "age", "alcohol_combined", "wealth_std"]
expr = y_var_name + " ~ "
i = 0
for X_var_name in X_var_names:
    if i > 0:
        expr = expr + ' + ' + X_var_name
    else:
        expr = expr + X_var_name
    i = i + 1

# dataset created
for i in range(len(b5_heads)):
    print(b5_heads)
    temp_b5 = b5_heads.copy()
    temp_expr = expr
    trait = b5_heads[i]
    temp_b5.pop(i)
    
    temp_expr = temp_expr + " + {}_low + {}_high".format(trait, trait)
    
    for t in temp_b5:
        temp_expr = temp_expr + ' + ' + t
        
    print('\n\nRegression expression for OLS with dummies=' + temp_expr)

    lsdv_model = smf.ols(formula=temp_expr, data=total_combined)
    lsdv_model_results = lsdv_model.fit()
    print('===============================================================================')
    print('============================== OLSR With Dummies ==============================')
    print(lsdv_model_results.summary())
    print('LSDV='+str(lsdv_model_results.ssr))

    
