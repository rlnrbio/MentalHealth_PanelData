# -*- coding: utf-8 -*-
"""
Created on Mon May 30 22:20:50 2022

@author: rapha
"""
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
from initial_data_loading import load_data_long, path
# load data
workpath = "C:/Users/rapha/Documents/PanelDataProject/"
variable = "alcohol_16"
mcs_long = load_health_long()
b5, high_low = load_b5(plot_save = False)
b5 = pd.merge(b5, high_low)


indiv_data = load_biodata()
wealth_data = load_wealthdata()

alcohol_data = load_alc()

print("B5", np.unique(b5["syear"]))
print("mcs", np.unique(mcs_long["syear"]))

print("wealth", np.unique(wealth_data["syear"]))

print("alcohol", np.unique(alcohol_data["syear"]))

# initial check:
# OLS Regression between Gender, Age and Personality traits:

b5_years = [2005,2009]
mcs_years = [2006, 2010]
wealth_years = [2007, 2012]

alcohol_years = [2006, 2010]

tv_years = [2008,2013]
smoking_years = [2004,2008]

total_combined = pd.DataFrame()

for i in range(len(b5_years)):
    b5y = b5_years[i]
    mcsy = mcs_years[i]
    wy = wealth_years[i]
    alc = alcohol_years[i]
    
    b5_year  = b5[b5["syear"] == b5y]
    
    indiv_year = indiv_data
    indiv_year.loc[:,"age"] = calc_ages(indiv_year, b5y)
    
    mcs_year = mcs_long[mcs_long["syear"] == mcsy].loc[:,["pid", "mcs_std"]]
    pcs_year = mcs_long[mcs_long["syear"] == mcsy].loc[:,["pid", "pcs_std"]]
    
    wealth_year = wealth_data[wealth_data["syear"] == wy].loc[:,["pid", "wealth_std"]]
    
    alc_year = alcohol_data[alcohol_data["syear"] == alc].loc[:,["pid", "alcohol_16"]]
    
    combined = b5_year
    combined = pd.merge(b5_year, indiv_year, on=['pid','pid'])
    combined = pd.merge(combined, mcs_year, on=['pid','pid'])
    combined = pd.merge(combined, pcs_year, on=['pid','pid'])

    combined = pd.merge(combined, wealth_year, on=['pid','pid'])
    combined = pd.merge(combined, alc_year, on=['pid','pid'])

    total_combined = total_combined.append(combined)

populations = ["open_low", "open_high", "open_normal",
                   "cons_low", "cons_high", "cons_normal", 
                   "extra_low", "extra_high", "extra_normal", 
                   "neuro_low", "neuro_high", "neuro_normal", 
                   "agree_low", "agree_high", "agree_normal"]


combined_results = pd.DataFrame()
combined_results_pvals = pd.DataFrame()
combined_results_combined = pd.DataFrame()

f = open('results/{}_selected_results.txt'.format(variable), 'w')
f.write("StandardModel\n")


y_var_name = "mcs_std"
X_var_names = ["open", "cons", "extra", "neuro", "agree", 
               "sex", "age", "alcohol_16", "wealth_std", "pcs_std"]
#Carve out the pooled Y
pooled_y=total_combined[y_var_name]
#Carve out the pooled X
pooled_X=total_combined[X_var_names]
#Add the placeholder for the regression intercept. When the model is fitted, the coefficient of
# this variable is the regression model's intercept β_0.
pooled_X = sm.add_constant(pooled_X)
#Build the OLS model
pooled_olsr_model = sm.OLS(endog=pooled_y, exog=pooled_X, missing='drop')
#Train the model and fetch the results
pooled_olsr_model_results = pooled_olsr_model.fit()
#Print the results summary
f.write('===============================================================================')
f.write('================================= Pooled OLS ==================================')
f.write(str(pooled_olsr_model_results.summary()))

res = pooled_olsr_model_results.params
res.name = "combined"
combined_results = combined_results.append(res)
combined_results_combined = combined_results_combined.append(res)

pvals = pooled_olsr_model_results.pvalues
pvals.name = "pval_combined"
combined_results_pvals = combined_results_pvals.append(pvals)
combined_results_combined = combined_results_combined.append(pvals)

for pop in populations:
    print(pop)
    
    f.write("\n\nSelected Population: {}\n".format(pop))
    total_selected = total_combined[total_combined[pop] == 1]
    X_var_temp = X_var_names.copy()
    p = pop.split("_")[0]
    print(total_selected.shape)
    X_var_temp.remove(p)    
    
    #Carve out the pooled Y
    pooled_y=total_selected[y_var_name]
    #Carve out the pooled X
    pooled_X=total_selected[X_var_temp]
    #Add the placeholder for the regression intercept. When the model is fitted, the coefficient of
    # this variable is the regression model's intercept β_0.
    pooled_X = sm.add_constant(pooled_X)
    #Build the OLS model
    pooled_olsr_model = sm.OLS(endog=pooled_y, exog=pooled_X, missing='drop')
    #Train the model and fetch the results
    pooled_olsr_model_results = pooled_olsr_model.fit()
    #Print the results summary
    f.write('===============================================================================')
    f.write('================================= Pooled OLS ==================================')
    f.write(str(pooled_olsr_model_results.summary()))
    
    res = pooled_olsr_model_results.params
    res.name = pop
    combined_results = combined_results.append(res)
    combined_results_combined = combined_results_combined.append(res)

    pvals = pooled_olsr_model_results.pvalues
    pvals.name = "pval_{}".format(pop)
    combined_results_pvals = combined_results_pvals.append(pvals)
    combined_results_combined = combined_results_combined.append(pvals)

    f.write('Mean value of residual errors='+str(pooled_olsr_model_results.resid.mean()))

f.close()
with pd.ExcelWriter(workpath + "results/{}_combined_results.xlsx".format(variable)) as writer:
    combined_results.to_excel(writer, sheet_name='params')
    combined_results_pvals.to_excel(writer, sheet_name = "pvals")    
    combined_results_combined.to_excel(writer, sheet_name = "combined")    




