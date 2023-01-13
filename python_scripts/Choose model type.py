# -*- coding: utf-8 -*-
"""
Created on Mon May 30 22:20:50 2022

@author: rapha
"""
import pandas as pd
import numpy as np
#import re

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
variable = "smoking"
mcs_long = load_health_long()
mcs_long = mcs_long.dropna()
b5, high_low = load_b5()
b5 = pd.merge(b5, high_low)


indiv_data = load_biodata()
wealth_data = load_wealthdata()

personal_df = pd.read_stata(path + "pl.dta")

smoking_data = load_data_long(personal_df, "smoking")
smoking_data = smoking_data.replace([2,1], [0,1])

fig4, ax4 = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
ax4.hist(smoking_data["ple0081_v2"], bins  = 2)
fig4.savefig(workpath + "figures/smoking.png")

print("B5", np.unique(b5["syear"]))
print("mcs", np.unique(mcs_long["syear"]))

print("wealth", np.unique(wealth_data["syear"]))

print("smoking", np.unique(smoking_data["syear"]))


# initial check:
# OLS Regression between Gender, Age and Personality traits:

b5_years = [2005, 2009, 2013, 2017, 2019]
mcs_years = [2006, 2010, 2014, 2018, 2020]
wealth_years = [2002, 2007, 2012, 2017, 2017]

smoking_years = [2006, 2010, 2014, 2018, 2020]

#tv_years = [2008,2013]
#smoking_years = [2004,2008]

total_combined = pd.DataFrame()

for i in range(len(b5_years)):
    b5y = b5_years[i]
    mcsy = mcs_years[i]
    wy = wealth_years[i]
    smk = smoking_years[i]
    
    b5_year  = b5[b5["syear"] == b5y]
    
    indiv_year = indiv_data
    indiv_year.loc[:,"age"] = calc_ages(indiv_year, b5y)
    
    mcs_year = mcs_long[mcs_long["syear"] == mcsy].loc[:,["pid", "mcs_std"]]
    pcs_year = mcs_long[mcs_long["syear"] == mcsy].loc[:,["pid", "pcs_std"]]

    
    wealth_year = wealth_data[wealth_data["syear"] == wy].loc[:,["pid", "wealth_std"]]
    
    smoke_year = smoking_data[smoking_data["syear"] == smk].loc[:,["pid", "ple0081_v2"]]
    
    combined = pd.merge(b5_year, indiv_year, on=['pid','pid'])
    combined = pd.merge(combined, mcs_year, on=['pid','pid'])
    combined = pd.merge(combined, pcs_year, on=['pid','pid'])

    combined = pd.merge(combined, wealth_year, on=['pid','pid'])
    combined = pd.merge(combined, smoke_year, on=['pid','pid'])

    total_combined = total_combined.append(combined)

populations = ["open_low", "open_high", "open_normal",
                   "cons_low", "cons_high", "cons_normal", 
                   "extra_low", "extra_high", "extra_normal", 
                   "neuro_low", "neuro_high", "neuro_normal", 
                   "agree_low", "agree_high", "agree_normal"]


combined_results = pd.DataFrame()
combined_results_pvals = pd.DataFrame()
combined_results_combined = pd.DataFrame()

total_combined= total_combined.rename(columns = {"syear":"year"})
total_combined["year"] = (total_combined["year"].astype(int))

total_combined=total_combined.set_index(['pid', 'year'], drop = False)

years = total_combined.index.get_level_values('year').to_list()
total_combined['year'] = pd.Categorical(years)

# Perform PooledOLS
from linearmodels import PooledOLS
import statsmodels.api as sm

exog = sm.tools.tools.add_constant(total_combined['ple0081_v2'])

endog = total_combined['mcs_std']
mod = PooledOLS(endog, exog)
pooledOLS_res = mod.fit(cov_type='clustered', cluster_entity=True)# Store values for checking homoskedasticity graphically
fittedvals_pooled_OLS = pooledOLS_res.predict().fitted_values
residuals_pooled_OLS = pooledOLS_res.resids

# 3A. Homoskedasticity
import matplotlib.pyplot as plt
 # 3A.1 Residuals-Plot for growing Variance Detection
fig, ax = plt.subplots()
ax.scatter(fittedvals_pooled_OLS, residuals_pooled_OLS, color = 'blue')
ax.axhline(0, color = 'r', ls = '--')
ax.set_xlabel('Predicted Values', fontsize = 15)
ax.set_ylabel('Residuals', fontsize = 15)
ax.set_title('Homoskedasticity Test', fontsize = 30)
plt.show()


# 3A.2 White-Test
from statsmodels.stats.diagnostic import het_white, het_breuschpagan
pooled_OLS_dataset = pd.concat([total_combined, residuals_pooled_OLS], axis=1)
pooled_OLS_dataset = pooled_OLS_dataset.drop(['year'], axis = 1).fillna(0)
exog = sm.tools.tools.add_constant(total_combined['ple0081_v2']).fillna(0)
white_test_results = het_white(pooled_OLS_dataset['residual'], exog)
labels = ['LM-Stat', 'LM p-val', 'F-Stat', 'F p-val'] 
print(dict(zip(labels, white_test_results)))


# 3A.3 Breusch-Pagan-Test
breusch_pagan_test_results = het_breuschpagan(pooled_OLS_dataset['residual'], exog)
labels = ['LM-Stat', 'LM p-val', 'F-Stat', 'F p-val'] 
print(dict(zip(labels, breusch_pagan_test_results)))


# 3.B Non-Autocorrelation
# Durbin-Watson-Test
from statsmodels.stats.stattools import durbin_watson

durbin_watson_test_results = durbin_watson(pooled_OLS_dataset['residual']) 
print(durbin_watson_test_results)


# Little autocorrelation, but no Homoskedasticity


# FE und RE model
from linearmodels import PanelOLS
from linearmodels import RandomEffects
exog = sm.tools.tools.add_constant(total_combined['ple0081_v2'])
endog = total_combined['mcs_std']
# random effects model
model_re = RandomEffects(endog, exog) 
re_res = model_re.fit() 
# fixed effects model
model_fe = PanelOLS(endog, exog, entity_effects = True) 
fe_res = model_fe.fit() 
#print results
print(re_res)
print(fe_res)



import numpy.linalg as la
from scipy import stats
import numpy as np
def hausman(fe, re):
 b = fe.params
 B = re.params
 v_b = fe.cov
 v_B = re.cov
 df = b[np.abs(b) < 1e8].size
 chi2 = np.dot((b - B).T, la.inv(v_b - v_B).dot(b - B)) 
 
 pval = stats.chi2.sf(chi2, df)
 return chi2, df, pval

hausman_results = hausman(fe_res, re_res) 
print('chi-Squared: ' + str(hausman_results[0]))
print('degrees of freedom: ' + str(hausman_results[1]))
print('p-Value: ' + str(hausman_results[2]))


#-> Fixed Effect model it is!
# https://towardsdatascience.com/a-guide-to-panel-data-regression-theoretics-and-implementation-with-python-4c84c5055cf8
##################################################################

f = open('results/{}_selected_results.txt'.format(variable), 'w')
f.write("StandardModel\n")


y_var_name = "mcs_std"
X_var_names = ["open", "cons", "extra", "neuro", "agree", 
               "sex", "age", "ple0081_v2", "wealth_std", "pcs_std"]
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


#Let's plot the Q-Q plot of the residual errors:
sm.qqplot(data=pooled_olsr_model_results.resid, line='45')
plt.show()

