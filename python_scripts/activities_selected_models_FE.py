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
from initial_data_loading import load_data_long, path, load_data_long_term

activities = {
    "pli0079": "Going out for dinner or drinks (café, pub, restaurant)",
    			
    "pli0080": "Visiting or being visited by neighbors, friends, or acquaintances",	
    		
    "pli0081": "Visiting or being visited by family members or relatives",
    
    "pli0181": "Keeping in touch with friends or relatives abroad (by telephone, e-mail, Internet phone, skype, etc.)",		
    
    "plh0390": "[de] Nutzen sozialer Online-Netzwerke / Chat-Dienste (z.B. Facebook / Instagram / Twitter / WhatsApp)",				
    
    "pli0082": "Going on an excursion or short trip",
    
    "pli0098_h":"Going to church, attending religious events",					
    
    "pli0083": "[de] Fernsehen / Filme, Serien oder Videos sehen (auch Mediatheken / Internet-Streams / DVD / etc.)",				
        
    "plh0391": "[de] Lesen von Büchern (auch eBooks)",		
    
    "plh0392": "[de] Lesen von (Tages-)Zeitungen (auch ePaper)",			
    
    "pli0093_h": "Artistic and musical activities (painting, music, photography, theater, dance)",
    
    "plh0393": "[de] Reparaturen am Haus, in der Wohnung oder an Fahrzeugen / Gartenarbeit / Fahrzeugpflege",	
    
    "pli0092_h":"Taking part in sports",			
    
    "pli0089": "Going to sporting events",					
    
    "pli0091_h": "Going to the cinema, pop concerts, dance events, clubs",					
		
    "pli0090_h":"Going to cultural events such as opera, classical concerts, theater, exhibitions",				
    
    "plh0394": "[de] Einfach nichts tun / abhängen / träumen"}


# pli0079 [1990 1995 1998 2003 2008 2013 2017 2018 2019]
# pli0080 [1990 1995 1998 2003 2008 2013 2019]
# pli0081 [1990 1995 1998 2003 2008 2013 2019]
# pli0181 [2008 2013 2017 2018 2019]
# plh0390 [2019]
# pli0082 [1990 1995 1998 2003 2008 2013 2017 2018 2019]
# pli0098_h [1990 1992 1994 1995 1996 1997 1998 1999 2001 2003 2005 2007 2008 2009 2011 2013 2015 2017 2018 2019]
# pli0083 [1990 1995 1998 2003 2008 2013 2019]
# plh0391 [2019]
# plh0392 [2019]
# pli0093_h [1990 1995 1998 2001 2003 2005 2007 2008 2009 2011 2013 2015 2017 2018 2019]
# plh0393 [2019]
# pli0092_h [1984 1985 1986 1988 1990 1992 1994 1995 1996 1997 1998 1999 2001 2003 2005 2007 2008 2009 2011 2013 2015 2017 2018 2019]
# pli0089 [1990 1995 1998 2003 2008 2013 2017 2018 2019]
# pli0091_h [1985 1986 1988 1990 1992 1994 1995 1996 1997 1998 1999 2001 2003 2005 2007 2008 2009 2011 2013 2015 2017 2018 2019]
# pli0090_h [1984 1985 1986 1988 1990 1992 1994 1995 1996 1997 1998 1999 2001 2003 2005 2007 2008 2009 2011 2013 2015 2017 2018 2019]
# plh0394 [2019]



# load data
workpath = "C:/Users/rapha/Documents/PanelDataProject/"
mcs_long = load_health_long()
mcs_long = mcs_long.dropna()
b5, high_low = load_b5()
b5 = pd.merge(b5, high_low)


indiv_data = load_biodata()
wealth_data = load_wealthdata()

personal_df = pd.read_stata(path + "pl.dta")

script = open('results/FE_all_activities.txt', 'w')
script.write('Included years for all activities:\n')
script.write("b5_years = [2005, 2009, 2013, 2019]\n")
script.write("mcs_years = [2006, 2010, 2014, 2020]\n")
script.write("wealth_years = [2002, 2007, 2012, 2017]\n\n\n")

total_params = pd.DataFrame()
total_pvals = pd.DataFrame()

for term in activities.keys():
    
    question = activities[term].replace("/", " ")
    script.write(term + ":   " + question + "\n\n")
    
    data = load_data_long_term(personal_df, term)
    data  = data.replace([1,2,3,4,5], [4,3,2,1,0])

    # plot 
    fig4, ax4 = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
    ax4.hist(data[term], bins  = 5)
    fig4.savefig(workpath + "figures/FE_{}.png".format(question))

    #print("B5", np.unique(b5["syear"]))
    #print("mcs", np.unique(mcs_long["syear"]))
    #print("wealth", np.unique(wealth_data["syear"]))
    print(term, np.unique(data["syear"]))
    unique_years = np.unique(data["syear"])
    script.write("All years with data: {}\n".format(unique_years))

    # initial check:
    # OLS Regression between Gender, Age and Personality traits:
    
    b5_years = [2005, 2009, 2013, 2019]
    mcs_years = [2006, 2010, 2014, 2020]
    wealth_years = [2002, 2007, 2012, 2017]
    
    if 2003 in unique_years:
        item_years = [2003, 2008, 2013, 2019]
        n_years = 4
    elif 2008 in unique_years:
        item_years = [2008, 2013, 2019]
        n_years = 3
    elif 2013 in unique_years:
        item_years = [2013, 2019]
        n_years = 2
    else:
        item_years = [2019]
        n_years = 1
    
    b5_years = b5_years[4-n_years:]
    mcs_years = mcs_years[4-n_years:]
    wealth_years = wealth_years[4-n_years:]

    script.write("Selected years with data: {}\n".format(item_years))
    
    
    # create total combined Data Frame in order to create true age for correlation
    total_combined = pd.DataFrame()
    
    for i in range(len(b5_years)):
        b5y = b5_years[i]
        mcsy = mcs_years[i]
        wy = wealth_years[i]
        iy = item_years[i]
        
        b5_year  = b5[b5["syear"] == b5y]
        
        indiv_year = indiv_data
        indiv_year.loc[:,"age"] = calc_ages(indiv_year, b5y)
        
        mcs_year = mcs_long[mcs_long["syear"] == mcsy].loc[:,["pid", "mcs_std"]]
        
        pcs_year = mcs_long[mcs_long["syear"] == mcsy].loc[:,["pid", "pcs_std"]]
        
        wealth_year = wealth_data[wealth_data["syear"] == wy].loc[:,["pid", "wealth_std"]]
        
        item_year = data[data["syear"] == iy].loc[:,["pid", term]]
        
        combined = pd.merge(b5_year, indiv_year, on=['pid','pid'])
        combined = pd.merge(combined, mcs_year, on=['pid','pid'])
        combined = pd.merge(combined, pcs_year, on=['pid','pid'])
        combined = pd.merge(combined, wealth_year, on=['pid','pid'])
        combined = pd.merge(combined, item_year, on=['pid','pid'])
    
        total_combined = total_combined.append(combined)
    
    total_combined = total_combined.sort_values("pid", axis = 0)

    total_combined["syear"] = (total_combined["syear"].astype(int))
    total_combined=total_combined.set_index(['pid', 'syear'], drop = False)
    script.write("Total number of datapoints: {}\n\n\n".format(total_combined.shape[0]))
    
    populations = ["open_low", "open_high", "open_normal",
                       "cons_low", "cons_high", "cons_normal", 
                       "extra_low", "extra_high", "extra_normal", 
                       "neuro_low", "neuro_high", "neuro_normal", 
                       "agree_low", "agree_high", "agree_normal"]

    # create result tables
    combined_results = pd.DataFrame()
    combined_results_pvals = pd.DataFrame()
    combined_results_combined = pd.DataFrame()

    f = open('results/FE_ALL_RES_{}.txt'.format(question), 'w')
    f.write("StandardModel\n")

    # Estimate Standard Model on the entire Population
    y_var_name = "mcs_std"
    X_var_names = ["open", "cons", "extra", "neuro", "agree", 
                   "sex", "age", term, "wealth_std", "pcs_std"]
    #Carve out the pooled Y
    pooled_y=total_combined[y_var_name]
    #Carve out the pooled X
    pooled_X=total_combined[X_var_names]
    #Add the placeholder for the regression intercept. When the model is fitted, the coefficient of
    # this variable is the regression model's intercept β_0.
    
    
    
    ########################### remove pooled OLS #############################
    #pooled_X = sm.add_constant(pooled_X)
    #Build the OLS model
    #pooled_olsr_model = sm.OLS(endog=pooled_y, exog=pooled_X, missing='drop')
    #Train the model and fetch the results
    #pooled_olsr_model_results = pooled_olsr_model.fit()
    #Print the results summary
    ##########################################################################
    
    from linearmodels.panel import PanelOLS
    mod = PanelOLS(pooled_y, pooled_X, entity_effects=True, check_rank=False, drop_absorbed = True)
    res = mod.fit(cov_type='clustered', cluster_entity=True)
    print(res)
    
    
    f.write('===============================================================================')
    f.write('================================= Pooled OLS ==================================')
    f.write(str(res.summary()))
    
    res = pooled_olsr_model_results.params
    res.name = "combined"
    combined_results = combined_results.append(res)
    combined_results_combined = combined_results_combined.append(res)
    
    pvals = pooled_olsr_model_results.pvalues
    pvals.name = "pval_combined"
    combined_results_pvals = combined_results_pvals.append(pvals)
    combined_results_combined = combined_results_combined.append(pvals)
    
    # Estimate Models for Subpopulations
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
    with pd.ExcelWriter(workpath + "results/FE_COM_RES_{}.xlsx".format(question)) as writer:
        combined_results.to_excel(writer, sheet_name='params')
        combined_results_pvals.to_excel(writer, sheet_name = "pvals")    
        combined_results_combined.to_excel(writer, sheet_name = "combined")  
    
    total_params[term] = combined_results[term]
    total_pvals[term] = combined_results_pvals[term]
script.close()

total_params.index = combined_results.index
total_pvals.index = combined_results_pvals.index
with pd.ExcelWriter(workpath + "results/FE_all_activities_params.xlsx") as writer:
     total_params.to_excel(writer, sheet_name='params')
     total_pvals.to_excel(writer, sheet_name = "pvals")    
