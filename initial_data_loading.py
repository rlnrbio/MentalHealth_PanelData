# -*- coding: utf-8 -*-
"""
Created on Wed May  4 23:45:48 2022

@author: rapha
"""
import pandas as pd
import numpy as np
import re

import seaborn as sns

from sklearn import preprocessing
import matplotlib.pyplot as plt


path = "E:/SOEP_DATA/Stata/"
rawpath = "E:/SOEP_DATA/Stata/raw/"

workpath = "C:/Users/rapha/Documents/PanelDataProject/"

######### Data recode ##########
# -1 No answer / don’t know
# -2 Does not apply
# -3 Implausible value
# -4 Inadmissable multiple response
# -5 Not included in this version of the questionnaire
# -6 Version of questionnaire with modified filtering
# -7 Only available in less restricted edition
# -8 Question not part of the survey program this year¹

all_missing = [-1,-2,-3,-4,-5,-6,-7,-8]

years_of_data = 37
years_included = [2002,2004,2006,2008,2010,2012,2014,2016,2018,2020]
b5_heads = ['open', 'cons', 'extra', 'neuro', 'agree']



def variable_widener(dataset, variable, dim1 = "syear", dim2 = "pid"):
    df_seg = dataset[[dim1, dim2, variable]]
    wide = df_seg.pivot(index = dim1, columns = dim2, values = variable)
    return wide

def drop_missing_survey_years(dataset):
    # only keeps rows where noone is marked with -8
    missing = (dataset == -8).sum(axis = 1)
    to_keep = missing==0
    index = dataset.index[to_keep]
    reduced_dataset = dataset.loc[index,:]
    return reduced_dataset

def drop_all_missing_individuals(dataset, mode = "both"):
    # drops rows where data is missing for all individuals
    dataset2 = dataset.fillna(0)
    missing = (dataset2 == 0).sum(axis = 0)
    to_keep = missing!=max(missing)
    columns = dataset.columns[to_keep]
    reduced_dataset = dataset.loc[:,columns]
    return reduced_dataset

def str_delabelling(string):
    if type(string) == str:
        missing = int(re.match("\[(.*?)\]", string).group(1))
        return missing
    if type(string) == float or type(string) == int:
        return string
    
def filter_pl(dataset, variable):
    f = dataset[variable]
    f = f.map(str_delabelling)
    f = f.replace(all_missing, 0)
    dataset_filtered = dataset[f != 0]
    
    return dataset_filtered

def calc_ages(indiv_data, year):
    geb_jahr = indiv_data["gebjahr"]
    geb_jahr = geb_jahr.replace(all_missing, np.nan)
    age = year-geb_jahr
    dead = indiv_data["todjahr"] >= 0 
    died = indiv_data["todjahr"] <= year
    already_dead = dead & died
    age[already_dead] = -1
    return age
    


############# mcs data ##################
def load_health_long():
    health_df = pd.read_stata(path + 'health.dta')
    scale= preprocessing.MinMaxScaler()
    mcs_long = health_df.loc[:,["pid","syear", "mcs", "pcs"]]
    mcs_long = mcs_long.replace(all_missing, np.nan)
    mcs_long.loc[:,"mcs_std"] = scale.fit_transform(
        np.array(mcs_long.loc[:,"mcs"]).reshape(-1, 1))
    mcs_long.loc[:,"pcs_std"] = scale.fit_transform(
        np.array(mcs_long.loc[:,"pcs"]).reshape(-1, 1))
    
    return mcs_long

def load_health_wide():
    health_df = pd.read_stata(path + 'health.dta')

    mcs_wide = variable_widener(health_df, "mcs")
    mcs_wide = drop_missing_survey_years(mcs_wide)
    
    # drop years from migration dataset
    mcs_wide = mcs_wide.loc[years_included, :]
    
    mcs_wide = mcs_wide.replace(all_missing, np.nan)
    mcs_wide = drop_all_missing_individuals(mcs_wide)
    return mcs_wide



# mcs_sample = mcs_wide.iloc[:,0:500]
# mcs_sample["year"] = mcs_sample.index
# mcs_wide["year"] = mcs_wide.index

# sns.lineplot(x = "year", y = 'value', hue = 'pid', 
#              data = pd.melt(mcs_sample, ['year']), ci = "sd")

# sns.lineplot(x = "year", y = 'value', 
#              data = pd.melt(mcs_wide, ['year']), ci = "sd")

# # mcs_wide = mcs_wide.replace(all_missing, 0).fillna(np.nan)


# birth_df = pd.read_stata(path + 'biobirth.dta')  
# birth  = birth_df[["pid", "gebjahr"]]

# younger25 = list(birth[birth["gebjahr"] >= 1997]["pid"])
# younger25 = list(map(int, younger25))

# personal_df = pd.read_stata(path + "pl.dta")
# personal_df_cols = personal_df.columns



def load_b5(plot_save = False):
    b5_heads = ['open', 'cons', 'extra', 'neuro', 'agree']

    # load big 5 data:
    scale= preprocessing.MinMaxScaler()
    b5 = pd.read_stata(workpath + "bf5-soep.dta")        
    df_counts = pd.DataFrame()

    if plot_save == True:
        mean = np.mean(b5, axis = 0)
        std = np.std(b5, axis = 0)
        frame = pd.DataFrame([mean, std], index = ["Mean", "Std"])
        frame.to_csv(workpath + "mean_b5.csv")
        
    b5_std = b5
    b5_std.loc[:,b5_heads] = scale.fit_transform(b5.loc[:,b5_heads])
    high_low = b5.loc[:,["pid", "syear"]]
    if plot_save == True:
        fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
    
    for head in b5_heads:
        print("Variable: {}".format(head))
        wide = variable_widener(b5, head)
        
        mean = np.mean(wide, axis = 0)
        std = np.std(wide, axis = 0)
        frame = pd.DataFrame([mean, std], index = ["Mean", "Std"])
        frame.to_csv(workpath + "mean_indiv_{}.csv".format(head))
        
        wide_std = variable_widener(b5_std, head)
        
        low = np.nanpercentile(wide, 20)
        high = np.nanpercentile(wide, 80)
        
        
        
        ############################################################
        print("{} Distributions".format(head))
        df_counts = df_counts.append(pd.Series([sum(b5[head]<low), sum((b5[head]>high)), 
                                    sum(((~(b5[head]<low))&(~(b5[head]>high))))], name = head))
        high_low.loc[:,"{}_low".format(head)] = (b5[head]<low).astype(int)
        high_low.loc[:,"{}_high".format(head)] = (b5[head]>high).astype(int)
        high_low.loc[:,"{}_normal".format(head)] = ((~(b5[head]<low))&(~(b5[head]>high))).astype(int)
        
        print("Elements low: {}, Elements high: {}")
        
        ############################################################
        if plot_save == True:

            ax.hist(b5_std[head], bins = 19)
            
            
            fig2, ax2 = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
            ax2.hist(b5_std[head], bins = 19)
            fig2.savefig(workpath + "figures/hist_{}.png".format(head))
    if plot_save == True:
        df_counts.to_excel(workpath + "counts_b5.xlsx")
        fig.savefig(workpath + "figures/hist_combined.png")
        high_low.to_csv(workpath + "high_low_b5.csv")
    return b5, high_low




def load_data_long(personal_data, datatype):
    if datatype == "internet":
        term = "pli0068"
    if datatype == "socialmedia":
        term = "plh0390"
    if datatype == "tv":
        term = "pli0083"
    if datatype == "smoking":
        term = "ple0081_v2"
    if datatype == "popculture":
        term = "pli0091_h"
    if datatype == "classic":
        term = "pli0090_h"
    if datatype == "church":
        term = "pli0098_h"
    if datatype == "phonecontact":
        term = "pli00"
    to_keep = ["pid", "syear", term]
    pdf_filtered = filter_pl(personal_data, term)
    pdf_filtered = pdf_filtered.loc[:, to_keep]
    pdf_filtered = pdf_filtered.applymap(str_delabelling)
    
    return pdf_filtered 

def load_data_long_term(personal_data, term):
    to_keep = ["pid", "syear", term]
    pdf_filtered = filter_pl(personal_data, term)
    pdf_filtered = pdf_filtered.loc[:, to_keep]
    pdf_filtered = pdf_filtered.applymap(str_delabelling)
    
    return pdf_filtered 

    
def load_alc():
    alc = pd.read_stata(workpath + "alcoholScale.dta")  
    #alc = variable_widener(alc, "alcohol")
    alc = alc.replace(all_missing, np.nan)
    return alc


def load_biodata():
    # personal data
    to_keep = ["pid", "sex", "gebjahr", "todjahr"]
    indiv_data = pd.read_stata(path + "ppath.dta")  
    indiv_data = indiv_data.loc[:, to_keep]
    
    indiv_data.loc[:,"sex"] = indiv_data.loc[:,"sex"].apply(
        str_delabelling).replace([1,2], [0,1])
    # 2 is female, 1 is male -> 1 is female, 0 is male
    # gender = pd.DataFrame(indiv_data.loc[:,"sex"], index = indiv_data.loc[:,"pid"] )
    # gender = gender.applymap(str_delabelling)
    return indiv_data
    
    
    
def load_wealthdata(plot = False):    
    # wealth data
    scale= preprocessing.MinMaxScaler()
    wealth_data = pd.read_stata(path + "pwealth.dta")  
    # using the netto wealth (gross wealth - debt)
    to_keep = [	"pid", "syear", 
               "w0111a"]
    
    wealth_data = wealth_data.loc[:, to_keep]
    if plot == False:
        fig3, ax3 = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
        ax3.hist(wealth_data["w0111a"], bins = 50)
    
    wealth_data = wealth_data[(wealth_data["w0111a"]<=500000)]
    wealth_data = wealth_data[(wealth_data["w0111a"]>=-100000)]
    
    wealth_data.loc[:,"wealth_std"] = scale.fit_transform(np.array(wealth_data.loc[:,"w0111a"]).reshape(-1, 1))
    
    if plot == False:
        fig4, ax4 = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
        ax4.hist(wealth_data["w0111a"], bins  = 50)
        fig4.savefig(workpath + "figures/wealth.png")
    return wealth_data
    
    
    

def main():

    personal_df = pd.read_stata(path + "pl.dta")
    
    
    mcs_long = load_health_long()
    mcs_wide = load_health_wide()
    b5, high_low = load_b5()
    
    # personal usage of internet: 2001 - pli0064
    internet_use = load_data_long(personal_df, "internet")
    
    # social media usage: 2019 - plh0390
    social_media = load_data_long(personal_df, "socialmedia")
    
    # Watch Television, Video  pli0083
    tv_usage = load_data_long(personal_df, "tv")
    
    
    # frequency of sport or excercise: 
    #sports = pd.read_stata(path)
        
    
    
    # big5_indivs = list(wide.columns)
    # mcs_indivs = list(mcs_wide.columns)
    
    # # get intersection between individuals
    # intersect = list(set(big5_indivs) & set(mcs_indivs))
    
    
    
    # data to use:
    ##########################################
    
    # Watch Television, Video  pli0083
    # tv_years = [2003,2008,2013,2019]
    # pdf_filtered = filter_pl(personal_df, "pli0083")
    # watchTV = variable_widener(pdf_filtered, "pli0083")
    # watchTV = watchTV.loc[tv_years, :]
    # print(watchTV.loc[2019,:].unique())
    
    # watchTV = watchTV.applymap(str_delabelling)
    # watchTV = drop_all_missing_individuals(watchTV)
    
    
    # Beer consumption - ple0090
    # wine, champagne  - ple0091
    # spirits consumpt - ple0092
    # mixed drinks     - ple0093
    # Creation of AlcoholScale - see do file
    # Data from 2006, 2008, 2010
    alc = load_alc()
    
    fig2, ax2 = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
    ax2.hist(alc["alcohol"], bins = 13)
    fig2.savefig(workpath + "figures/alcohol.png")
    
    
    # Do you smoke - ple0081
    # Year 2012 is missing for some reason
    smoking_years = [2004, 2006, 2008, 2010, 2014, 2016, 2018, 2020]
    
    smoke = load_data_long(personal_df, "smoking")
    
    
    # # Age when started smoking - ple0082
    # # Years 2002, 2012
    # smoking_years = [2002, 2012]
    
    # #pdf_filtered = filter_pl(personal_df, "ple0082")
    # smokeYear = variable_widener(personal_df, "ple0082")
    # smokeYear = smokeYear.loc[smoking_years, :]
    # smokeYear = smokeYear.applymap(str_delabelling)
    # smokeYear = drop_all_missing_individuals(smokeYear)
    # smokeYear = drop_all_missing_individuals(smokeYear)
    
    indiv_data = load_biodata()

    # wealth is provided in 2002, 2007, 2012, 2017
    wealth_data = load_wealthdata()
    
    print(np.unique(wealth_data["syear"]))

    # only keep wealth data between -100,000 and 500,000
    # This looses 126, 1612 datapoints respectively

    import scipy.stats as st
    import statsmodels.api as sm
    import statsmodels.graphics.tsaplots as tsap
    from statsmodels.compat import lzip
    from statsmodels.stats.diagnostic import het_white
    import seaborn as sns

    ############################################################
    # Now use extreme personalities for prediction
    
    b5_years = [2005,2009,2013,2017,2019]
    mcs_years = [2006, 2010, 2014, 2018, 2020]
    wealth_years = [2007, 2012, 2012, 2017, 2017]
    total_combined = pd.DataFrame()
    
    total_combined = pd.DataFrame()
    
    for i in range(len(b5_years)):
        b5y = b5_years[i]
        mcsy = mcs_years[i]
        wy = wealth_years[i]
    
        b5_year  = high_low[high_low["syear"] == b5y]
        
        indiv_year = indiv_data
        indiv_year.loc[:,"age"] = calc_ages(indiv_year, b5y)
        
        mcs_year = mcs_long[mcs_long["syear"] == mcsy].loc[:,["pid", "mcs_std"]]
        
        wealth_year = wealth_data[wealth_data["syear"] == wy].loc[:,["pid", "wealth_std"]]
        
        combined = pd.merge(b5_year, indiv_year, on=['pid','pid'])
        combined = pd.merge(combined, mcs_year, on=['pid','pid'])
        combined = pd.merge(combined, wealth_year, on=['pid','pid'])
        
        total_combined = total_combined.append(combined)
    
    
    
    
    y_var_name = "mcs_std"
    X_var_names = ["open_low", "open_high", "open_normal",
                   "cons_low", "cons_high", "cons_normal", 
                   "extra_low", "extra_high", "extra_normal", 
                   "neuro_low", "neuro_high", "neuro_normal", 
                   "agree_low", "agree_high", "agree_normal", 
                   "sex", "age", "wealth_std"]
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
    print('===============================================================================')
    print('================================= Pooled OLS ==================================')
    print(pooled_olsr_model_results.summary())
        
    #Let's plot the Q-Q plot of the residual errors:
    sm.qqplot(data=pooled_olsr_model_results.resid, line='45')
    plt.show()



# num_rows = 20
# years = list(range(1990, 1990 + num_rows))
# data_preproc = pd.DataFrame({
#     'Year': years, 
#     'A': np.random.randn(num_rows).cumsum(),
#     'B': np.random.randn(num_rows).cumsum(),
#     'C': np.random.randn(num_rows).cumsum(),
#     'D': np.random.randn(num_rows).cumsum()})
# sns.lineplot(x='Year', y='value', hue='variable', 
#              data=pd.melt(data_preproc, ['Year']))
  