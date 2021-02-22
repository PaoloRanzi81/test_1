"""
TITLE: "Feature importance/forest plot for the hierachical bayesian model"
AUTHOR: Paolo Ranzi 
PYTHON VERSION: 3.6.9

DESCRIPTION: 
Please change the following sections according to your individual input preferences:
    - '2. PARAMETERS TO BE SET!!!'

"""


###############################################################################
## 1. IMPORTING LIBRARIES
# import required Python libraries
import platform
import os
import numpy as np
import pandas as pd
import pymc3 as pm
import theano
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns 
import arviz as az
from itertools import chain
from mdutils.mdutils import MdUtils
import time
import requests
import xarray as xr

   
###############################################################################
## 2. PARAMETERS TO BE SET!!!
# check the machine you are using 
RELEASE = platform.release()

# set the correct pathways/folders
BASE_DIR_INPUT = ('.')
BASE_DIR_OUTPUT = BASE_DIR_INPUT
INPUT_FOLDER = ('./..')

# skip notifications and set correct pathways in case of testing on local
# computer
# LOCAL_COMPUTER = '5.4.0-1035-aws'
LOCAL_COMPUTER ='4.19.0-14-cloud-amd64' 
# LOCAL_COMPUTER = '5.4.0-62-generic'


# # TEST:
# if RELEASE == LOCAL_COMPUTER: # Linux laptop
#     BASE_DIR_INPUT = ('/home/paolo/Dropbox/Univ/scripts/python_scripts/tensorflow/wright_keith/403_bayesian_hbm_20201220')
#     BASE_DIR_OUTPUT = BASE_DIR_INPUT
#     INPUT_FOLDER = BASE_DIR_INPUT


# set input/output file names
input_file_name_01 = ('input/user_activities_data_set_train.csv')
input_file_name_03 = ('input/dictionary_variants_id.csv')
input_file_name_07 = ('input/config/url_for_pushing_notification.csv')
output_file_name_02 = ('output/403_analysis_saved_trace')
output_file_name_11 = ('output/403_analysis_feature_name_list.csv')
output_file_name_13 = ('output/403_feat_imp_interpretation.md')
output_file_name_15 = ('output/403_feature_importance.pdf')
output_file_name_19 = ('output/403_forest_plot_arviz_original.pdf') 
output_file_name_21 = ('output/403_analysis_trace_model.joblib')
output_file_name_22 = ('output/403_analysis_trace.h5')
output_file_name_23 = ('output/403_bayesian_summary_statistics.csv')
output_file_name_24 = ('output/403_feature_importance_boxplot.csv')
output_file_name_25 = ('output/403_feature_importance_raw_data.csv')
output_file_name_26 = ('output/403_feature_importance_absolute_value.csv') 
output_file_name_28 = ('output/403_feature_importance_boxplot.pdf')  
output_file_name_31 = ('output/403_posterior_mcmc_traces_cont') 
output_file_name_32 = ('output/403_posterior_mcmc_traces_cat') 


###############################################################################
## 3. LOADING DATA-SET 
# setting testing mode 
RELEASE = platform.release()

# start clocking time
start_time = time.time()

# loading the .csv files with raw data 
user_activities = pd.read_csv(os.path.sep.join([INPUT_FOLDER, 
                                         input_file_name_01]), header = 0)
    
variant_df = pd.read_csv(os.path.sep.join([INPUT_FOLDER ,
                                         input_file_name_03]), header = 0)

feature_name_list = pd.read_csv(os.path.sep.join([BASE_DIR_INPUT, 
                                         output_file_name_11]), header = 0)

url = pd.read_csv(os.path.sep.join([INPUT_FOLDER, 
                                         input_file_name_07]), header = None, 
                                     dtype = 'str')

# load ArviZ NetCDF data-set containig MCMC samples
arviz_inference = az.from_netcdf(filename=os.path.sep.join([BASE_DIR_OUTPUT, output_file_name_22]))


################################################################################
## 4. PRE-PROCESSING
# save  folder name since it contains the model version (e.g. "906_logistic_regression_20200731")
folder_name = pd.Series(os.getcwd()).rename('model_version_in_the_folder_name')

# # TEST: you do not need it since it has been already done at the training step
# #  save one-hot-encoded column's names as .csv file
# folder_name.to_csv(os.path.sep.join([BASE_DIR_OUTPUT, 
#                                      output_file_name_27]), index = False)

# drop rows with NaN
user_activities.dropna(axis = 0, inplace = True)

# drop duplicates
user_activities.drop_duplicates(inplace = True)

# sort 'variant_description' in alphabetical order
variant_df.sort_values(by = 'variant_description', 
                       axis = 0, 
                       inplace = True, 
                       ascending = True)

# re-set index
variant_df.reset_index(drop = True, inplace = True)

# build a dictionary in order to replace video_id with more meaningful names
replace_variant_name = dict(zip(variant_df.loc[:, 'variant_id'], 
                                variant_df.loc[:, 'variant_description']))

# convert video_id to more meaningful names
user_activities.loc[:,'variant_id'].replace(to_replace = replace_variant_name, 
            inplace = True)
user_activities.loc[:,'best_video_selected'].replace(to_replace = replace_variant_name, 
            inplace = True)

# re-set index
user_activities.reset_index(drop = True, inplace = True)

# convert user_id to integers
user_id_integer_tmp = user_activities.loc[:, 'user_id'].astype(int)

# rename column 
user_id_integer= user_id_integer_tmp.rename('user_id_int') 

# concatenate 
user_activities_concat = pd.concat([user_activities, user_id_integer], axis = 1)

# convert UTC time-stamps to Unix time stamps
unix_timestamp = pd.DataFrame()
for index_01 in range(0, user_activities_concat.shape[0]):
    unix_timestamp_tmp = pd.Series(pd.Timestamp(user_activities_concat.loc[index_01, 'datetime']).timestamp())
    
    # append
    unix_timestamp = unix_timestamp.append(unix_timestamp_tmp, 
                                       ignore_index = True, sort = False)

# combine
user_activities_concat_02 = pd.concat([user_activities_concat, unix_timestamp], axis = 1)

# rename column
user_activities_concat_02.rename(columns = {0: 'timestamp'}, inplace = True)

# convert timestamps to integers
timestamp_integers = user_activities_concat_02.loc[:, 'timestamp'].astype(int)

# drop redundant columns
user_activities_concat_02.drop(labels = ['timestamp'], inplace = True, axis = 1)

# combine
user_activities_concat_03 = pd.concat([user_activities_concat_02, timestamp_integers], axis = 1)

# initialize the encoding of labels/strings into integers
le = LabelEncoder()

# rename column
user_activities_concat_03.rename(columns = {'best_video_selected': 'best_video'}, 
                inplace = True)

# deep copy
user_activities_concat_04 = user_activities_concat_03.copy()

# drop duplicates
user_activities_concat_04.drop_duplicates(inplace = True)

# drop redundant columns
user_activities_concat_04.drop(labels = ['datetime', 
                        'user_id',
                        'experiment_id', 
                        'ad_campaign_id',
                        'property_name',                         
                        'experiment_title', 
                        # 'experiment_lift_type', # no more needed as for 20201012
                        'country_code'], 
                         inplace = True,
                         axis = 1)

# rename column
user_activities_concat_04.rename(columns = {'session_id': 'session_id_original', 
                           'user_id_int': 'user_id_original',
                           'variant_id': 'variant_id_original',
                           'best_video': 'best_video_original',
                           'browser': 'browser_original', 
                           'experiment_audience_type': 'experiment_audience_type_original', 
                           'city': 'city_original', 
                           'device_type': 'device_type_original'}, 
                            inplace = True)

# deep copy column 
user_activities_concat_04['selection_order_original'] = user_activities_concat_04.loc[:, 'selection_order'].copy()

# build a dictionary of integers (0, 1, 2...)which should be replaced 
# by strings (first, second, third...).
# Going higher than 10 videos looked weird to me! Anyway, here I have put the 
# list up to 15th just to accomodate the craziest scenarios!  
replace_order_by_strings = {0: '01first', 
                1: '02second', 
                2: '03third', 
                3: '04fourth',
                4: '05fifth',
                5: '06sixth',
                6: '07seventh',
                7: '08eighth',
                8: '09ninth',
                9: '10tenth',
                10: '11eleventh',
                11: '12twelfth',
                12: '13thirteenth',
                13: '14fourteenth',
                14: '15fifteenth'}

# convert integers for 'event_action' into percentages
user_activities_concat_04.loc[:,'selection_order_original'].replace(to_replace = replace_order_by_strings, 
            inplace = True)

# label encoding
user_activities_concat_04.loc[:,'session_id'] = le.fit_transform(user_activities_concat_04.loc[:,'session_id_original'])
user_activities_concat_04.loc[:,'user_id'] = le.fit_transform(user_activities_concat_04.loc[:,'user_id_original'])
user_activities_concat_04.loc[:,'variant_id'] = le.fit_transform(user_activities_concat_04.loc[:,'variant_id_original'])
user_activities_concat_04.loc[:,'best_video'] = le.fit_transform(user_activities_concat_04.loc[:,'best_video_original'])
user_activities_concat_04.loc[:,'browser'] = le.fit_transform(user_activities_concat_04.loc[:,'browser_original'])
user_activities_concat_04.loc[:,'experiment_audience_type'] = le.fit_transform(user_activities_concat_04.loc[:,'experiment_audience_type_original'])
user_activities_concat_04.loc[:,'city'] = le.fit_transform(user_activities_concat_04.loc[:,'city_original'])
user_activities_concat_04.loc[:,'device_type'] = le.fit_transform(user_activities_concat_04.loc[:,'device_type_original'])

# concatenate
user_activities_concat_05 = pd.concat([user_activities_concat_04.loc[:, 'best_video'],
                                       user_activities_concat_04.loc[:, 'percentage_watched'],
                                       user_activities_concat_04.loc[:, 'selection_order'],
                                       user_activities_concat_04.loc[:, 'experiment_audience_type'],
                                       user_activities_concat_04.loc[:, 'browser'],
                                       user_activities_concat_04.loc[:, 'city'], 
                                       user_activities_concat_04.loc[:, 'device_type']], axis = 1)

# legacy code in case the hierarchical model should be reused
hierarchical_variable = []

# build output (dependent variable) to be fed into statistical analysis
y = user_activities_concat_05.pop('best_video')

# selected_variables
selected_variables = pd.Series(user_activities_concat_05.columns)

# build features (independent variable/s) to be fed into statistical analysis
features_to_be_fed = user_activities_concat_05.loc[:, selected_variables]

# build Pandas Series 
selected_variables_series = pd.Series(selected_variables)

# delete 'hierarchical_variable' from original list, in order to build
# the variables_to_be_used
variables_to_be_used = selected_variables_series.copy()

# re-build a new DataFrame without hierachical variable
X = features_to_be_fed.loc[:, variables_to_be_used]


###############################################################################
## 5. MCMC'S CHAINS SUMMARY STATISTICS
# bayesian summary statistics
summary_statistics_trace = az.summary(arviz_inference)

# re-set index
summary_statistics_trace.reset_index(drop=False,
                                          inplace=True)

# rename column
summary_statistics_trace.rename(columns={'index': 'priors'},
                                    inplace=True)

# save summary statistics as .csv file
summary_statistics_trace.to_csv(os.path.sep.join([BASE_DIR_OUTPUT,
                                        output_file_name_23]),
                                    index=False)

# # TEST: loading custom summary statistics. Useful when de-bugging.
# summary_statistics_trace = pd.read_csv(os.path.sep.join([BASE_DIR_INPUT, 
#                                           output_file_name_23]), header = 0)


###########################################################################
## 6. QUICK DATA MANIPULATION: PREPARING TO THE AUTOMATED INTERPRETATION
# build different masks
mask_cont = summary_statistics_trace.loc[:, 'priors'].str.contains('beta_con_tmp_percentage')
mask_cat_selection = summary_statistics_trace.loc[:, 'priors'].str.contains('beta_cat_tmp_selection')
mask_cat_audience = summary_statistics_trace.loc[:, 'priors'].str.contains('beta_cat_tmp_audience')
mask_cat_browser = summary_statistics_trace.loc[:, 'priors'].str.contains('beta_cat_tmp_browser')
mask_cat_city = summary_statistics_trace.loc[:, 'priors'].str.contains('beta_cat_tmp_city')
mask_cat_device = summary_statistics_trace.loc[:, 'priors'].str.contains('beta_cat_tmp_device')

# concatenate masks
mask_subset_concat = pd.concat([mask_cont,
                                mask_cat_selection, 
                                mask_cat_audience, 
                                mask_cat_browser, 
                                mask_cat_city, 
                                mask_cat_device], axis = 1)

# sum the booleans (indeed, each mask is a boolean True/False)
mask_subset = mask_subset_concat.sum(axis = 1).astype('bool')

# select only the priors for the "slopes" ("beta_tmp...") of the logistic regression
feature_importance_tmp_01 = summary_statistics_trace.loc[mask_subset, :].reset_index(drop = True)

# select specific columns
feature_importance_tmp_02 = feature_importance_tmp_01.loc[:, ['priors','mean']]

# build Series with the variable's name specific for each specific prior for
# the logistic  regression's slope (it is the first digit within the square brackets of the prior.
# E.g. in "beta_tmp[0,1]" 0 means 'event_action' -thus the variable's name-)
dummy_series_variable = feature_name_list.loc[:, 'feature_name'].repeat((variant_df.shape[0] - 1)).reset_index(drop=True).rename('variable')
dummy_series_first_level = feature_name_list.loc[:, 'first_level_var'].repeat((variant_df.shape[0] - 1)).reset_index(drop=True).rename('first_level_var')
dummy_series_second_level = feature_name_list.loc[:, 'second_level_var'].repeat((variant_df.shape[0] - 1)).reset_index(drop=True).rename('second_level_var')

# deep copy
event_label_types = variant_df.loc[: , 'variant_description'].copy()

# re-set index
deleting_control = pd.Series(event_label_types.iloc[1:]).reset_index(drop=True)

# initialize variables 
dummy_series_video_id_df = pd.DataFrame()
dummy_series_video_id_tmp = []

# build Series with the variant's name specific for each specific prior for 
# the logistic  regression's slope (it is the second digit within the square brackets of the prior.
# E.g. in "beta_tmp[0,1]" 1 means 'variant_01' -thus the variant's name-)
for index_03 in range(0, int(feature_importance_tmp_02.shape[0]/(variant_df.shape[0] - 1))):   
    
    # append
    dummy_series_video_id_tmp.append(deleting_control)
    
    # concatenate Series to Series (I know it is weird... but it works!)
    dummy_series_video_id_df = pd.concat(dummy_series_video_id_tmp)
    
# re-set index
dummy_series_video_id = dummy_series_video_id_df.reset_index(drop=True)

# concatenate horizontally
feature_importance_raw = pd.concat([dummy_series_variable, 
                                    dummy_series_video_id, 
                                    feature_importance_tmp_02], axis=1)

# rename columns
feature_importance_raw.rename(columns = { 'mean' : 'actual_coefficient', 
                                     'variant_description' : 'target_variable'}, inplace = True)

# combine the strings of 2 columns into 1 column
feature_importance_raw.loc[:,'var_name_forest'] = feature_importance_raw.loc[:, 'variable'].str.cat(feature_importance_raw.loc[:, 'target_variable'], sep = '_')

# create column by computing the absolute value of the actual coefficients
feature_importance_raw.loc[:,'coef_sub_value'] = feature_importance_raw.loc[:, 'actual_coefficient'].abs()

# re-scale coefficients: now the range is from 0 to 1
feature_importance_raw.loc[:, 'coef_abs_value'] = MinMaxScaler().fit_transform(feature_importance_raw.loc[:, 'coef_sub_value'].to_numpy().reshape(-1, 1))

# change order of columns
feature_importance_tmp_03 = feature_importance_raw.loc[:, ['priors', 
                    'var_name_forest', 
                    'actual_coefficient', 
                    'variable', 
                    'target_variable', 
                    'coef_abs_value']]


########################################################################################################################
## TEST: Arviz diagnostics
#set shared theano variables
y_data = y.to_numpy('int8')
X_continuos = X.loc[:, variables_to_be_used[0]]
X_categorical_selection = X.loc[:, variables_to_be_used[1]]
X_categorical_audience = X.loc[:, variables_to_be_used[2]]
X_categorical_browser = X.loc[:, variables_to_be_used[3]]
X_categorical_city = X.loc[:, variables_to_be_used[4]]
X_categorical_device = X.loc[:, variables_to_be_used[5]]

# build coordinates for ARviz (according to 20200629's tutorial): 
# https://docs.pymc.io/notebooks/multilevel_modeling.html
coords = {
          "X_continuos_column_name": ('percentage_watched'),
          "X_continuos_index": X_continuos.index,
          "X_categorical_selection_column_name": ('selection_order'),
          "X_categorical_selection_index": X_categorical_selection.index,
          "X_categorical_audience_column_name": ('experiment_audience_type'),
          "X_categorical_audience_index": X_categorical_audience.index,
          "X_categorical_browser_column_name": ('browser'),
          "X_categorical_browser_index": X_categorical_browser.index,
          "X_categorical_city_column_name": ('city'),          
          "X_categorical_city_index": X_categorical_city.index,
          "X_categorical_device_column_name": ('device_type'),          
          "X_categorical_device_index": X_categorical_device.index,
          #"X_categorical_column_name": list(X_categorical_city.columns),
          }

# diagnostics of posterior + MCMC traces (continuos)
az.plot_trace(data=arviz_inference,
                    var_names = 'beta_con_tmp_percentage',
                    divergences = 'bottom');
plt.ylabel('variables/features',
            fontsize = 8)
plt.xlabel('posterior + MCMC traces for {}'.format('_continuos'),
            fontsize = 8)

# save plot as .pdf file
plt.savefig(os.path.sep.join([BASE_DIR_OUTPUT, ('{}_{}.pdf'.format(
      output_file_name_31, 
      'continuos'))]))

# close pic in order to avoid overwriting with previous pics
plt.clf()

# loop thorough all variables: diagnostics of posterior + MCMC traces (categorical_selection)
for index_10, variab_name_10 in enumerate(pd.unique(X_categorical_selection.sort_values(kind='mergesort'))):

    az.plot_trace(data=arviz_inference,
                        var_names = 'beta_cat_tmp_selection',
                        coords = {'beta_cat_tmp_selection_dim_0': [index_10]},
                      divergences = 'bottom'); 
    plt.ylabel('variables/features',
                fontsize = 8)
    plt.xlabel('posterior + MCMC traces for {}'.format(variab_name_10),
                fontsize = 8)
    
    # save plot as .pdf file
    plt.savefig(os.path.sep.join([BASE_DIR_OUTPUT, ('{}_selection_{}.pdf'.format(
          output_file_name_32,
          variab_name_10))]))

# close pic in order to avoid overwriting with previous pics
plt.clf()


# loop thorough all variables: diagnostics of posterior + MCMC traces (categorical_audience)
for index_11, variab_name_11 in enumerate(pd.unique(X_categorical_audience.sort_values(kind='mergesort'))):

    az.plot_trace(data=arviz_inference,
                        var_names = 'beta_cat_tmp_audience',
                        coords = {'beta_cat_tmp_audience_dim_0': [index_11]},
                      divergences = 'bottom'); 
    plt.ylabel('variables/features',
                fontsize = 8)
    plt.xlabel('posterior + MCMC traces for {}'.format(variab_name_11),
                fontsize = 8)
    
    # save plot as .pdf file
    plt.savefig(os.path.sep.join([BASE_DIR_OUTPUT, ('{}_audience_{}.pdf'.format(
          output_file_name_32,
          variab_name_11))]))

# close pic in order to avoid overwriting with previous pics
plt.clf()

# loop thorough all variables: diagnostics of posterior + MCMC traces (categorical_browser)
for index_12, variab_name_12 in enumerate(pd.unique(X_categorical_browser.sort_values(kind='mergesort'))):

    az.plot_trace(data=arviz_inference,
                        var_names = 'beta_cat_tmp_browser',
                        coords = {'beta_cat_tmp_browser_dim_0': [index_12]},
                      divergences = 'bottom'); 
    plt.ylabel('variables/features',
                fontsize = 8)
    plt.xlabel('posterior + MCMC traces for {}'.format(variab_name_12),
                fontsize = 8)
    
    # save plot as .pdf file
    plt.savefig(os.path.sep.join([BASE_DIR_OUTPUT, ('{}_browser_{}.pdf'.format(
          output_file_name_32,
          variab_name_12))]))

# close pic in order to avoid overwriting with previous pics
plt.clf()

# loop thorough all variables: diagnostics of posterior + MCMC traces (categorical_city)
for index_13, variab_name_13 in enumerate(pd.unique(X_categorical_city.sort_values(kind='mergesort'))):

    az.plot_trace(data=arviz_inference,
                        var_names = 'beta_cat_tmp_city',
                        coords = {'beta_cat_tmp_city_dim_0': [index_13]},
                      divergences = 'bottom'); 
    plt.ylabel('variables/features',
                fontsize = 8)
    plt.xlabel('posterior + MCMC traces for {}'.format(variab_name_13),
                fontsize = 8)
    
    # save plot as .pdf file
    plt.savefig(os.path.sep.join([BASE_DIR_OUTPUT, ('{}_city_{}.pdf'.format(
          output_file_name_32,
          variab_name_13))]))

# close pic in order to avoid overwriting with previous pics
plt.clf()

# loop thorough all variables: diagnostics of posterior + MCMC traces (categorical_device)
for index_14, variab_name_14 in enumerate(pd.unique(X_categorical_device.sort_values(kind='mergesort'))):

    az.plot_trace(data=arviz_inference,
                        var_names = 'beta_cat_tmp_device',
                        coords = {'beta_cat_tmp_device_dim_0': [index_14]},
                      divergences = 'bottom'); 
    plt.ylabel('variables/features',
                fontsize = 8)
    plt.xlabel('posterior + MCMC traces for {}'.format(variab_name_14),
                fontsize = 8)
    
    # save plot as .pdf file
    plt.savefig(os.path.sep.join([BASE_DIR_OUTPUT, ('{}_device_{}.pdf'.format(
          output_file_name_32,
          variab_name_14))]))

# close pic in order to avoid overwriting with previous pics
plt.clf()


########################################################################################################################

# concatenate horizontally
feature_importance = pd.concat([feature_importance_tmp_03,
                               dummy_series_first_level,
                               dummy_series_second_level], 
                              axis = 1, ignore_index = False, 
                              sort = False)

# create new columns
feature_importance.loc[:,'third_level_var'] = feature_importance.loc[:, 'target_variable'].copy()

# build a dictionary in order to replace video_id with more meaningful names
replace_third_level_var = dict(zip(pd.unique(variant_df.loc[:, 'variant_description']), variant_df.loc[:, 'variant_name']))

# convert dummy strings to more meaningful strings
feature_importance.loc[:,'third_level_var'].replace(to_replace = replace_third_level_var, 
            inplace = True)

# build a dummy column about 'variant_id' 
feature_importance.loc[:,'variant_id'] = feature_importance.loc[:, 'target_variable'].copy()

# build a dictionary in order to replace 'target_variable' with 'variant_id'
replace_variant_id = dict(zip(variant_df.loc[: , 'variant_description'],
                                variant_df.loc[: , 'variant_id']))

# set the correct 'variant_id' 
feature_importance.loc[:,'variant_id'].replace(to_replace = replace_variant_id,
            inplace = True)

# extract verbose variables' names
variables_to_be_used_verbose = pd.Series(pd.unique(feature_importance.loc[:, 'variable'])) 

# sort coefficients by descending order
feature_importance.sort_values(by = ['coef_abs_value'], 
                                    axis = 0, 
                                    inplace = True, 
                                    ascending = False) 

# deep copy
feat_imp_absolute_value_tmp = feature_importance.copy()

# drop useless columns
feat_imp_absolute_value_tmp.drop(labels = ['priors', 
                                           'var_name_forest', 
                                           'target_variable',
                                           'actual_coefficient'],
                                 axis = 1, 
                                 inplace = True)

# sum the coefficients for each variable
feat_imp_absolute_value = feat_imp_absolute_value_tmp.groupby(by = ['variable']).sum()

# re-scale coefficients: now the range is from 0 to 1
feat_imp_absolute_value.loc[:, 'coef_abs_value'] = MinMaxScaler().fit_transform(feat_imp_absolute_value.loc[:, 'coef_abs_value'].to_numpy().reshape(-1, 1))

# re-set index
feat_imp_absolute_value.reset_index(drop = False, inplace = True)

# sort coefficients by descending order
feat_imp_absolute_value.sort_values(by = ['coef_abs_value'], 
                                    axis = 0, 
                                    inplace = True, 
                                    ascending = False)

# save always positive coefficients as a .csv file
feat_imp_absolute_value.to_csv(os.path.sep.join([BASE_DIR_OUTPUT,
                                       output_file_name_26]),
                                    index=False)


###############################################################################
## 6. MCMC'S CHAINS PLOTTING + SUMMARY STATISTICS
# convert MCMC traces stored in ArviZ inference object to Pandas DataFrame
items_list_tmp = pd.DataFrame(arviz_inference.posterior.data_vars)
# items_list_tmp = pd.DataFrame(arviz_inference.posterior.dims) # TEST:

# rename columns 
items_list_tmp.rename(columns = { 0: 'item_name'}, inplace = True)

# build different masks
mask_cont_arviz = items_list_tmp.loc[:, 'item_name'].str.contains('beta_con_tmp_percentage')
mask_cat_selection_arviz = items_list_tmp.loc[:, 'item_name'].str.contains('beta_cat_tmp_selection')
mask_cat_audience_arviz = items_list_tmp.loc[:, 'item_name'].str.contains('beta_cat_tmp_audience')
mask_cat_browser_arviz = items_list_tmp.loc[:, 'item_name'].str.contains('beta_cat_tmp_browser')
mask_cat_city_arviz = items_list_tmp.loc[:, 'item_name'].str.contains('beta_cat_tmp_city')
mask_cat_device_arviz = items_list_tmp.loc[:, 'item_name'].str.contains('beta_cat_tmp_device')

# concatenate masks
mask_subset_concat_arviz = pd.concat([mask_cont_arviz,
                                mask_cat_selection_arviz, 
                                mask_cat_audience_arviz, 
                                mask_cat_browser_arviz, 
                                mask_cat_city_arviz, 
                                mask_cat_device_arviz], axis = 1)

# sum the booleans (indeed, each mask is a boolean True/False)
mask_subset_arviz = mask_subset_concat_arviz.sum(axis = 1).astype('bool')

# select only the priors for the "slopes" ("beta_tmp...") of the logistic regression
items_list = items_list_tmp.loc[mask_subset_arviz, 'item_name'].reset_index(drop = True)

# sort coefficients by descending order
feature_importance.sort_values(by = ['variable'], 
                                    axis = 0, 
                                    inplace = True, 
                                    ascending = True) 

# initialize DataFrame
trace_quantiles_tmp = pd.DataFrame()

# unnest ArviZ inference object
for items_12, item_type_12 in enumerate(items_list):
    
    # convert portion of ArviZ inference object to Pandas DataFrame
    arviz_tmp_01 = arviz_inference.posterior[item_type_12].to_dataframe() 
    
    # re-set index    
    arviz_tmp_02 = arviz_tmp_01.reset_index(drop = False)
      
    # delete useless columns 
    arviz_tmp_02.drop(columns = ['chain', 'draw'], inplace = True)
    
    
    # compute 0.03 + 0.97 quantiles
    trace_tmp_hdi_94 = az.hdi(ary = arviz_inference, 
                             hdi_prob = 0.94,
                             var_names =[item_type_12], 
                             filter_vars = ['like'])
    trace_hdi_94 = trace_tmp_hdi_94.to_dataframe()
    
    # move columns which are considered as index as official columns
    trace_hdi_94.reset_index(drop = False, inplace = True)
        
    # build a dictionary in order to strings with actual quantiles' floats
    replace_trace_hdi_94 = dict(zip(pd.unique(trace_hdi_94.loc[:, 'hdi']),
                                    [0.03, 0.97]))
    
    # convert strings to actual quantiles' floats
    trace_hdi_94.loc[:,'hdi'].replace(to_replace = replace_trace_hdi_94,
                inplace = True)
    
    # compute 0.25 + 0.75 quantiles
    trace_tmp_hdi_50 = az.hdi(ary = arviz_inference, 
                             hdi_prob = 0.5,
                             var_names =[item_type_12], 
                             filter_vars = ['like'])
    trace_hdi_50 = trace_tmp_hdi_50.to_dataframe()
    
    # move columns which are considered as index as official columns
    trace_hdi_50.reset_index(drop = False, inplace = True)
    
    # build a dictionary in order to strings with actual quantiles' floats
    replace_trace_hdi_50 = dict(zip(pd.unique(trace_hdi_50.loc[:, 'hdi']),
                                    [0.25, 0.75]))
    
    # convert strings to actual quantiles' floats
    trace_hdi_50.loc[:,'hdi'].replace(to_replace = replace_trace_hdi_50,
                inplace = True)
    
    # compute the median  
    trace_hdi_median = arviz_tmp_02.groupby(by = list(arviz_tmp_02.columns[:-1])).mean()
    
    # move columns which are considered as index as official columns
    trace_hdi_median.reset_index(drop = False, inplace = True)
    
    # build a Pandas Series
    quantile_median_tmp = pd.Series([0.5])
    
    # repeat values as of trace_hdi_median's rows
    quantile_median = quantile_median_tmp.repeat(trace_hdi_median.shape[0]).reset_index(drop=True, 
                                                                      inplace= False)
    
    # insert the constant quantile '0.5' representing the median
    trace_hdi_median.insert(loc = 2, 
                            column = 'hdi',
                            value = quantile_median, 
                            allow_duplicates = True)
    
    # initialize DataFrame
    trace_tmp = pd.DataFrame()
       
    # concatenate vertically
    trace_tmp = pd.concat([trace_hdi_94,
                           trace_hdi_50, 
                           trace_hdi_median],                      
                           axis = 0, 
                           ignore_index = True, 
                           sort = False)
           
    # rename columns
    trace_tmp.rename(columns = {trace_tmp.columns[0]: 'variable', 
                                trace_tmp.columns[1]: 'target_variable',
                                trace_tmp.columns[2]: 'quantiles',
                                trace_tmp.columns[3]: 'actual_coefficient'}, inplace = True)
    
    # sub-set meaningful variable's name
    var_name_arviz = feature_importance.loc[feature_importance.loc[:, 'variable'].str.contains(item_type_12[13:]), 'variable']
    
    # build a dictionary in order to replace integers by meaningful variable's name
    replace_var_name_arviz = dict(zip(pd.unique(trace_tmp.loc[:, 'variable']),
                                    pd.unique(var_name_arviz)))
    
    # convert integers to meaningful variable's name
    trace_tmp.loc[:,'variable'].replace(to_replace = replace_var_name_arviz,
                inplace = True)
    
    # append
    trace_quantiles_tmp = trace_quantiles_tmp.append(trace_tmp, 
                                       ignore_index = True, sort = False)
    
# build a dictionary in order to replace integers with more meaningful names
replace_target_variable = dict(zip(pd.unique(trace_quantiles_tmp.loc[:, 'target_variable']), variant_df.loc[1:, 'variant_description']))

# convert integers to more meaningful strings
trace_quantiles_tmp.loc[:,'target_variable'].replace(to_replace = replace_target_variable, 
            inplace = True)

# # TEST: save MCMC traces summary statistics as a .csv file
# trace_quantiles_tmp.to_csv(os.path.sep.join([BASE_DIR_OUTPUT,
#                                        'output/401_trace_df.csv']),
#                                     index=False)

# # TEST: loading custom MCMC traces summary statistics. Useful when de-bugging.
# trace_quantiles_tmp = pd.read_csv(os.path.sep.join([BASE_DIR_OUTPUT, 
#                                           'output/401_trace_df.csv']), header = 0)

# re-set index
trace_quantiles_tmp.reset_index(drop = True, inplace = True)

# new quantile name
new_quantiles_name = pd.Series(['hdi_3%',
                                'hdi_25%',
                                'hdi_50%',
                                'hdi_75%',
                                'hdi_97%'])

# build a dictionary in order to replace video_id with more meaningful names
replace_quantiles = dict(zip([0.03, 0.25, 0.5, 0.75, 0.97],
                                new_quantiles_name))

# convert video_id to more meaningful names
trace_quantiles_tmp.loc[:,'quantiles'].replace(to_replace = replace_quantiles,
            inplace = True)

# add a columns of the variable to the box plot data-set, as well
trace_slopes_long = trace_quantiles_tmp.merge(feature_importance.loc[:, ['variable', 'target_variable','var_name_forest', 'first_level_var',
                                                   'second_level_var', 
                                                   'third_level_var', 
                                                   'variant_id']], 
                                                    how = 'left',
                                                    on = ['variable', 'target_variable'])

# initialize DataFrame
not_overlapping_whisker = pd.DataFrame()

# build a column of showing when the left-most whisker is not overlapping with
# '0'. This means the effect is considered statistically significant
for items_16, video_type_16 in enumerate(pd.unique(trace_slopes_long.loc[:, 'var_name_forest'])):
    
    # sub-set     
    mask_positive_coeff_tmp = trace_slopes_long.loc[:, 'var_name_forest'].eq(video_type_16)
    positive_coeff_tmp = trace_slopes_long.loc[mask_positive_coeff_tmp, :].reset_index(drop = True)
    
    # select the median of the distribution of the coefficient only            
    positive_coeff = positive_coeff_tmp.loc[positive_coeff_tmp.loc[:, 'quantiles'].eq('hdi_50%'), 'actual_coefficient']  
   
    # select a positive coefficient only
    if positive_coeff.to_numpy() > 0.0: 
                   
        # select left-most whisker only 
        left_whisker = positive_coeff_tmp.loc[positive_coeff_tmp.loc[:, 'quantiles'].eq('hdi_3%'), 'actual_coefficient']
        
        # check whether the left-most does NOT overlap with zero
        if left_whisker.to_numpy() > 0.0:
            
           # initialize DataFrame
           not_overlapping_whisker_tmp = pd.DataFrame()
                     
           # collect statistically significant variable name
           not_overlapping_whisker_name = pd.Series(video_type_16)
           
           # collect the significant status
           not_overlapping_whisker_status = pd.Series([1])
           
           # concatenate horizontally
           not_overlapping_whisker_tmp = pd.concat([not_overlapping_whisker_name,
                                           not_overlapping_whisker_status], 
                                          axis = 1, ignore_index = False, 
                                          sort = False)
           
        else:
            # initialize DataFrame
            not_overlapping_whisker_tmp = pd.DataFrame()
               
            # collect statistically significant variable name
            not_overlapping_whisker_name = pd.Series(video_type_16)
            
            # collect the significant status
            not_overlapping_whisker_status = pd.Series([0])
            
            # concatenate horizontally
            not_overlapping_whisker_tmp = pd.concat([not_overlapping_whisker_name,
                                            not_overlapping_whisker_status], 
                                           axis = 1, ignore_index = False, 
                                           sort = False)
        

    else:
        
        # initialize DataFrame
        not_overlapping_whisker_tmp = pd.DataFrame()
           
        # collect statistically significant variable name
        not_overlapping_whisker_name = pd.Series(video_type_16)
        
        # collect the significant status
        not_overlapping_whisker_status = pd.Series([0])
        
        # concatenate horizontally
        not_overlapping_whisker_tmp = pd.concat([not_overlapping_whisker_name,
                                        not_overlapping_whisker_status], 
                                       axis = 1, ignore_index = False, 
                                       sort = False)
           
    
    # append
    not_overlapping_whisker = not_overlapping_whisker.append(not_overlapping_whisker_tmp, 
                                   ignore_index = True, 
                                   sort = False)
    
# drop duplicates
not_overlapping_whisker.drop_duplicates(inplace = True)        

# rename column
not_overlapping_whisker.rename(columns = {0 : 'var_name_forest', 
                                          1 : 'stat_signif'}, inplace = True)  

# build dummy column
trace_slopes_long['stat_signif'] = trace_slopes_long.loc[:, 'var_name_forest']
feature_importance['stat_signif'] = feature_importance.loc[:, 'var_name_forest']
  
# build a dictionary in order to replace 'var_name_forest' with a meaningful status (either significant == 1,
# or not significant == 0)
replace_stat_signif = dict(zip(not_overlapping_whisker.loc[:, 'var_name_forest'], 
                                not_overlapping_whisker.loc[:, 'stat_signif']))

# build an extra column indicating the status (either significant or not)
trace_slopes_long.loc[:,'stat_signif'].replace(to_replace = replace_stat_signif, 
            inplace = True)
feature_importance.loc[:,'stat_signif'].replace(to_replace = replace_stat_signif, 
            inplace = True)

# set the same color for the same video_id
colors_variant_id = {'control': 'red',
                    'test1': 'lime',
                    'test2': 'blue',
                    'test3': 'orange',
                    'test4': 'cyan',
                    'test5': 'purple',
                    'test6': 'black',
                    'test7': 'slategrey',
                    'test8': 'aqua',
                    'test9': 'lightcoral',
                    'test10': 'saddlebrown',
                    'test11': 'palegreen',
                    'test12': 'teal',
                    'test13': 'htopink',
                    'test14': 'mediumpurple'}

# force plot to show the variants according to a specific order 
hue_order = variant_df.loc[1:, 'variant_description']

# save summary statistics as .csv file
feature_importance.to_csv(os.path.sep.join([BASE_DIR_OUTPUT,
                                       output_file_name_25]),
                                    index=False)

# plot feature importance
sns.set(font_scale = 0.5)
pdf_01 = sns.catplot(y = 'variable', 
                     x = 'coef_abs_value', 
                     hue = 'target_variable',
                     hue_order = hue_order,
                     data = feature_importance, 
                     kind = 'bar', 
                     orient = 'h',
                     palette = colors_variant_id)
plt.title('403_bayesian: feature importance', fontsize=10)
plt.ylabel('variables/features', fontsize=5)
plt.xlabel('statistical significance (by standardizing Bayesian logistic regression coefficients)', fontsize=5)
plt.tight_layout()

# save plot as .pdf file
pdf_01.savefig(os.path.sep.join([BASE_DIR_OUTPUT, output_file_name_15]))

# close pic in order to avoid overwriting with previous pics
plt.clf()

# # TEST: loading custom MCMC traces summary statistics. Useful when de-bugging.
# trace_slopes_long = pd.read_csv(os.path.sep.join([BASE_DIR_OUTPUT, 
#                                           output_file_name_24]), header = 0)


# save quantiles from MCMC traces which have to be fed into Seaborn
trace_slopes_long.to_csv(os.path.sep.join([BASE_DIR_OUTPUT,
                                        output_file_name_24]),
                                    index = False)

# plot forest plot by Seaborn
sns.set(font_scale = 0.5)
pdf_02 = sns.boxplot(y = 'var_name_forest', 
                      x = 'actual_coefficient',
                      data = trace_slopes_long,
                      hue = 'target_variable',
                      hue_order = hue_order,
                      orient = 'h',
                      palette = colors_variant_id)
plt.title('403_bayesian: forest plot', fontsize=10)
plt.ylabel('variables/features', fontsize=5)
plt.xlabel('statistical significance (by Bayesian logistic regression coefficients)', fontsize=5)
plt.tight_layout()

# save plot as .pdf file
plt.savefig(os.path.sep.join([BASE_DIR_OUTPUT, output_file_name_28]))

# close pic in order to avoid overwriting with previous pics
plt.clf()

# build a Series and reverse order of the labels, otherwise Matplotlib put
# the labels in the wrong order
feat_labels_tmp = pd.Series(feature_importance.loc[:, 'var_name_forest']).sort_index(ascending=False)

# convert from Pandas Series to list
feature_importance_numpy = list(feat_labels_tmp.to_numpy())

# set index of the y-axis labels
tree_indices = np.arange(0, len(feature_importance.loc[:, 'var_name_forest'])) - 0.2


# plot forest plot by ArviZ
axes = az.plot_forest(arviz_inference,
                      var_names = ['beta_con_tmp_percentage', 
                                   'beta_cat_tmp_selection', 
                                   'beta_cat_tmp_audience',
                                   'beta_cat_tmp_browser', 
                                   'beta_cat_tmp_city', 
                                   'beta_cat_tmp_device'],
                      combined = True,
                      textsize = 8.0,
                      linewidth = 2,
                      markersize = 4);
axes[0].set_title('403_bayesian: forest plot, original plot',
                  fontsize = 5)

## TEST: not working as expected, since it mis-aligns the labels on the y-axis 
# plt.yticks(tree_indices, 
#             feature_importance_numpy, 
#                     fontsize=5)
# axes[0].set_yticklabels(feature_importance_numpy, 
#                         ha = 'right', 
#                         va = 'center_baseline')
# axes[0].tick_params(length=2, width=1, pad = 0.002)
# axes[0].get_yticklabels()
# axes.align_labels() # not working

plt.ylabel('variables/features',
           fontsize = 8)
plt.xlabel('statistical significance (by Bayesian logistic regression coefficients)',
           fontsize = 8)

# save plot as .pdf file
plt.savefig(os.path.sep.join([BASE_DIR_OUTPUT, output_file_name_19]))

# close pic in order to avoid overwriting with previous pics
plt.clf()


###########################################################################
## 5. GENERATE AUTOMATICALLY INTERPRETATION OF FEATURE IMPORTANCE PLOT     
# builds several masks in order to sub-set feature importance matrix
mask_percentage_watched = feature_importance.loc[:, 'variable'].str.contains('percentage')
mask_variant_id = feature_importance.loc[:, 'variable'].str.contains('variant_id')
mask_selection_order = feature_importance.loc[:, 'variable'].str.contains('selection')
mask_audience = feature_importance.loc[:, 'variable'].str.contains('audience')
mask_browser = feature_importance.loc[:, 'variable'].str.contains('browser')
mask_city = feature_importance.loc[:, 'variable'].str.contains('city')
mask_device = feature_importance.loc[:, 'variable'].str.contains('device')

# build separated a sub-set according to the variable's name
percentage_watched_subset = feature_importance.loc[mask_percentage_watched, :].reset_index(drop = True)
variant_id_subset = feature_importance.loc[mask_variant_id, :].reset_index(drop = True)
selection_order_subset = feature_importance.loc[mask_selection_order, :].reset_index(drop = True)
experiment_audience_type_subset = feature_importance.loc[mask_audience, :].reset_index(drop = True)
browser_subset = feature_importance.loc[mask_browser, :].reset_index(drop = True)
city_subset = feature_importance.loc[mask_city, :].reset_index(drop = True)
device_subset = feature_importance.loc[mask_device, :].reset_index(drop = True)

# initialize DataFrame
percentage_watched_string = pd.DataFrame()
variant_id_string = pd.DataFrame()
selection_order_string = pd.DataFrame()
experiment_audience_type_string = pd.DataFrame()
browser_string = pd.DataFrame()
city_string = pd.DataFrame()
device_string = pd.DataFrame()


# conditionals regarding the coefficients/weigths of the feature importance plot 
## 'percentage_watched'
# check whether 'event_action' variable exists
if (not percentage_watched_subset.empty):
    
    # increase granularity 
    for items_10, video_type_10 in enumerate(pd.unique(percentage_watched_subset.loc[:, 'variable'])):
        
        #sub-set
        mask_percentage_watched_tmp = percentage_watched_subset.loc[:, 'variable'].eq(video_type_10)
        percentage_watched_subset_tmp = percentage_watched_subset.loc[mask_percentage_watched_tmp, :].reset_index(drop = True)
           
        for items_11, target_var_11 in enumerate(percentage_watched_subset_tmp.loc[:, 'target_variable']):
    
            # initialize Series
            percentage_watched_tmp = []    
            
            # build a meaningful string for each significant video (control included)
            if percentage_watched_subset_tmp.loc[items_11, 'coef_abs_value'] > 0.2:
                
                #sub-set
                mask_percentage_watched_actual_tmp = trace_slopes_long.loc[:, 'variable'].eq(video_type_10)
                percentage_watched_actual_subset_tmp = trace_slopes_long.loc[mask_percentage_watched_actual_tmp, :].reset_index(drop = True)
                mask_percentage_watched_actual_subset_tmp = trace_slopes_long.loc[:, 'target_variable'].eq(target_var_11)
                percentage_watched_actual_subset_variant_tmp = percentage_watched_actual_subset_tmp.loc[mask_percentage_watched_actual_subset_tmp, :].reset_index(drop = True)
                
                # select positive (i.e. earning money) coefficient only
                percentage_watched_positive_coeff = percentage_watched_actual_subset_variant_tmp.loc[percentage_watched_actual_subset_variant_tmp.loc[:, 'quantiles'].eq('hdi_50%'), 'actual_coefficient']
                
                # check whether the coefficient is either negative or positive
                if percentage_watched_positive_coeff.to_numpy() > 0.0: 
                   
                   # select left-most whisker only 
                   percentage_watched_left_whisker = percentage_watched_actual_subset_variant_tmp.loc[percentage_watched_actual_subset_variant_tmp.loc[:, 'quantiles'].eq('hdi_3%'), 'actual_coefficient']
                   
                   # check whether the left-most does NOT overlap with zero
                   if percentage_watched_left_whisker.to_numpy() > 0.0: 
                      
                      #building result's interpretation string
                      percentage_watched_tmp = pd.Series(('01. {} , GOOD: video {} has a significant impact on such a variable rather than control video!'.format(
                          video_type_10, target_var_11))).astype('string')     
                  
            else: 
                pass
            
            # append
            percentage_watched_string = percentage_watched_string.append(percentage_watched_tmp, 
                                           ignore_index = True, sort = False)
else: 
    pass    

       
## 'variant_id'
# check whether 'variant_id' variable exists
if (not variant_id_subset.empty):
    if (np.median(variant_id_subset.loc[:, 'coef_abs_value']) > 0.2): 
        variant_id_tmp = pd.Series(('02. variant_id, WARNING: Unbalanced way of watching videos, since some user did not watch all videos just once.')).astype('string')
        
        # append
        variant_id_string = variant_id_string.append(variant_id_tmp, 
                                                ignore_index = True, sort = False)
            
    else: 
        pass
    
else: 
    pass


## 'selection_order'
# check whether 'variant_id' variable exists
if (not selection_order_subset.empty):

    # increase granularity 
    for items_14, video_type_14 in enumerate(pd.unique(selection_order_subset.loc[:, 'variable'])):
        
        #sub-set
        mask_selection_order_tmp = selection_order_subset.loc[:, 'variable'].eq(video_type_14)
        selection_order_subset_tmp = selection_order_subset.loc[mask_selection_order_tmp, :].reset_index(drop = True)
           
        for items_15, target_var_15 in enumerate(selection_order_subset_tmp.loc[:, 'target_variable']):
    
            # initialize Series
            selection_order_tmp = []    
            
            # build a meaningful string for each significant video (control included)
            if selection_order_subset_tmp.loc[items_15, 'coef_abs_value'] > 0.2:
                                             
                #sub-set
                mask_selection_order_actual_tmp = trace_slopes_long.loc[:, 'variable'].eq(video_type_14)
                selection_order_actual_subset_tmp = trace_slopes_long.loc[mask_selection_order_actual_tmp, :].reset_index(drop = True)
                mask_selection_order_actual_subset_tmp = trace_slopes_long.loc[:, 'target_variable'].eq(target_var_15)
                selection_order_actual_subset_variant_tmp = selection_order_actual_subset_tmp.loc[mask_selection_order_actual_subset_tmp, :].reset_index(drop = True)
                
                # select positive (i.e. earning money) coefficient only
                selection_order_positive_coeff = selection_order_actual_subset_variant_tmp.loc[selection_order_actual_subset_variant_tmp.loc[:, 'quantiles'].eq('hdi_50%'), 'actual_coefficient']
                
                # check whether the coefficient is either negative or positive
                if selection_order_positive_coeff.to_numpy() > 0.0: 
                   
                   # select left-most whisker only 
                   selection_order_left_whisker = selection_order_actual_subset_variant_tmp.loc[selection_order_actual_subset_variant_tmp.loc[:, 'quantiles'].eq('hdi_3%'), 'actual_coefficient']
                   
                   # check whether the left-most does NOT overlap with zero
                   if selection_order_left_whisker.to_numpy() > 0.0: 
                      
                      #building result's interpretation string
                      selection_order_tmp = pd.Series(('03. {} , GOOD: video {} has a significant impact on such a variable rather than control video!'.format(
                          video_type_14, target_var_15))).astype('string')     
                                  
            else: 
                pass
            
            # append
            selection_order_string = selection_order_string.append(selection_order_tmp, 
                                           ignore_index = True, sort = False)
else: 
    pass

        
## 'experiment_audience_type'
# check whether 'experiment_audience_type' variable exists
if (not experiment_audience_type_subset.empty):

    # increase granularity 
    for items_04, video_type_04 in enumerate(pd.unique(experiment_audience_type_subset.loc[:, 'variable'])):
        
        #sub-set
        mask_audience_tmp = experiment_audience_type_subset.loc[:, 'variable'].eq(video_type_04)
        audience_subset_tmp = experiment_audience_type_subset.loc[mask_audience_tmp, :].reset_index(drop = True)
           
        for items_05, target_var_05 in enumerate(audience_subset_tmp.loc[:, 'target_variable']):
    
            # initialize Series
            audience_tmp = []    
            
            # build a meaningful string for each significant video (control included)
            if audience_subset_tmp.loc[items_05, 'coef_abs_value'] > 0.2:
                
                #sub-set
                mask_audience_actual_tmp = trace_slopes_long.loc[:, 'variable'].eq(video_type_04)
                audience_actual_subset_tmp = trace_slopes_long.loc[mask_audience_actual_tmp, :].reset_index(drop = True)
                mask_audience_actual_subset_tmp = trace_slopes_long.loc[:, 'target_variable'].eq(target_var_05)
                audience_actual_subset_variant_tmp = audience_actual_subset_tmp.loc[mask_audience_actual_subset_tmp, :].reset_index(drop = True)
                
                # select positive (i.e. earning money) coefficient only
                audience_positive_coeff = audience_actual_subset_variant_tmp.loc[audience_actual_subset_variant_tmp.loc[:, 'quantiles'].eq('hdi_50%'), 'actual_coefficient']
                
                # check whether the coefficient is either negative or positive
                if audience_positive_coeff.to_numpy() > 0.0: 
                   
                   # select left-most whisker only 
                   audience_left_whisker = audience_actual_subset_variant_tmp.loc[audience_actual_subset_variant_tmp.loc[:, 'quantiles'].eq('hdi_3%'), 'actual_coefficient']
                   
                   # check whether the left-most does NOT overlap with zero
                   if audience_left_whisker.to_numpy() > 0.0: 
                      
                      #building result's interpretation string
                      audience_tmp = pd.Series(('04. {} , GOOD: video {} has a significant impact on such a variable rather than control video!'.format(
                          video_type_04, target_var_05))).astype('string')     
                
            else: 
                pass
            
            # append
            experiment_audience_type_string = experiment_audience_type_string.append(audience_tmp, 
                                           ignore_index = True, sort = False)
else: 
    pass


## 'browser'
# check whether 'browser' variable exists
if (not browser_subset.empty):
    
    # increase granularity 
    for items_02, video_type_02 in enumerate(pd.unique(browser_subset.loc[:, 'variable'])):
        
        #sub-set
        mask_browser_tmp = browser_subset.loc[:, 'variable'].eq(video_type_02)
        browser_subset_tmp = browser_subset.loc[mask_browser_tmp, :].reset_index(drop = True)
           
        for items_03, target_var_03 in enumerate(browser_subset_tmp.loc[:, 'target_variable']):
    
            # initialize Series
            browser_tmp = []        
    
            # build a meaningful string for each significant video (control included)
            if browser_subset_tmp.loc[items_03, 'coef_abs_value'] > 0.2:
                
                #sub-set
                mask_browser_actual_tmp = trace_slopes_long.loc[:, 'variable'].eq(video_type_02)
                browser_actual_subset_tmp = trace_slopes_long.loc[mask_browser_actual_tmp, :].reset_index(drop = True)
                mask_browser_actual_subset_tmp = trace_slopes_long.loc[:, 'target_variable'].eq(target_var_03)
                browser_actual_subset_variant_tmp = browser_actual_subset_tmp.loc[mask_browser_actual_subset_tmp, :].reset_index(drop = True)
                
                # select positive (i.e. earning money) coefficient only
                browser_positive_coeff = browser_actual_subset_variant_tmp.loc[browser_actual_subset_variant_tmp.loc[:, 'quantiles'].eq('hdi_50%'), 'actual_coefficient']
                
                # check whether the coefficient is either negative or positive
                if browser_positive_coeff.to_numpy() > 0.0: 
                   
                   # select left-most whisker only 
                   browser_left_whisker = browser_actual_subset_variant_tmp.loc[browser_actual_subset_variant_tmp.loc[:, 'quantiles'].eq('hdi_3%'), 'actual_coefficient']
                   
                   # check whether the left-most does NOT overlap with zero
                   if browser_left_whisker.to_numpy() > 0.0: 
                      
                      #building result's interpretation string
                      browser_tmp = pd.Series(('05. {} , GOOD: video {} has a significant impact on such a variable rather than control video!'.format(
                          video_type_02, target_var_03))).astype('string')     
                
            else: 
                pass
            
            # append
            browser_string = browser_string.append(browser_tmp, 
                                           ignore_index = True, sort = False)
else: 
    pass

        
## 'city'
# check whether 'city' variable exists
if (not city_subset.empty):
    
    # increase granularity 
    for items_06, video_type_06 in enumerate(pd.unique(city_subset.loc[:, 'variable'])):
        
        #sub-set
        mask_city_tmp = city_subset.loc[:, 'variable'].eq(video_type_06)
        city_subset_tmp = city_subset.loc[mask_city_tmp, :].reset_index(drop = True)
           
        for items_07, target_var_07 in enumerate(city_subset_tmp.loc[:, 'target_variable']):
    
            # initialize Series
            city_tmp = []    
            
            # build a meaningful string for each significant video (control included)
            if city_subset_tmp.loc[items_07, 'coef_abs_value'] > 0.2:
                
                #sub-set
                mask_city_actual_tmp = trace_slopes_long.loc[:, 'variable'].eq(video_type_06)
                city_actual_subset_tmp = trace_slopes_long.loc[mask_city_actual_tmp, :].reset_index(drop = True)
                mask_city_actual_subset_tmp = trace_slopes_long.loc[:, 'target_variable'].eq(target_var_07)
                city_actual_subset_variant_tmp = city_actual_subset_tmp.loc[mask_city_actual_subset_tmp, :].reset_index(drop = True)
                
                # select positive (i.e. earning money) coefficient only
                city_positive_coeff = city_actual_subset_variant_tmp.loc[city_actual_subset_variant_tmp.loc[:, 'quantiles'].eq('hdi_50%'), 'actual_coefficient']
                
                # check whether the coefficient is either negative or positive
                if city_positive_coeff.to_numpy() > 0.0: 
                   
                   # select left-most whisker only 
                   city_left_whisker = city_actual_subset_variant_tmp.loc[city_actual_subset_variant_tmp.loc[:, 'quantiles'].eq('hdi_3%'), 'actual_coefficient']
                   
                   # check whether the left-most does NOT overlap with zero
                   if city_left_whisker.to_numpy() > 0.0: 
                      
                      #building result's interpretation string
                      city_tmp = pd.Series(('06. {} , GOOD: video {} has a significant impact on such a variable rather than control video!'.format(
                          video_type_06, target_var_07))).astype('string')             
                
            else: 
                pass
            
            # append
            city_string = city_string.append(city_tmp, 
                                           ignore_index = True, sort = False)
else: 
    pass


## 'device_type'
# check whether 'device_type' variable exists
if (not device_subset.empty):
    
    # increase granularity 
    for items_08, video_type_08 in enumerate(pd.unique(device_subset.loc[:, 'variable'])):
        
        #sub-set
        mask_device_tmp = device_subset.loc[:, 'variable'].eq(video_type_08)
        device_subset_tmp = device_subset.loc[mask_device_tmp, :].reset_index(drop = True)
           
        for items_09, target_var_09 in enumerate(device_subset_tmp.loc[:, 'target_variable']):
    
            # initialize Series
            device_tmp = []    
            
            # build a meaningful string for each significant video (control included)
            if device_subset_tmp.loc[items_09, 'coef_abs_value'] > 0.2:
                
                #sub-set
                mask_device_actual_tmp = trace_slopes_long.loc[:, 'variable'].eq(video_type_08)
                device_actual_subset_tmp = trace_slopes_long.loc[mask_device_actual_tmp, :].reset_index(drop = True)
                mask_device_actual_subset_tmp = trace_slopes_long.loc[:, 'target_variable'].eq(target_var_09)
                device_actual_subset_variant_tmp = device_actual_subset_tmp.loc[mask_device_actual_subset_tmp, :].reset_index(drop = True)
                
                # select positive (i.e. earning money) coefficient only
                device_positive_coeff = device_actual_subset_variant_tmp.loc[device_actual_subset_variant_tmp.loc[:, 'quantiles'].eq('hdi_50%'), 'actual_coefficient']
                
                # check whether the coefficient is either negative or positive
                if device_positive_coeff.to_numpy() > 0.0: 
                   
                   # select left-most whisker only 
                   device_left_whisker = device_actual_subset_variant_tmp.loc[device_actual_subset_variant_tmp.loc[:, 'quantiles'].eq('hdi_3%'), 'actual_coefficient']
                   
                   # check whether the left-most does NOT overlap with zero
                   if  device_left_whisker.to_numpy() > 0.0: 
                      
                      #building result's interpretation string
                      device_tmp = pd.Series(('07. {} , GOOD: video {} has a significant impact on such a variable rather than control video!'.format(
                          video_type_08, target_var_09))).astype('string')     
                
            else: 
                pass
            
            # append
            device_string = device_string.append(device_tmp, 
                                           ignore_index = True, sort = False)
else: 
    pass


# build vector with all interpretations (one for each variable)
vector_interpretation = pd.Series(['percentage_watched_string', 
                                   'variant_id_string', 
                                   'selection_order_string', 
                                   'experiment_audience_type_string',
                                   'browser_string',
                                   'city_string',
                                   'device_string'])

# initialize Pandas DataFrame
interpretation_text = pd.DataFrame()

# discard empty Pandas DataFrame
for item_12, item_name_12 in enumerate(vector_interpretation) :
    
    # initialize Pandas DataFrame
    name_df = pd.DataFrame()
    
    # evaluate string
    name_df = pd.eval(item_name_12)
    
    # if not empty, keep it
    if (not name_df.empty):
        
        # append
        interpretation_text = interpretation_text.append(name_df, 
                                       ignore_index = True, sort = False)      
        
    # if not empty, discard it
    else: 
        pass
 
# rename column
interpretation_text.rename(columns = {0 : 'interpretation_description'}, inplace = True)    

# convert strings in a Pandas DataFrame to a list of strings
items_list_interpretation = interpretation_text.values.tolist()

# un-nest the list 
items_interpretation = list(chain.from_iterable(items_list_interpretation))

                      
###########################################################################
## 6. CONVERT AUTOMATED INTERPRETATION TO MARKDOWN FILE                         
# MarkDown: initialize a MarkDown file
mdFile = MdUtils(file_name=os.path.sep.join([BASE_DIR_OUTPUT, output_file_name_13]), 
                  title='Automated interpretation') 

# MarkDown: add new header
mdFile.new_header(3, "Short description")

# MarkDown: add new paragraph
mdFile.new_paragraph("As you can see in the legend below, the forest plot\n"
                     "takes the control video as baseline. All other variants are instead\n"
                     "compared against it.Therefore, all the other variants are shown in the forest plot,\n"
                     "whereas the control video is not shown. In the legend below the column\n"
                     "priors (e.g. beta_tmp[0,0]) is shown in the y-axis of\n"
                     "the forest plot. The first digit in beta_tmp[0,0] corresponds\n"
                     "to the column named variable in the legend. Instead, the second digit\n"
                     "corresponds to the column named target variable in the legend.\n"
                     "In general, everything which is away from zero is significant.\n"
                     "The oppsite is also true: everything which is close to zero is not significant.\n")

# MarkDown: add empty line
mdFile.new_line()

# MarkDown: add new header
mdFile.new_header(3, "Legend explaining how to interpret beta_tmp[ , ] nomenclature")

# convert strings in a Pandas DataFrame to a list of strings
list_of_strings = feature_importance.columns[0:3].values.tolist()


# concatenate rows of the summary statistics table
for index_14 in range(feature_importance.shape[0]):
	list_of_strings.extend([feature_importance.loc[index_14, 'priors'],
                         feature_importance.loc[index_14, 'var_name_forest'],
                         str(feature_importance.loc[index_14, 'actual_coefficient'])])
    
# MarkDown: add empty line    
mdFile.new_line()

# MarkDown: create a table
mdFile.new_table(columns=(feature_importance.shape[1] - 8),
                 rows=(feature_importance.shape[0] + 1), 
                 text=list_of_strings, text_align='center')

# MarkDown: add empty line    
mdFile.new_line()

# MarkDown: add new header
mdFile.new_header(3, "Hierarchical and not-hierarchical variables for the bayesian model")

# MarkDown: add new paragraph
mdFile.new_paragraph("The bayesian model needs a hierachical variable and \n"
                      "some not hierarchical variables. Usually, the hierachical\n"
                      "variable is the one which explains the highest variance.\n"
                      "E.g. the variable gender can explain the highest variance\n"
                      "for videos showing either cars or beauty products. Instead, the variable\n"
                      "browser should explain the least variance for those video types.\n"
                      "Note that the hierarchical variable is never shown in the forest plot\n"
                      "since all the results are compared against it. In simple words, by using\n"
                      "a hierarchical variable is like squeezing out the last drop of knowledge\n"
                      "from the data-set. In conclusion, the model using the hierarchical variable outperforms\n"
                      "the model not using it.\n")
                      
# MarkDown: add empty line    
mdFile.new_line()

# MarkDown: add new line    
mdFile.new_line("The HIERARCHICAL VARIABLE used was: {}\n".format(hierarchical_variable))

# MarkDown: add new line    
mdFile.new_line("The NOT-HIERARCHICAL VARIABLES used were: {}\n".format(variables_to_be_used_verbose))

# MarkDown: add new header
mdFile.new_header(3, "Interpretation of the forest plot by variable name: significant/not significant interactions")
      
# MarkDown: add list of strings (each string is an interpretation for a specific
# variable)
mdFile.new_list(items=items_interpretation, marked_with='1') 

# MarkDown: save MarkDown file   
mdFile.create_md_file() 


# end time according to computer clock
end_time = time.time()

# calculate total execution time
total_execution_time = pd.Series(np.round((end_time - start_time), 2)).rename('total_runtime_seconds')

# shows run-time's timestamps + total execution time
print('start time (unix timestamp):{}'.format(start_time))
print('end time (unix timestamp):{}'.format(end_time))
print('total execution time (seconds):{}'.format(total_execution_time.iloc[0]))

# configure notification sending to server when the script has finishes
# url ="test_fake" # TEST
payload = {'ModelName': str(folder_name[0]), 
           'ModelOutputResult': str('The "BAYESIAN MODEL PLOTS" are ready!')}
headers = {
  'Content-Type': 'application/json'
}

# check the machine you are using and send notification only when on server
if RELEASE == LOCAL_COMPUTER: # Linux laptop
    pass
else:
    try:
        # send notification
        response = requests.request(method = "POST", 
                                    url = url.iloc[0, 0], 
                                    headers = headers, 
                                    json = payload)
        
        # print locally?
        print(response.text.encode('utf8'))
    except: 
        print('Notification for reaching the end of the script not sent!')




  




  

