"""
TITLE: "Plotting feature importance from gradient boosting classifier's coefficients
    by loading .joblib file"
AUTHOR: Paolo Ranzi 
PYTHON VERSION: 3.8.5

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
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import shap
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import chain
from mdutils.mdutils import MdUtils
import time
import requests

   
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
# LOCAL_COMPUTER = '5.4.0-1037-gcp'
LOCAL_COMPUTER ='4.9.0-14-amd64'
# LOCAL_COMPUTER = '5.4.0-77-generic'


# TEST:
if RELEASE == LOCAL_COMPUTER: # Linux laptop
    # BASE_DIR_INPUT = ('/home/paolo/Dropbox/Univ/scripts/python_scripts/wright_keith/production_safety_test/758_grad_boost')
    BASE_DIR_INPUT = ('/home/paolo/Dropbox/Univ/scripts/python_scripts/wright_keith/758_grad_boost_20210502')
    BASE_DIR_OUTPUT = BASE_DIR_INPUT
    INPUT_FOLDER = BASE_DIR_INPUT


# set input/output file names
input_file_name_01 = ('252_SPLITTING/output/input_cross_validation/train_set.csv')
input_file_name_03 = ('input/dictionary_variants_id.csv')
input_file_name_07 = ('input/config/url_for_pushing_notification.csv')
output_file_name_11 = ('output/758_analysis_feature_name_list.csv')
output_file_name_13 = ('output/758_feat_imp_interpretation.md') 
output_file_name_14 = ('output/758_columns_names.csv') 
output_file_name_15 = ('output/758_analysis_feature_importance.pdf') 
output_file_name_16 = ('output/758_feat_imp_permutation.joblib') 
output_file_name_17 = ('output/758_feat_imp_permutation_sklearn.joblib') 
output_file_name_19 = ('output/758_feature_importance_shap_original.pdf') 
output_file_name_20 = ('output/758_feature_importance_raw_data.csv') 
output_file_name_21 = ('output/758_feature_importance_absolute_value.csv')  
output_file_name_24 = ('output/758_feature_importance_neg_pos_bar_plot.csv')
output_file_name_25 = ('output/758_feature_importance_neg_pos_bar_plot.pdf')
output_file_name_28 = ('output/758_feature_importance_positive_only_plot.csv')
output_file_name_29 = ('output/758_feature_importance_positive_only_plot.pdf')
output_file_name_30 = ('output/758_feature_importance_sklearn.csv')
output_file_name_31 = ('output/758_feature_importance_sklearn.pdf') 



###############################################################################
## 3. LOADING DATA-SET 
# start clocking time
start_time = time.time()

# loading the .csv files with raw data 
try: 
    user_activities = pd.read_csv(os.path.sep.join([INPUT_FOLDER, 
                                              input_file_name_01]), header = 0)
except OSError:
    # print("OS error: {0}".format(err))
    print("\n"
          "\n"
          "WARNING: The train-set was not found!!!!! \n"
          "The train-set MUST be present in the followig folder: 252_SPLITTING/output/input_cross_validation/. \n"
          "Please run module 252 before running the present module. \n"
          "\n"
          "\n")

# loading the .csv files 
variant_df = pd.read_csv(os.path.sep.join([INPUT_FOLDER, 
                                         input_file_name_03]), header = 0)

feature_name_list_tmp = pd.read_csv(os.path.sep.join([BASE_DIR_INPUT, 
                                         output_file_name_11]), header = 0)

columns_names = pd.read_csv(os.path.sep.join([BASE_DIR_INPUT, 
                                         output_file_name_14]), header = 0)

# check whether "url_for_pushing_notification.csv" exists
if os.path.exists(os.path.sep.join([INPUT_FOLDER, 
                                         input_file_name_07])):
    url = pd.read_csv(os.path.sep.join([INPUT_FOLDER, 
                                             input_file_name_07]), header = None, 
                                         dtype = 'str')
else:
    # create empty variable
    url = pd.DataFrame()

selected_variables = pd.Series(['percentage_watched',
                                'gender', 
                                'selection_order', 
                                'experiment_audience_type', 
                                'browser', 
                                'city', 
                                'device_type']).rename('selected_variables')

# load .joblib objects
result_importance = joblib.load(os.path.sep.join([BASE_DIR_OUTPUT, output_file_name_16])) 

result_importance_sklearn = joblib.load(os.path.sep.join([BASE_DIR_OUTPUT, output_file_name_17])) 


################################################################################
## 4. PRE-PROCESSING
# save  folder name since it contains the model version (e.g. "906_logistic_regression_20200731")
folder_name = pd.Series(os.getcwd()).rename('model_version_in_the_folder_name')

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
user_id_integer_tmp = user_activities.loc[:, 'user_id']

# rename column 
user_id_integer = user_id_integer_tmp.rename('user_id_int') 

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
                           'gender': 'gender_original',                          
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
user_activities_concat_04.loc[:,'gender'] = le.fit_transform(user_activities_concat_04.loc[:,'gender_original'])
user_activities_concat_04.loc[:,'best_video'] = le.fit_transform(user_activities_concat_04.loc[:,'best_video_original'])
user_activities_concat_04.loc[:,'experiment_audience_type'] = le.fit_transform(user_activities_concat_04.loc[:,'experiment_audience_type_original'])
user_activities_concat_04.loc[:,'browser'] = le.fit_transform(user_activities_concat_04.loc[:,'browser_original'])
user_activities_concat_04.loc[:,'city'] = le.fit_transform(user_activities_concat_04.loc[:,'city_original'])
user_activities_concat_04.loc[:,'device_type'] = le.fit_transform(user_activities_concat_04.loc[:,'device_type_original'])

# one-hot-encoding categorical variables with the requirement of performing
# one-hot-encoding only when there are >= 2 items in each variable. Indeed, you 
# can NOT do one-hot-encoding with a constant! At the same time you can NOT run
# statistics with a constant! Thus better ignorning costants when feeding in
# data to a statistical model. 
## 'selection_order'
if pd.unique(user_activities_concat_04.loc[:, 'selection_order']).shape[0] >= 2: 
    dummy_selection_order = pd.get_dummies(user_activities_concat_04.loc[:, 'selection_order'], 
                                   prefix = 'selection_order', 
                                   drop_first = False)     
else:
    dummy_selection_order = user_activities_concat_04.loc[:, 'selection_order'].copy() 
    
## 'gender'
if pd.unique(user_activities_concat_04.loc[:, 'gender']).shape[0] >= 2: 
    dummy_gender = pd.get_dummies(user_activities_concat_04.loc[:, 'gender'], 
                                   prefix = 'gender', 
                                   drop_first = False)     
else:
    dummy_gender = user_activities_concat_04.loc[:, 'gender'].copy() 
       
## 'experiment_audience_type'
if pd.unique(user_activities_concat_04.loc[:, 'experiment_audience_type']).shape[0] >= 2: 
    dummy_experiment_audience_type = pd.get_dummies(user_activities_concat_04.loc[:, 'experiment_audience_type'],
                                    prefix = 'experiment_audience_type', 
                                    drop_first = False)   
else:
    dummy_experiment_audience_type = user_activities_concat_04.loc[:, 'experiment_audience_type'].copy()

    
## 'browser'
if pd.unique(user_activities_concat_04.loc[:, 'browser']).shape[0] >= 2: 
    dummy_browser = pd.get_dummies(user_activities_concat_04.loc[:, 'browser'], 
                                   prefix = 'browser', 
                                   drop_first = False)     
else:
    dummy_browser = user_activities_concat_04.loc[:, 'browser'].copy()

    
## 'city'
if pd.unique(user_activities_concat_04.loc[:, 'city']).shape[0] >= 2: 
    dummy_city = pd.get_dummies(user_activities_concat_04.loc[:, 'city'], 
                                    prefix = 'city', 
                                    drop_first = False)    
else:
    dummy_city = user_activities_concat_04.loc[:, 'city'].copy()
    
## 'device_type'
if pd.unique(user_activities_concat_04.loc[:, 'device_type']).shape[0] >= 2: 
    dummy_device_type = pd.get_dummies(user_activities_concat_04.loc[:, 'device_type'], 
                                    prefix = 'device', 
                                    drop_first = False)    
else:
    dummy_device_type = user_activities_concat_04.loc[:, 'device_type'].copy()

    
# concatenate
user_activities_concat_05 = pd.concat([user_activities_concat_04.loc[:, ['session_id', 
                                       'user_id', 
                                       'percentage_watched', 
                                       'best_video']],
                                        dummy_selection_order,
                                        dummy_gender,
                                        dummy_experiment_audience_type, 
                                        dummy_browser, 
                                        dummy_city, 
                                        dummy_device_type], axis = 1)

# initialize Pandas DataFrame
string_contained = pd.DataFrame()

# loop through the variables that we want to feed into the statistical analysis
for items_20, variable_name in enumerate(selected_variables):
    
    # print(items_20, variable_name)
    
    # initialize Pandas Series
    string_contained_tmp =  []
    
    # make explicit special cases otherwise Pandas.str.contains does not work!
    
    if variable_name == 'percentage_watched': 
    
        # always include it
        string_contained_tmp = pd.Series(data = user_activities_concat_05.columns.str.contains('percentage_')).rename('bool_tmp')
              
    elif variable_name == 'selection_order': 
        
        ## 'selection_order': exclude such a variable from statistical
        ## computation in case it has only one item inside (thus it is a constant)
        if pd.unique(user_activities_concat_04.loc[:, 'selection_order']).shape[0] >= 2:
            
            # select the one-hot-encoded columns from the list of variables we want to 
            # feed into the statistical analysis
            string_contained_tmp = pd.Series(data = user_activities_concat_05.columns.str.contains('selection_order_')).rename('bool_tmp')
            
        else:
            pass
        
    elif variable_name == 'gender': 
        
        ## 'gender': exclude such a variable from statistical
        ## computation in case it has only one item inside (thus it is a constant)
        if pd.unique(user_activities_concat_04.loc[:, 'gender']).shape[0] >= 2:
            
            # select the one-hot-encoded columns from the list of variables we want to 
            # feed into the statistical analysis
            string_contained_tmp = pd.Series(data = user_activities_concat_05.columns.str.contains('gender_')).rename('bool_tmp')
            
        else:
            pass
             
    elif variable_name == 'experiment_audience_type': 
        
        ## 'experiment_audience_type': exclude such a variable from statistical
        ## computation in case it has only one item inside (thus it is a constant)
        if pd.unique(user_activities_concat_04.loc[:, 'experiment_audience_type']).shape[0] >= 2:
            
            # select the one-hot-encoded columns from the list of variables we want to 
            # feed into the statistical analysis
            string_contained_tmp = pd.Series(data = user_activities_concat_05.columns.str.contains('audience_')).rename('bool_tmp')
            
        else:
            pass
        
    elif variable_name == 'browser': 
        
        ## 'browser': exclude such a variable from statistical
        ## computation in case it has only one item inside (thus it is a constant)
        if pd.unique(user_activities_concat_04.loc[:, 'browser']).shape[0] >= 2:
            
            # select the one-hot-encoded columns from the list of variables we want to 
            # feed into the statistical analysis
            string_contained_tmp = pd.Series(data = user_activities_concat_05.columns.str.contains('browser_')).rename('bool_tmp')
            
        else:
            pass
        
    elif variable_name == 'city': 
        
        ## 'city': exclude such a variable from statistical
        ## computation in case it has only one item inside (thus it is a constant)
        if pd.unique(user_activities_concat_04.loc[:, 'city']).shape[0] >= 2:
            
            # select the one-hot-encoded columns from the list of variables we want to 
            # feed into the statistical analysis
            string_contained_tmp = pd.Series(data = user_activities_concat_05.columns.str.contains('city_')).rename('bool_tmp')
            
        else:
            pass    
        
    elif variable_name == 'device_type': 
        
        ## 'device_type': exclude such a variable from statistical
        ## computation in case it has only one item inside (thus it is a constant)
        if pd.unique(user_activities_concat_04.loc[:, 'device_type']).shape[0] >= 2:
            
            # select the one-hot-encoded columns from the list of variables we want to 
            # feed into the statistical analysis
            string_contained_tmp = pd.Series(data = user_activities_concat_05.columns.str.contains('device_')).rename('bool_tmp')
            
        else:
            pass
        
    # append
    string_contained = string_contained.append(string_contained_tmp, 
                                    ignore_index = False, sort = False)

# transpose + change data type of DataFrame
string_contained_trans = string_contained.T.astype('bool')

# sum the booleans (indeed, each mask is a boolean True/False) 
selected_columns_tmp = string_contained_trans.sum(axis = 1).astype('bool')

# select only the one-hot-encoded columns with 'True'
selected_columns = selected_columns_tmp.index[selected_columns_tmp]

# build data-set to be fed into statistical analysis
X = user_activities_concat_05.iloc[:, selected_columns]
y = user_activities_concat_05.loc[:, 'best_video']


###############################################################################
## 4. DATA MANIPULATION OF SHAPELY VALUES
# convert DataFrame into Series
feature_name_list = feature_name_list_tmp.copy()

# convert DataFrame into Series
feature_name_variables = feature_name_list.loc[:, 'feature_name']

# initialize Pandas DataFrame
unnest_df = pd.DataFrame() 

# un-nest list containing Shapely values
for counter in range(0, len(result_importance)):
      
    # sub-setting   
    unnest_single_column_tmp = result_importance[counter]
    
    # initialize Pandas DataFrame
    unnest_median_series = pd.Series(dtype = 'float64')
       
    # delete the Shapely values with "zeros" for each column 
    for counter_var in range(0, unnest_single_column_tmp.shape[1]):
                
        # sub-setting by column   
        unnest_single_column = pd.Series(unnest_single_column_tmp[np.flatnonzero(unnest_single_column_tmp[:, counter_var]), counter_var])
        
        if (not unnest_single_column.empty):
        
            # computing the median           
            unnest_median = pd.Series(unnest_single_column.abs().median().round(decimals = 3))
                        
        else:
            # computing the median           
            unnest_median = pd.Series(np.zeros(1))
            
        
        # concatenate vertically            
        unnest_median_series = unnest_median_series.append(unnest_median, 
                                   ignore_index=True)
                      
        # transpose Series and build a DataFrame
        unnest_single_video = pd.DataFrame(unnest_median_series).transpose()
    
    # append   
    unnest_df = unnest_df.append(unnest_single_video, 
                                   ignore_index=True,
                                   sort = False) 
       
# build Pandas Dataframe by combining variables' names  
# and Shapley values
feature_importance_tmp = pd.DataFrame(unnest_df.to_numpy(), columns = feature_name_variables)

# reset index and keep the "index/type of videos" column
feature_importance_tmp.reset_index(drop = False, inplace = True)

# rename column
feature_importance_tmp.rename(columns = {'index' : 'target_variable'}, inplace = True)

# build dummy vector: it corresponds to the classes of control + variants 
# available in the target variable 'best_video'
dummy_target = np.linspace(start = 0, 
                stop = (variant_df.shape[0] - 1), 
                num = variant_df.shape[0], 
                dtype = 'int8')

# build a dictionary in order to replace video_id with more meaningful names
replace_variant_name_inverse = dict(zip(dummy_target, variant_df.loc[:, 'variant_description']))

# convert video_id for 'event_label' into more meaningful names
feature_importance_tmp.loc[:,'target_variable'].replace(to_replace = replace_variant_name_inverse, 
            inplace = True)

# reshape DataFrame. WARNING: it is creating two new columns with new names ('variable' + 'value')
feature_importance_raw = pd.melt(feature_importance_tmp, id_vars = ['target_variable'])

# rename column
feature_importance_raw.rename(columns = {'value' : 'actual_coefficient', 
                                     'feature_name' : 'variable'}, inplace = True)

# create column by computing the absolute value of the actual coefficients
feature_importance_raw.loc[:,'not_rescaled_coeff'] = feature_importance_raw.loc[:, 'actual_coefficient'].abs()

# combine the strings of 2 columns into 1 column
feature_importance_raw['var_name_forest'] = feature_importance_raw.loc[:, 'variable'].str.cat(feature_importance_raw.loc[:, 'target_variable'], sep = '_')

# initialize DataFrame
feature_importance = pd.DataFrame()

# subtract control's coefficient and rescale variant's coefficient, accordingly
for index_01, var_name_01 in enumerate(pd.unique(feature_importance_raw.loc[:, 'variable'])):

    #sub-set
    mask_feature_importance_raw = feature_importance_raw.loc[:, 'variable'].eq(var_name_01)
    feature_importance_raw_tmp = feature_importance_raw.loc[mask_feature_importance_raw, :].reset_index(drop = True)
    
    # find control's coefficient and variant's coefficient
    control_tmp = feature_importance_raw_tmp.loc[feature_importance_raw.loc[:, 'target_variable'].eq('control'), :].reset_index(drop = True)
    variant_tmp = feature_importance_raw_tmp.loc[feature_importance_raw.loc[:, 'target_variable'].ne('control'), :].reset_index(drop = True)
    
    # re-scaling (i.e. subtraction) of variant's coefficient vs control's coefficient
    variant_tmp.loc[:,'coef_sub_value'] = variant_tmp.loc[:, 'not_rescaled_coeff'].sub(control_tmp.loc[:, 'not_rescaled_coeff'].abs()[0]).abs()
    
    # append
    feature_importance = feature_importance.append(variant_tmp, 
                                   ignore_index = True, sort = False)

# re-scale coefficients: now the range is from 0 to 1
feature_importance.loc[:, 'coef_abs_value'] = MinMaxScaler().fit_transform(feature_importance.loc[:, 'coef_sub_value'].to_numpy().reshape(-1, 1))

# copy dummy strings before replacing tham with the correct ones
feature_importance.loc[:, 'first_level_var'] = feature_importance.loc[:, 'variable'].copy()
feature_importance.loc[:, 'second_level_var'] = feature_importance.loc[:, 'variable'].copy()
feature_importance.loc[:, 'third_level_var'] = feature_importance.loc[:, 'target_variable'].copy()
feature_importance.loc[:, 'variant_id'] = feature_importance.loc[:, 'target_variable'].copy()

# build a dictionary in order to replace video_id with more meaningful names
replace_first_level_var = dict(zip(pd.unique(feature_importance.loc[:, 'variable']), 
                                   feature_name_list.loc[:, 'first_level_var']))
replace_second_level_var = dict(zip(pd.unique(feature_importance.loc[:, 'variable']), 
                                    feature_name_list.loc[:, 'second_level_var']))
replace_third_level_var = dict(zip(pd.unique(variant_df.loc[:, 'variant_description']), 
                                   variant_df.loc[:, 'variant_name']))
replace_variant_id = dict(zip(variant_df.loc[: , 'variant_description'],
                                variant_df.loc[: , 'variant_id']))

# convert dummy strings to more meaningful strings
feature_importance.loc[:,'first_level_var'].replace(to_replace = replace_first_level_var, 
            inplace = True)
feature_importance.loc[:,'second_level_var'].replace(to_replace = replace_second_level_var, 
            inplace = True)
feature_importance.loc[:,'third_level_var'].replace(to_replace = replace_third_level_var, 
            inplace = True)
feature_importance.loc[:,'variant_id'].replace(to_replace = replace_variant_id,
            inplace = True)

# sort coefficients by descending order
feature_importance.sort_values(by = ['coef_abs_value'], 
                                    axis = 0, 
                                    inplace = True, 
                                    ascending = False)      

# initialize Pandas DataFrame
unnest_single_video_boxplot = pd.DataFrame() 
unnest_df_boxplot = pd.DataFrame() 

# un-nest list containing Shapely values
for counter_boxplot in range(0, len(result_importance)):
    
    # sub-setting by 'video_id'   
    unnest_single_video_boxplot_tmp = pd.DataFrame(result_importance[counter_boxplot])
    
    # collect 'target_variable' value
    unnest_boxplot_target_variable_tmp = pd.Series(counter_boxplot)
    
    # creating 'target_variable' column
    unnest_boxplot_target_variable = unnest_boxplot_target_variable_tmp.repeat(unnest_single_video_boxplot_tmp.shape[0]).rename('target_variable')
    
    # re-set index of Pandas Series
    unnest_boxplot_target_variable.reset_index(drop = True, inplace = True) 
    
    # copy Series as new column
    unnest_single_video_boxplot_tmp['target_variable']= unnest_boxplot_target_variable.copy()
       
    # append   
    unnest_df_boxplot = unnest_df_boxplot.append(unnest_single_video_boxplot_tmp, 
                                   ignore_index=True,
                                   sort = False) 

# convert video_id for 'target_variable' into more meaningful names
unnest_df_boxplot.loc[:,'target_variable'].replace(to_replace = replace_variant_name_inverse, 
            inplace = True)

# build a dictionary in order to replace columns with more meaningful names
unnest_df_boxplot_columns = dict(zip(unnest_df_boxplot.iloc[:, :-1].columns, feature_name_variables))

# rename columns
unnest_df_boxplot.rename(columns = unnest_df_boxplot_columns, inplace = True)  

# initialize Pandas DataFrame
unnest_df_boxplot_long = pd.DataFrame()

# loop through all columns of the unnested Shapely results
for index_03, var_name_03 in enumerate(unnest_df_boxplot.iloc[:, :-1].columns):
    
    # index_03 = 11
    # var_name_03 = 'city_New York'
    
    # initialize Pandas DataFrame
    unnest_df_boxplot_subset = pd.DataFrame()
    
    # build a mask
    mask_tmp = np.flatnonzero(unnest_df_boxplot.loc[:, var_name_03])
       
    # sub-setting rows by 2 criteria: 
    # 1. only Shapley values different from '0'; 
    # 2. Shapley values which belong to the specific variable 'var_name_03'
    unnest_df_boxplot_subset = unnest_df_boxplot.loc[mask_tmp, [var_name_03, 'target_variable']]
    
    # re-set index
    unnest_df_boxplot_subset.reset_index(drop = True, inplace = True)
    
    # check whether the whole variable/column has Shapely values all set to '0'
    # if it is not empty do something...
    if (not unnest_df_boxplot_subset.empty):
        
        # check whether there are missing 'target_variable' bacause they have Shapely values all set to '0'
        if pd.Series(pd.unique(unnest_df_boxplot_subset.loc[:, 'target_variable']).shape[0]).ne(len(pd.unique(y))).bool():
            
            # compute median 
            unnest_df_boxplot_not_missing = unnest_df_boxplot_subset.groupby(by = ['target_variable']).median()
                       
            # re-set index 
            unnest_df_boxplot_not_missing.reset_index(drop = False, inplace = True) 
            
            # rename columns
            unnest_df_boxplot_not_missing.rename(columns = {var_name_03 : 'actual_coefficient'}, inplace = True) 
            
            # collect variable's name
            variable_column_name_not_missing_tmp = pd.Series(var_name_03)
            
            # creating 'variable' column
            variable_column_tmp_not_missing = variable_column_name_not_missing_tmp.repeat(unnest_df_boxplot_not_missing.shape[0])
            
            # re-set index of Pandas Series
            variable_column_tmp_not_missing.reset_index(drop = True, inplace = True) 
            
            # concatenate the two DataFrame horizontally
            unnest_df_boxplot_not_missing['variable'] = variable_column_tmp_not_missing.copy()
           
            # initialize DataFrame
            missing_target_variable = pd.DataFrame() 
            
            # loop through the total number of 'target_variable'
            for index_04, var_name_04 in enumerate(pd.unique(unnest_df_boxplot.loc[:, 'target_variable'])):
                
                if unnest_df_boxplot_not_missing.loc[:, 'target_variable'].str.contains(var_name_04).any():
                    pass
                
                else:
                    
                    # collect the missing 'target_variable'
                    missing_target_variable_tmp = pd.Series(var_name_04)
                
                    # build list of missing 'target_variable'
                    missing_target_variable = missing_target_variable.append(missing_target_variable_tmp, 
                                         ignore_index=True,
                                         sort = False) 
            
            # collect variable's name
            variable_column_name = pd.Series(var_name_03)
            
            # creating 'variable' column
            variable_column_tmp = variable_column_name.repeat(missing_target_variable.shape[0])
            
            # re-set index of Pandas Series
            variable_column_tmp.reset_index(drop = True, inplace = True) 
            
            # create fake 'actual_coefficient' 's value
            filler_actual_coefficient_tmp = pd.Series(0.0001)
            
            # creating fake 'actual_coefficient' column
            filler_actual_coefficient = filler_actual_coefficient_tmp.repeat(missing_target_variable.shape[0])
            
            # re-set index of Pandas Series
            filler_actual_coefficient.reset_index(drop = True, inplace = True)
            
            # build dictionary with filler data
            filler_data = {'target_variable': list(missing_target_variable.iloc[:, 0]),
                         'actual_coefficient': list(filler_actual_coefficient),
                         'variable': list(variable_column_tmp)}
                         
            # build filler DataFrame
            filler_df = pd.DataFrame(data = filler_data)
            
            # concatenate the two DataFrame vertically
            unnest_df_boxplot_long_tmp = pd.concat([unnest_df_boxplot_not_missing,
                                          filler_df], 
                                  axis = 0, ignore_index = True, 
                                  sort = False).reset_index(drop = True)
            
            # sort values alphabetically
            unnest_df_boxplot_long_tmp.sort_values(by = 'target_variable', 
                       axis = 0, 
                       inplace = True, 
                       ascending = True)

               
        # if it is not empty ALL all the 'target_variable' are present, do something...
        else:
    
            # compute median 
            unnest_df_boxplot_long_tmp = unnest_df_boxplot_subset.groupby(by = ['target_variable']).median()
                       
            # re-set index 
            unnest_df_boxplot_long_tmp.reset_index(drop = False, inplace = True) 
            
            # rename columns
            unnest_df_boxplot_long_tmp.rename(columns = {var_name_03 : 'actual_coefficient'}, inplace = True) 
            
            # collect variable's name
            variable_column_name = pd.Series(var_name_03)
            
            # creating 'variable' column
            variable_column_tmp = variable_column_name.repeat(unnest_df_boxplot_long_tmp.shape[0]).rename('variable')
            
            # re-set index of Pandas Series
            variable_column_tmp.reset_index(drop = True, inplace = True) 
            
            # copy Series as new column
            unnest_df_boxplot_long_tmp['variable'] = variable_column_tmp.copy()
            
            # rename columns
            unnest_df_boxplot_long_tmp.rename(columns = {var_name_03 : 'actual_coefficient'}, inplace = True)    
    
    
    # if it is empty do something...   
    else: 
        
        # collect variable's name
        variable_column_name = pd.Series(var_name_03)
        
        # creating 'variable' column
        variable_column_tmp = variable_column_name.repeat(pd.unique(unnest_df_boxplot.loc[:, 'target_variable']).shape[0])
        
        # re-set index of Pandas Series
        variable_column_tmp.reset_index(drop = True, inplace = True) 
        
        # create fake 'actual_coefficient' 's value
        filler_actual_coefficient_tmp = pd.Series(0.0001)
        
        # creating fake 'actual_coefficient' column
        filler_actual_coefficient = filler_actual_coefficient_tmp.repeat(pd.unique(unnest_df_boxplot.loc[:, 'target_variable']).shape[0])
        
        # re-set index of Pandas Series
        filler_actual_coefficient.reset_index(drop = True, inplace = True)
        
        # build dictionary with filler data
        filler_data = {'target_variable': list(pd.unique(unnest_df_boxplot.loc[:, 'target_variable'])),
                     'actual_coefficient': list(filler_actual_coefficient),
                     'variable': list(variable_column_tmp)}
                     
        # build filler DataFrame
        unnest_df_boxplot_long_tmp = pd.DataFrame(data = filler_data)
        
       
    # append   
    unnest_df_boxplot_long = unnest_df_boxplot_long.append(unnest_df_boxplot_long_tmp, 
                                   ignore_index=True,
                                   sort = False) 

# deep copy    
boxplot_quantiles = unnest_df_boxplot_long.copy()           
    
# combine the strings of 2 columns into 1 column
boxplot_quantiles['var_name_forest'] = boxplot_quantiles.loc[:, 'variable'].str.cat(boxplot_quantiles.loc[:, 'target_variable'], sep = '_')

# create column by computing the absolute value of the actual coefficients
boxplot_quantiles['not_rescaled_coeff'] = boxplot_quantiles.loc[:, 'actual_coefficient']

# initialize DataFrame
neg_pos_bar_plot_rescaled = pd.DataFrame()

# subtract control's coefficient and rescale variant's coefficient, accordingly
for index_02, var_name_02 in enumerate(pd.unique(boxplot_quantiles.loc[:, 'variable'])):

    #sub-set
    mask_feature_importance_raw_boxplot = boxplot_quantiles.loc[:, 'variable'].eq(var_name_02)
    feature_importance_raw_tmp_boxplot = boxplot_quantiles.loc[mask_feature_importance_raw_boxplot, :].reset_index(drop = True)
    
    # find control's coefficient and variant's coefficient
    control_tmp_boxplot = feature_importance_raw_tmp_boxplot.loc[feature_importance_raw_tmp_boxplot.loc[:, 'target_variable'].eq('control'), :].reset_index(drop = True)
    variant_tmp_boxplot = feature_importance_raw_tmp_boxplot.loc[feature_importance_raw_tmp_boxplot.loc[:, 'target_variable'].ne('control'), :].reset_index(drop = True)
    
    # select the median of control's coefficient
    control_tmp_boxplot_median_value = control_tmp_boxplot.loc[:, 'actual_coefficient'].copy()
    # control_tmp_boxplot_median_value = pd.Series(control_tmp_boxplot.loc[control_tmp_boxplot.loc[:, 'quantiles'].eq('hdi_50%'), 'actual_coefficient'])
    
    # broadcast single value (otherwise subtraction -see later- does not work)
    control_tmp_boxplot_median = control_tmp_boxplot_median_value.repeat(variant_tmp_boxplot.shape[0])
    
    # re-set index of Pandas Series
    control_tmp_boxplot_median.reset_index(drop = True, inplace = True) 
    
    # re-scaling (i.e. subtraction) of variant's coefficient vs median of control's coefficient
    variant_tmp_boxplot['coef_sub_value'] = variant_tmp_boxplot.loc[:, 'not_rescaled_coeff'].sub(control_tmp_boxplot_median.abs())
    
    # select the median of variant's coefficient
    variant_tmp_boxplot_subset = variant_tmp_boxplot.loc[:, :].reset_index(drop = True)
    # variant_tmp_boxplot_subset = variant_tmp_boxplot.loc[variant_tmp_boxplot.loc[:, 'quantiles'].eq('hdi_50%'), :].reset_index(drop = True)
        
    # append
    neg_pos_bar_plot_rescaled = neg_pos_bar_plot_rescaled.append(variant_tmp_boxplot_subset, 
                                   ignore_index = True, sort = False)

# add a small '0.0001' in order to avoid the plot looks empty
neg_pos_bar_plot_rescaled.loc[:, 'coef_sub_value'] = neg_pos_bar_plot_rescaled.loc[:, 'coef_sub_value'] + 0.0001

# emphasize more positive (namely, earning money) results by multiplying them by 3
# and dividing the others by 10
for index_03 in range(0, neg_pos_bar_plot_rescaled.shape[0]):
    
    if neg_pos_bar_plot_rescaled.loc[index_03, 'coef_sub_value'] > 0:
       
        neg_pos_bar_plot_rescaled.loc[index_03, 'coef_sub_value'] = neg_pos_bar_plot_rescaled.loc[index_03, 'coef_sub_value']*2
        
    else: 
        neg_pos_bar_plot_rescaled.loc[index_03, 'coef_sub_value'] = neg_pos_bar_plot_rescaled.loc[index_03, 'coef_sub_value']/10
        
 
# deep copy
feat_imp_absolute_value_tmp = feature_importance.copy()

# drop useless columns
feat_imp_absolute_value_tmp.drop(labels = ['target_variable', 
                                           'actual_coefficient', 
                                           'not_rescaled_coeff'], 
                                 axis = 1, 
                                 inplace = True)

# sum the coefficients for each variable
feat_imp_absolute_value = feat_imp_absolute_value_tmp.groupby(by = ['variable']).sum()

# re-set index
feat_imp_absolute_value.reset_index(drop = False, inplace = True)

# re-scale coefficients: now the range is from 0 to 1
feat_imp_absolute_value.loc[:, 'coef_abs_value'] = MinMaxScaler().fit_transform(feat_imp_absolute_value.loc[:, 'coef_abs_value'].to_numpy().reshape(-1, 1))

# sort coefficients by descending order
feat_imp_absolute_value.sort_values(by = ['coef_abs_value'], 
                                    axis = 0, 
                                    inplace = True, 
                                    ascending = False) 

# save always positive coefficients as a .csv file
feat_imp_absolute_value.to_csv(os.path.sep.join([BASE_DIR_OUTPUT,
                                       output_file_name_21]),
                                    index=False)


###############################################################################
## 5. DATA MANIPULATION OF "PERMUTATION_IMPORTANCE" VALUES
# convert Numpy array to Pandas Series
feat_imp_values_series_sklearn  = pd.DataFrame(result_importance_sklearn.importances_mean)

# horizontally concatenate
feature_importance_sklearn_tmp = pd.concat([feat_imp_values_series_sklearn, 
                                feature_name_list], 
                              axis = 1, ignore_index = True, 
                              sort = False).reset_index(drop = True)

# rename columns
feature_importance_sklearn_tmp.rename(columns = {0 : 'actual_coefficient', 
                                     1 : 'variable', 
                                     2 : 'first_level_var', 
                                     3 : 'second_level_var'}, inplace = True)

# invert columns' order: 
feature_importance_sklearn = feature_importance_sklearn_tmp.iloc[:, ::-1]

# re-set index
feature_importance_sklearn.reset_index(drop = True, inplace = True)

# sort coefficients by descending order
feature_importance_sklearn.sort_values(by = ['actual_coefficient'], 
                                    axis = 0, 
                                    inplace = True, 
                                    ascending = False) 

# save "permutation_important" values as a .csv file
feature_importance_sklearn.to_csv(os.path.sep.join([BASE_DIR_OUTPUT,
                                       output_file_name_30]),
                                    index=False)


###########################################################################
## 6. PLOT FEATURE IMPORTANCE
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
          'test13': 'hotpink', 
          'test14': 'mediumpurple'}

# force plot to show the variants according to a specific order 
hue_order = variant_df.loc[1:, 'variant_description'] 

# save raw data as a .csv file
feature_importance.to_csv(os.path.sep.join([BASE_DIR_OUTPUT,
                                        output_file_name_20]),
                                    index=False)

###############################################################################
## THE WHOLE SECTION HAS BEEN SWITCHED OFF BECAUSE ITS CONFUSING HAVING
## ONLY POSITIVE COEFFICIENT
# # plot feature importance
# sns.set(font_scale = 0.5)
# pdf_01 = sns.catplot(y = 'variable', 
#                      x = 'coef_abs_value', 
#                      hue = 'target_variable',
#                      hue_order = hue_order,
#                      data = feature_importance, 
#                      kind = 'bar', 
#                      orient = 'h',
#                      palette = colors_variant_id)
# plt.title('758_grad_boost: stacked feature importance', fontsize=10)
# plt.ylabel('variables/features', fontsize=5)
# plt.xlabel('statistical significance (by standardizing Shapely values)', fontsize=5)
# plt.tight_layout()

# save quantiles which have to be fed into Seaborn
neg_pos_bar_plot_rescaled.to_csv(os.path.sep.join([BASE_DIR_OUTPUT,
                                        output_file_name_24]),
                                    index = False)

# plot bar plot by Seaborn
sns.set(font_scale = 0.5)
pdf_03 = sns.catplot(y = 'var_name_forest', 
                     x = 'coef_sub_value', 
                     hue = 'target_variable', 
                     hue_order = hue_order,
                     data = neg_pos_bar_plot_rescaled, 
                     kind = 'bar', 
                     orient = 'h',
                     palette = colors_variant_id)
plt.title('758_grad_boost: negative/positive bar plot feature importance', fontsize=10)
plt.ylabel('variables/features', fontsize=5)
plt.xlabel('statistical significance (by standardizing Shapely values)', fontsize=5)
plt.tight_layout()

# save plot as .pdf file
plt.savefig(os.path.sep.join([BASE_DIR_OUTPUT, output_file_name_25]))

# close pic in order to avoid overwriting with previous pics
plt.clf()

# force plot to show the variants according to a specific order + include "control"
hue_order_with_control = variant_df.loc[:, 'variant_description'] 

# save quantiles which have to be fed into Seaborn
feature_importance_raw.to_csv(os.path.sep.join([BASE_DIR_OUTPUT,
                                        output_file_name_28]),
                                    index = False)

# plot bar plot by Seaborn
sns.set(font_scale = 0.5)
pdf_04 = sns.catplot(y = 'var_name_forest', 
                     x = 'actual_coefficient', 
                     hue = 'target_variable', 
                     hue_order = hue_order_with_control,
                     data = feature_importance_raw, 
                     kind = 'bar', 
                     orient = 'h',
                     palette = colors_variant_id)
plt.title('758_grad_boost: positive only bar plot feature importance', fontsize=10)
plt.ylabel('variables/features', fontsize=5)
plt.xlabel('Shapely values)', fontsize=5)
plt.tight_layout()

# save plot as .pdf file
plt.savefig(os.path.sep.join([BASE_DIR_OUTPUT, output_file_name_29]))

# close pic in order to avoid overwriting with previous pics
plt.clf()

# # save plot as .pdf file
# pdf_01.savefig(os.path.sep.join([BASE_DIR_OUTPUT, output_file_name_15]))

# # close pic in order to avoid overwriting with previous pics
# plt.clf()


# # TEST: visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
# shap.force_plot(explainer.expected_value, shap_values[0,:], X.iloc[0,:])

## ALTERNATIVE PLOTTING BY USING THE ACTUAL SHAP LIBRARY
# plot Shapely values
# WARNING: convert Pandas DataFrame to Numpy, otherwise it breaks!
pdf_02 = shap.summary_plot(result_importance, 
                        features=X.to_numpy(), 
                        feature_names=feature_name_variables.to_numpy(),
                        class_names=variant_df.loc[:, 'variant_description'],
                        plot_type='bar',
                        show=False)
plt.title('758_grad_boost: feature importance (original Shapely values)', fontsize=10)
plt.ylabel('variables/features', fontsize=5)
plt.xlabel('statistical significance (by Shapely values)', fontsize=5)
plt.tight_layout()

# save plot as .pdf file 
plt.savefig(os.path.sep.join([BASE_DIR_OUTPUT, output_file_name_19]))

# close pic in order to avoid overwriting with previous pics
plt.clf()

# # TEST: adding "dependence plots" in order to check the direction of the effect
# # for the significant variable and singificant variant
# shap.dependence_plot("city_0", result_importance[1], X)


# plot feature permutation_importance
sns.set(font_scale = 0.5)
pdf_05 = sns.catplot(y = 'variable', 
                     x = 'actual_coefficient', 
                     data = feature_importance_sklearn, 
                     kind = 'bar', 
                     orient = 'h')
plt.title('758_grad_boost: feature permutation_importance', fontsize=10)
plt.ylabel('variables/features', fontsize=5)
plt.xlabel('statistical significance (by standardizing Impurity values)', fontsize=5)
plt.tight_layout()

# save plot as .pdf file
pdf_05.savefig(os.path.sep.join([BASE_DIR_OUTPUT, output_file_name_31]))

# close pic in order to avoid overwriting with previous pics
plt.clf()


###########################################################################
## 6. GENERATE AUTOMATICALLY INTERPRETATION OF FEATURE IMPORTANCE PLOT     
# builds several masks in order to sub-set feature importance matrix
mask_percentage_watched = neg_pos_bar_plot_rescaled.loc[:, 'variable'].str.contains('percentage')
mask_selection_order = neg_pos_bar_plot_rescaled.loc[:, 'variable'].str.contains('selection')
mask_gender = neg_pos_bar_plot_rescaled.loc[:, 'variable'].str.contains('gender')
mask_audience = neg_pos_bar_plot_rescaled.loc[:, 'variable'].str.contains('audience')
mask_browser = neg_pos_bar_plot_rescaled.loc[:, 'variable'].str.contains('browse_')
mask_city = neg_pos_bar_plot_rescaled.loc[:, 'variable'].str.contains('city')
mask_device = neg_pos_bar_plot_rescaled.loc[:, 'variable'].str.contains('device')

# build separated a sub-set according to the variable's name
percentage_watched_subset = neg_pos_bar_plot_rescaled.loc[mask_percentage_watched, :].reset_index(drop = True)
selection_order_subset = neg_pos_bar_plot_rescaled.loc[mask_selection_order, :].reset_index(drop = True)
gender_subset = neg_pos_bar_plot_rescaled.loc[mask_gender, :].reset_index(drop = True)
experiment_audience_type_subset = neg_pos_bar_plot_rescaled.loc[mask_audience, :].reset_index(drop = True)
browser_subset = neg_pos_bar_plot_rescaled.loc[mask_browser, :].reset_index(drop = True)
city_subset = neg_pos_bar_plot_rescaled.loc[mask_city, :].reset_index(drop = True)
device_subset = neg_pos_bar_plot_rescaled.loc[mask_device, :].reset_index(drop = True)

# initialize DataFrame
percentage_watched_string = pd.DataFrame()
selection_order_string = pd.DataFrame()
gender_string = pd.DataFrame()
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
            if percentage_watched_subset_tmp.loc[items_11, 'coef_sub_value'] > 0.1:
                
                #building result's interpretation string
                percentage_watched_tmp = pd.Series(('01. {}, GOOD: video {} is very interesting since it is statistically significant for such a variable!'.format(
                    video_type_10, target_var_11))).astype('string')
                
            else: 
                pass
            
            # append
            percentage_watched_string = percentage_watched_string.append(percentage_watched_tmp, 
                                           ignore_index = True, sort = False)
else: 
    pass    


## 'selection_order'
# check whether 'selection_order' variable exists
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
            if selection_order_subset_tmp.loc[items_15, 'coef_sub_value'] > 0.1:
                              
                #building result's interpretation string
                selection_order_tmp = pd.Series(('02. {} , GOOD: video {} is very interesting since it is statistically significant for such a variable!'.format(
                     video_type_14, target_var_15))).astype('string')
                                
            else: 
                pass
            
            # append
            selection_order_string = selection_order_string.append(selection_order_tmp, 
                                           ignore_index = True, sort = False)
else: 
    pass


## 'gender'
# check whether 'gender' variable exists
if (not gender_subset.empty):

    # increase granularity 
    for items_17, video_type_17 in enumerate(pd.unique(gender_subset.loc[:, 'variable'])):
        
        #sub-set
        mask_gender_tmp = gender_subset.loc[:, 'variable'].eq(video_type_17)
        gender_subset_tmp = gender_subset.loc[mask_gender_tmp, :].reset_index(drop = True)
           
        for items_18, target_var_18 in enumerate(gender_subset_tmp.loc[:, 'target_variable']):
    
            # initialize Series
            gender_tmp = []    
            
            # build a meaningful string for each significant video (control included)
            if gender_subset_tmp.loc[items_17, 'coef_sub_value'] > 0.1:
                              
                #building result's interpretation string
                gender_tmp = pd.Series(('03. {} , GOOD: video {} is very interesting since it is statistically significant for such a variable!'.format(
                     video_type_17, target_var_18))).astype('string')
                                
            else: 
                pass
            
            # append
            gender_string = gender_string.append(gender_tmp, 
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
            if audience_subset_tmp.loc[items_05, 'coef_sub_value'] > 0.1:
                               
                #building result's interpretation string
                audience_tmp = pd.Series(('04. {}, GOOD: video {} is very interesting since it is statistically significant for such a variable!'.format(
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
            if browser_subset_tmp.loc[items_03, 'coef_sub_value'] > 0.1:
                              
                #building result's interpretation string
                browser_tmp = pd.Series(('05. {}, GOOD: video {} is very interesting since it is statistically significant for such a variable!'.format(
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
            if city_subset_tmp.loc[items_07, 'coef_sub_value'] > 0.1:
                
                #building result's interpretation string
                city_tmp = pd.Series(('06. {}, GOOD: video {} is very interesting since it is statistically significant for such a variable!'.format(
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
            if device_subset_tmp.loc[items_09, 'coef_sub_value'] > 0.1:
                
                
                #building result's interpretation string              
                device_tmp = pd.Series(('07. {}, GOOD: video {} is very interesting since it is statistically significant for such a variable!'.format(
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
                                   'selection_order_string', 
                                   'gender_string',
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
## 7. CONVERT AUTOMATED INTERPRETATION TO MARKDOWN FILE                         
# MarkDown: initialize a MarkDown file
mdFile = MdUtils(file_name=os.path.sep.join([BASE_DIR_OUTPUT, output_file_name_13]), 
                  title='Automated interpretation') 

# MarkDown: add new header
mdFile.new_header(level=3, 
                  title="Short description")

# create variable tracking the number of variables analyzed 
num_of_variables = (selected_variables.shape[0]) 

# MarkDown: add new paragraph
mdFile.new_paragraph("A ONE-HOT-ENCODING procedure has been used in order to provide\n"
                     "increased granularity in the interpretation of the variables. In other\n"
                     "words, one-hot-encoding was used with the purpose of investigating which\n"
                     "variables item determines more strongly the choice of a specific video as\n"
                     "the best one.")

# MarkDown: add empty line
mdFile.new_line()

# MarkDown: add new header
mdFile.new_header(level=3, 
                  title="Interpretation of the feature importance plot by variable name: significant/not significant interactions")
      
# MarkDown: add list of strings (each string is an interpreation for a specific
# variable)
mdFile.new_list(items=items_interpretation, 
                marked_with='1') 

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
            'ModelOutputResult': str('The "GRADIENT BOOSTING PLOTS" are ready!')}
headers = {
  'Content-Type': 'application/json'
}

# check the machine you are using and send notification only when on server
if RELEASE == LOCAL_COMPUTER: # Linux laptop
    pass
elif url.empty: 
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

                      




  

