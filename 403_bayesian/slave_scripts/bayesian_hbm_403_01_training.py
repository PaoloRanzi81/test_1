

"""
TITLE: "Hierarchical Bayesian Logistic Regression for predicting 
four different video trailers"
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
import pymc3 as pm
import theano
import joblib
from multiprocessing import cpu_count
from sklearn.preprocessing import LabelEncoder
import arviz as az
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
LOCAL_COMPUTER ='5.4.0-1036-gcp'
# LOCAL_COMPUTER ='4.9.0-14-amd64'
# LOCAL_COMPUTER = '5.4.0-65-generic'


# # TEST:
# if RELEASE == LOCAL_COMPUTER: # Linux laptop
#     BASE_DIR_INPUT = ('/home/paolo/Dropbox/Univ/scripts/python_scripts/tensorflow/wright_keith/403_bayesian_hbm_20201201')
#     BASE_DIR_OUTPUT = BASE_DIR_INPUT
#     INPUT_FOLDER = BASE_DIR_INPUT


# set input/output file names
input_file_name_01 = ('input/user_activities_data_set_train.csv')
input_file_name_03 = ('input/dictionary_variants_id.csv')
input_file_name_05 = ('input/config/run_modality.csv')
input_file_name_06 = ('input/user_activities_data_set_test.csv')
input_file_name_07 = ('input/config/url_for_pushing_notification.csv') 
output_file_name_02 = ('output/403_analysis_saved_trace')
output_file_name_11 = ('output/403_analysis_feature_name_list.csv')
output_file_name_12 = ('output/403_analysis_mcc_score.csv')  
output_file_name_13 = ('output/403_feat_imp_interpretation.md') 
output_file_name_14 = ('output/403_columns_names.csv') 
output_file_name_15 = ('output/403_analysis_feature_importance.pdf') 
output_file_name_19 = ('output/403_analysis_feature_importance_arviz_original.pdf') 
output_file_name_21 = ('output/403_analysis_trace_model.joblib')
output_file_name_22 = ('output/403_analysis_trace.h5')
output_file_name_23 = ('output/403_bayesian_summary_statistics.csv')
output_file_name_24 = ('output/403_feature_importance_boxplot.csv')
output_file_name_25 = ('output/403_feature_importance_raw_data.csv')
output_file_name_26 = ('output/403_feature_importance_absolute_value.csv') 
output_file_name_27 = ('output/403_tracking_model_version.csv')
output_file_name_28 = ('output/403_feature_importance_boxplot.pdf')
output_file_name_29 = ('output/403_total_training_time.csv')
output_file_name_30 = ('output/403_posterior_mcmc_traces_cont') 
output_file_name_31 = ('output/403_posterior_mcmc_traces_cat') 


###############################################################################
## 3. LOADING DATA-SET 
# start clocking time
start_time = time.time()

# loading the .csv files with raw data 
user_activities_tmp = pd.read_csv(os.path.sep.join([INPUT_FOLDER, 
                                         input_file_name_01]), header = 0)
    
variant_df = pd.read_csv(os.path.sep.join([INPUT_FOLDER, 
                                         input_file_name_03]), header = 0)
 
# run_modality, either: 
# - 'testing' : without grid-search => quick; 
# - 'production' : with grid-search => slow;
run_modality = pd.read_csv(os.path.sep.join([INPUT_FOLDER, 
                                         input_file_name_05]), header = None, 
                                     dtype = 'str')

url = pd.read_csv(os.path.sep.join([INPUT_FOLDER, 
                                         input_file_name_07]), header = None, 
                                     dtype = 'str')

selected_variables = pd.Series(['percentage_watched',
                                'gender', 
                                'selection_order', 
                                'experiment_audience_type', 
                                'browser', 
                                'city', 
                                'device_type']).rename('selected_variables')


################################################################################
## 4. PRE-PROCESSING
# save  folder name since it contains the model version (e.g. "906_logistic_regression_20200731")
folder_name = pd.Series(os.getcwd()).rename('model_version_in_the_folder_name')

# save one-hot-encoded column's names as .csv file
folder_name.to_csv(os.path.sep.join([BASE_DIR_OUTPUT, 
                                     output_file_name_27]), index = False)

# sample from big data-set in case submissions are > 350
if pd.unique(user_activities_tmp.loc[:, 'session_id']).shape[0] > 350: 
    
    # list of unique session-id
    session_id_series = pd.Series(pd.unique(user_activities_tmp.loc[:, 'session_id']))
    
    # sample the session-id
    sampled_session_id = session_id_series.sample(n = 700).reset_index(drop = True).rename('session_id')
    
    # sub-setting by using sampled session-id
    user_activities = pd.merge(user_activities_tmp, sampled_session_id, how ='right', 
                     on = 'session_id') 
else:
    # deep copy (=> no data loss) in submissions <= 350 
    user_activities = user_activities_tmp.copy()     

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

# concatenate
user_activities_concat_05 = pd.concat([user_activities_concat_04.loc[:, 'best_video'],
                                        user_activities_concat_04.loc[:, 'percentage_watched'],
                                        user_activities_concat_04.loc[:, 'selection_order'],
                                        user_activities_concat_04.loc[:, 'gender'],
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

# convert one-hot-encoded columns to Pandas Series
columns_names = pd.Series(variables_to_be_used).rename('columns_names')

# save one-hot-encoded column's names as .csv file
columns_names.to_csv(os.path.sep.join([BASE_DIR_OUTPUT, 
                                     output_file_name_14]), index = False)


##############################################################################
## 5. BUILDING SMALL TABLE OF FEATURES' NAMES FOR PLOTTING: The table will
## become useful later when the feature importance will be plot 

# Pandas Series with a list of variables' names + the specific and unique items within each variable
gender_types = pd.Series(pd.unique(user_activities_concat_04.loc[:,'gender_original'])).rename('gender_types')
selection_order_types = pd.Series(pd.unique(user_activities_concat_04.loc[:,'selection_order'])).rename('selection_order_types')
browser_types = pd.Series(pd.unique(user_activities_concat_04.loc[:,'browser_original'])).rename('browser_types')
experiment_audience_types = pd.Series(pd.unique(user_activities_concat_04.loc[:,'experiment_audience_type_original'])).rename('experiment_audience_types')
city_types = pd.Series(pd.unique(user_activities_concat_04.loc[:,'city_original'])).rename('city_types')
device_types = pd.Series(pd.unique(user_activities_concat_04.loc[:,'device_type_original'])).rename('device_types')

# sorting the variables names by an alphabetical order
gender_types.sort_values(axis = 0, inplace = True, 
                     ascending = True)
selection_order_types.sort_values(axis = 0, inplace = True, 
                     ascending = True)
browser_types.sort_values(axis = 0, inplace = True, 
                     ascending = True)
experiment_audience_types.sort_values(axis = 0, inplace = True, 
                     ascending = True) 
city_types.sort_values(axis = 0, inplace = True, 
                     ascending = True)
device_types.sort_values(axis = 0, inplace = True, 
                     ascending = True)  

# re-set index
gender_types.reset_index(drop = True, inplace = True)
selection_order_types.reset_index(drop = True, inplace = True)
browser_types.reset_index(drop = True, inplace = True) 
experiment_audience_types.reset_index(drop = True, inplace = True) 
city_types.reset_index(drop = True, inplace = True) 
device_types.reset_index(drop = True, inplace = True) 

# initialize Pandas Series
gender_types_string = pd.DataFrame()
selection_order_types_string = pd.DataFrame()
browser_types_string = pd.DataFrame()
experiment_audience_types_string = pd.DataFrame()
city_types_string = pd.DataFrame()
device_types_string = pd.DataFrame()


# check whether 'gender' variable exists
if (not selected_variables[selected_variables.str.contains('gender')].empty):

    # check whether the variable under scrutiny is NOT a hierachical variable
    if (hierarchical_variable != ['gender']):
        
        # for loop for building a list variables' names by combining to strings for 'gender'
        for items_07, gender_type in enumerate(pd.unique(gender_types)):
            
            # initialize Pandas Series + avoid Pandas' warning
            gender_tmp = []
                           
            # initialize Pandas Series + avoid Pandas' warning
            gender_selected_name = []
            
            # select the variable name
            gender_selected_name = selected_variables[selected_variables.str.contains('gender')].reset_index(drop = True)
            
            # build Pandas DataFrame
            gender_first_level = pd.DataFrame(pd.Series(gender_selected_name[0][0:6]), columns = ['first_level_var'])
            
            # build Pandas DataFrame
            gender_second_level = pd.DataFrame(pd.Series(gender_type), columns = ['second_level_var'])
                           
            # build a meaningful string
            gender_feat_tmp = pd.Series(('{}_{}'.format(gender_selected_name[0][0:6],
                                                gender_second_level.iloc[0,0]))).astype('string')
            
            # build a meaningful string
            gender_tmp = pd.concat([gender_feat_tmp,
                                         gender_first_level,
                                         gender_second_level], 
                              axis = 1, ignore_index = True, 
                              sort = False).reset_index(drop = True)
                                             
            # append
            gender_types_string = gender_types_string.append(gender_tmp, 
                                                   ignore_index = True, sort = False)
    else:
        # if the variable under scrutiny is a hierachical variable, 
        #build an empty Pandas DataFram
        gender_types_string = pd.DataFrame()
else:
    gender_types_string = pd.DataFrame()   

  
# check whether 'selection_order' variable exists
if (not selected_variables[selected_variables.str.contains('selection_order')].empty):    
    # check whether the variable under scrutiny is NOT a hierachical variable
    if (hierarchical_variable != ['selection_order']):
        
        # for loop for building a list variables' names by combining to strings for 'event_label'
        for items_01, selection_order_type in enumerate(pd.unique(selection_order_types)):
            
            # initialize Pandas Series + avoid Pandas' warning
            selection_order_tmp = []
                           
            # initialize Pandas Series + avoid Pandas' warning
            selection_order_selected_name = []
            
            # select the variable name
            selection_order_selected_name = selected_variables[selected_variables.str.contains('selection_order')].reset_index(drop = True)
            
            # build Pandas DataFrame
            selection_order_first_level = pd.DataFrame(pd.Series(selection_order_selected_name[0][0:15]), columns = ['first_level_var'])
            
            # build Pandas DataFrame
            selection_order_second_level = pd.DataFrame(pd.Series(selection_order_type), columns = ['second_level_var'])
            
            # convert integers for 'event_action' into percentages
            selection_order_second_level.replace(to_replace = replace_order_by_strings, 
                inplace = True)
                           
            # build a meaningful string
            selection_order_feat_tmp = pd.Series(('{}_{}'.format(selection_order_selected_name[0][0:15],
                                                selection_order_second_level.iloc[0,0]))).astype('string').rename('feature_name') 
                            
            # build a meaningful string
            selection_order_tmp = pd.concat([selection_order_feat_tmp,
                                             selection_order_first_level,
                                             selection_order_second_level], 
                              axis = 1, ignore_index = True, 
                              sort = False).reset_index(drop = True)         
                             
            # append
            selection_order_types_string = selection_order_types_string.append(selection_order_tmp, 
                                                   ignore_index = True, sort = False)
    
    else:
        # if the variable under scrutiny is a hierachical variable, 
        #build an empty Pandas DataFram
        selection_order_types_string = pd.DataFrame()
else:
    selection_order_types_string = pd.DataFrame()  
    
    
# check whether 'experiment_audience_type' variable exists
if (not selected_variables[selected_variables.str.contains('experiment_audience_type')].empty):
 
    # check whether the variable under scrutiny is NOT a hierachical variable
    if (hierarchical_variable != ['experiment_audience_type']):
        
        # for loop for building a list variables' names by combining to strings for 'experiment_audience_type'
        for items_02, audience_type in enumerate(pd.unique(experiment_audience_types)):
            
            # initialize Pandas Series + avoid Pandas' warning
            audience_tmp = []
                              
            # initialize Pandas Series + avoid Pandas' warning
            audience_selected_name = []
            
            # select the variable name
            audience_selected_name = selected_variables[selected_variables.str.contains('audience')].reset_index(drop = True)
            
            # build Pandas DataFrame
            audience_first_level = pd.DataFrame(pd.Series(audience_selected_name[0][0:24]), columns = ['first_level_var'])
            
            # build Pandas DataFrame
            audience_second_level = pd.DataFrame(pd.Series(audience_type), columns = ['second_level_var'])
                           
            # build a meaningful string
            audience_feat_tmp = pd.Series(('{}_{}'.format(audience_selected_name[0][0:24],
                                                audience_second_level.iloc[0,0]))).astype('string')
            
            # build a meaningful string
            audience_tmp = pd.concat([audience_feat_tmp,
                                      audience_first_level,
                                      audience_second_level], 
                              axis = 1, ignore_index = True, 
                              sort = False).reset_index(drop = True)
                                         
            # append
            experiment_audience_types_string = experiment_audience_types_string.append(audience_tmp, 
                                                   ignore_index = True, sort = False)
    
    else:
        # if the variable under scrutiny is a hierachical variable, 
        #build an empty Pandas DataFram
        experiment_audience_types_string = pd.DataFrame()
else:
    experiment_audience_types_string = pd.DataFrame()  
    
    
# check whether 'browser' variable exists
if (not selected_variables[selected_variables.str.contains('browser')].empty):

    # check whether the variable under scrutiny is NOT a hierachical variable
    if (hierarchical_variable != ['browser']):
        
        # for loop for building a list variables' names by combining to strings for 'browser'
        for items_03, browser_type in enumerate(pd.unique(browser_types)):
            
            # initialize Pandas Series + avoid Pandas' warning
            browser_tmp = []
                
            # initialize Pandas Series + avoid Pandas' warning
            browser_selected_name = []
            
            # select the variable name
            browser_selected_name = selected_variables[selected_variables.str.contains('browser')].reset_index(drop = True)
            
            # build Pandas DataFrame
            browser_first_level = pd.DataFrame(pd.Series(browser_selected_name[0][0:7]), columns = ['first_level_var'])
            
            # build Pandas DataFrame
            browser_second_level = pd.DataFrame(pd.Series(browser_type), columns = ['second_level_var'])
                           
            # build a meaningful string
            browser_feat_tmp = pd.Series(('{}_{}'.format(browser_selected_name[0][0:7],
                                                browser_second_level.iloc[0,0]))).astype('string')
            
            # build a meaningful string
            browser_tmp = pd.concat([browser_feat_tmp,
                                     browser_first_level,
                                     browser_second_level], 
                              axis = 1, ignore_index = True, 
                              sort = False).reset_index(drop = True)
                                   
            # append
            browser_types_string = browser_types_string.append(browser_tmp, 
                                                   ignore_index = True, sort = False)
    
    else:
        # if the variable under scrutiny is a hierachical variable, 
        #build an empty Pandas DataFram
        browser_types_string = pd.DataFrame()
else:
    browser_types_string = pd.DataFrame()

    
# check whether 'city' variable exists
if (not selected_variables[selected_variables.str.contains('city')].empty):   
    # check whether the variable under scrutiny is NOT a hierachical variable
    if (hierarchical_variable != ['city']):    
        
        # for loop for building a list variables' names by combining to strings for 'city'
        for items_04, city_type in enumerate(pd.unique(city_types)):
            
            # initialize Pandas Series + avoid Pandas' warning
            city_tmp = []
                            
            # initialize Pandas Series + avoid Pandas' warning
            city_selected_name = []
            
            # select the variable name
            city_selected_name = selected_variables[selected_variables.str.contains('city')].reset_index(drop = True)
            
            # build Pandas DataFrame
            city_first_level = pd.DataFrame(pd.Series(city_selected_name[0][0:4]), columns = ['first_level_var'])
            
            # build Pandas DataFrame
            city_second_level = pd.DataFrame(pd.Series(city_type), columns = ['second_level_var'])
                           
            # build a meaningful string
            city_feat_tmp = pd.Series(('{}_{}'.format(city_selected_name[0][0:4],
                                                city_second_level.iloc[0,0]))).astype('string')
            
            # build a meaningful string
            city_tmp = pd.concat([city_feat_tmp,
                                  city_first_level,
                                  city_second_level], 
                              axis = 1, ignore_index = True, 
                              sort = False).reset_index(drop = True)
                                       
            # append
            city_types_string = city_types_string.append(city_tmp, 
                                                   ignore_index = True, sort = False)
    else:
        # if the variable under scrutiny is a hierachical variable, 
        #build an empty Pandas DataFram
        city_types_string = pd.DataFrame()
else:
    city_types_string = pd.DataFrame()


# check whether 'device_type' variable exists
if (not selected_variables[selected_variables.str.contains('device')].empty):
    # check whether the variable under scrutiny is NOT a hierachical variable
    if (hierarchical_variable != ['device_type']):        
        
        # for loop for building a list variables' names by combining to strings for 'device_type'
        for items_06, device_type in enumerate(pd.unique(device_types)):
            
            # initialize Pandas Series + avoid Pandas' warning
            device_tmp = []
            
            # initialize Pandas Series + avoid Pandas' warning
            device_selected_name = []
            
            # select the variable name
            device_selected_name = selected_variables[selected_variables.str.contains('device')].reset_index(drop = True)
            
            # build Pandas DataFrame
            device_first_level = pd.DataFrame(pd.Series(device_selected_name[0][0:6]), columns = ['first_level_var'])
            
            # build Pandas DataFrame
            device_second_level = pd.DataFrame(pd.Series(device_type), columns = ['second_level_var'])
                           
            # build a meaningful string
            device_feat_tmp = pd.Series(('{}_{}'.format(device_selected_name[0][0:6],
                                                device_second_level.iloc[0,0]))).astype('string')
            
            # build a meaningful string
            device_tmp = pd.concat([device_feat_tmp,
                                    device_first_level,
                                    device_second_level], 
                              axis = 1, ignore_index = True, 
                              sort = False).reset_index(drop = True)
                                                   
            # append
            device_types_string = device_types_string.append(device_tmp, 
                                                   ignore_index = True, sort = False)
    else:
        # if the variable under scrutiny is a hierachical variable, 
        #build an empty Pandas DataFram
        device_types_string = pd.DataFrame()
else:
    device_types_string = pd.DataFrame()    
    
        
# build an empty DataFrame for 'percentage_watched' since it is a continuos 
# variable
percentage_watched_types_string = pd.DataFrame(data = [pd.Series('percentage_watched'), 
                                                       pd.Series('percentage_watched'),
                                                       pd.Series('percentage_watched')]).transpose()

# concatenate vertically not-baseline variables' names
feature_name_list = pd.concat([percentage_watched_types_string,
                               gender_types_string,                               
                               selection_order_types_string,
                               experiment_audience_types_string,
                               browser_types_string, 
                               city_types_string, 
                               device_types_string], 
                              axis = 0, ignore_index = True, 
                              sort = False).reset_index(drop = True) 

# rename
feature_name_list.rename(columns={0: 'feature_name', 
                                  1: 'first_level_var', 
                                  2: 'second_level_var'}, inplace = True)   

feature_name_list.to_csv(os.path.sep.join([BASE_DIR_OUTPUT, 
                                     output_file_name_11]), index = False)


###############################################################################
## 5. LOAD TRAINING SET
# set shared theano variables
y_data = y.to_numpy('int8')
X_continuos = X.loc[:, variables_to_be_used[0]]
X_categorical_selection = X.loc[:, variables_to_be_used[1]]
X_categorical_gender = X.loc[:, variables_to_be_used[2]]
X_categorical_audience = X.loc[:, variables_to_be_used[3]]
X_categorical_browser = X.loc[:, variables_to_be_used[4]]
X_categorical_city = X.loc[:, variables_to_be_used[5]]
X_categorical_device = X.loc[:, variables_to_be_used[6]]


###############################################################################
## 6. TRAINING HIERARCHICAL BAYESIAN MODEL
# select configuration settings
if run_modality.iloc[0, 0] == 'testing':
       
    chains = 4
    draws = 10
    cores = 4
    tune = 50
    

elif run_modality.iloc[0, 0] == 'production':
    
    # sample from big data-set in case submissions are > 900
    if pd.unique(user_activities_tmp.loc[:, 'session_id']).shape[0] <= 350: 
        
        # number of MCMC chains
        chains = int(round((cpu_count()), 0)) 
        
        # actual number of samples (called also "draws") from the NUTS sampler    
        # ideal: 440000 total draws (here instead we set a total of 102000 draws for speed's sake)
        draws = 1000
    
        # tuning steps for warming up the NUTS sampler. They are also called "burn-in steps". 
        # they are discarded when building the MCMC trace
        tune = int(round((450000/chains - draws), 0))
        # tune = int(round((102000/chains - draws), 0)) # original
        
        # number of vCPUs to be used. Usually, they correspond to the number of vCPUs available
        cores = int(round((cpu_count()), 0))
        
    elif (pd.unique(user_activities_tmp.loc[:, 'session_id']).shape[0] > 350) and (int(round((cpu_count()), 0)) <= 25):
        
        # number of MCMC chains
        chains = int(round((cpu_count()), 0)) 
        
        # actual number of samples (called also "draws") from the NUTS sampler    
        draws = 1000
        # draws = 500
    
        # tuning steps for warming up the NUTS sampler. They are also called "burn-in steps". 
        # they are discarded when building the MCMC trace
        # tune = 500
        tune = int(round((102000/chains - draws), 0)) # original
        
        # number of vCPUs to be used. Usually, they correspond to the number of vCPUs available
        cores = int(round((cpu_count()), 0))
        
    elif (pd.unique(user_activities_tmp.loc[:, 'session_id']).shape[0] > 350) and (int(round((cpu_count()), 0)) >= 50):
        
        # number of MCMC chains
        chains = int(round((cpu_count()), 0)) 
        
        # actual number of samples (called also "draws") from the NUTS sampler    
        draws = 1000
        # draws = 500
    
        # tuning steps for warming up the NUTS sampler. They are also called "burn-in steps". 
        # they are discarded when building the MCMC trace
        # tune = 500
        tune = int(round((102000/chains - draws), 0)) # original
        
        # number of vCPUs to be used. Usually, they correspond to the number of vCPUs available
        cores = int(round((cpu_count()), 0))
            

# build coordinates for ARviz (according to 20200629's tutorial): 
# https://docs.pymc.io/notebooks/multilevel_modeling.html
coords = {
          "X_continuos_column_name": ('percentage_watched'),
          "X_continuos_index": X_continuos.index,
          "X_categorical_selection_column_name": ('selection_order'),
          "X_categorical_selection_index": X_categorical_selection.index,
          "X_categorical_gender_column_name": ('gender'),
          "X_categorical_gender_index": X_categorical_gender.index,
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


# function which initialize the bayesian model
def model_factory(X_continuos, 
                  X_categorical_selection, 
                  X_categorical_gender, 
                  X_categorical_audience,
                  X_categorical_browser,
                  X_categorical_city,
                  X_categorical_device,
                  y_data, 
                  variables_to_be_used, 
                  variant_df):
    
    """Build Bayesian model in PyMC3.
      Args:
        X_...: Pandas Series of each variable
        y_data: target variable/output.
        variables_to_be_used: used for getting the number of columns for
            X_combined. That number informs the shape of the probability 
            distribution.
        variables_df: number of target variable/output (e.g. control + variants).
            That number informs the shape of the probability distribution.    
        
      Returns:
        varying_intercept_slope_noncentered: it is the log-likelihood. 
    """

    with pm.Model(coords=coords) as varying_intercept_slope_noncentered:
        
        # build tensors from Pandas DataFrame/Series              
        X_continuos_var = pm.Data('X_continuos', X_continuos, dims=("X_continuos_index"))
        X_categorical_selection_var = pm.Data('X_categorical_selection', X_categorical_selection, dims=("X_categorical_selection_index"))
        X_categorical_gender_var = pm.Data('X_categorical_gender', X_categorical_gender, dims=("X_categorical_gender_index"))
        X_categorical_audience_var = pm.Data('X_categorical_audience', X_categorical_audience, dims=("X_categorical_audience_index"))
        X_categorical_browser_var = pm.Data('X_categorical_browser', X_categorical_browser, dims=("X_categorical_browser_index"))
        X_categorical_city_var = pm.Data('X_categorical_city', X_categorical_city, dims=("X_categorical_city_index"))
        X_categorical_device_var = pm.Data('X_categorical_device', X_categorical_device, dims=("X_categorical_device_index"))
        
        # hyperpriors for intercept
        mu_alpha_tmp = pm.Laplace('mu_alpha_tmp', mu=0.05, b=1.,
                                  shape=(variant_df.shape[0] - 1))
        mu_alpha = theano.tensor.concatenate([[0], mu_alpha_tmp])

        sigma_alpha_tmp = pm.HalfNormal('sigma_alpha_tmp', sigma=1.,
                                        shape=(variant_df.shape[0] - 1))
        sigma_alpha = theano.tensor.concatenate([[0], sigma_alpha_tmp])
        
        # prior for non-centered random intercepts
        u = pm.Laplace('u', mu=0.05, b=1.)
        
        # random intercept
        alpha_eq = mu_alpha + u*sigma_alpha
        alpha_eq_deter = pm.Deterministic('alpha_eq_deter', alpha_eq)
        alpha = pm.Laplace('alpha', mu=alpha_eq_deter, b=1.,
                              shape=(variant_df.shape[0]))
                
        #######################################################################

        # hyperpriors for slopes (continuos)
        mu_beta_continuos_tmp = pm.Laplace('mu_beta_continuos_tmp', mu=0.05, b=1.,
                                       shape=(1, (variant_df.shape[0] - 2)))
        mu_beta_continuos = theano.tensor.concatenate([np.zeros((1, 1)), mu_beta_continuos_tmp], axis=1)
        sigma_beta_continuos_tmp = pm.HalfNormal('sigma_beta_continuos_tmp', sigma=1.,
                                   shape=(1, (variant_df.shape[0] - 2)))
        sigma_beta_continuos = theano.tensor.concatenate([np.zeros((1, 1)), sigma_beta_continuos_tmp], axis=1)

        # prior for non-centered random slope (continuos)
        g = pm.Laplace('g', mu=0.05, b=1., shape=(1, 1))

        # random slopes (continuos)
        beta_continuos_eq = mu_beta_continuos + pm.math.dot(g, sigma_beta_continuos)
        beta_con_deter_percentage = pm.Deterministic('beta_con_deter_percentage', beta_continuos_eq)
        beta_con_tmp_percentage = pm.Laplace('beta_con_tmp_percentage', mu=beta_con_deter_percentage, b=1.,
                                  shape=(1, (variant_df.shape[0] - 1)))
        beta_con_percentage  = theano.tensor.concatenate([np.zeros((1, 1)), beta_con_tmp_percentage ], axis=1)

        # expected value (continuos)
        dot_product_continuos = pm.math.dot(theano.tensor.shape_padaxis(X_continuos_var, axis=1), beta_con_percentage)
        
        #######################################################################
        
        # hyperpriors for slopes (categorical_selection)
        mu_beta_categorical_selection_tmp = pm.Laplace('mu_beta_categorical_selection_tmp', mu=0.05, b=1.,
                                                 shape=(len(pd.unique(X_categorical_selection)), (variant_df.shape[0] - 2)))
        mu_beta_categorical_selection = theano.tensor.concatenate([np.zeros((len(pd.unique(X_categorical_selection)), 1)), 
                                                                   mu_beta_categorical_selection_tmp], axis=1)
        sigma_beta_categorical_selection_tmp = pm.HalfNormal('sigma_beta_categorical_selection_tmp', sigma=1.,
                                                   shape=(len(pd.unique(X_categorical_selection)), (variant_df.shape[0] - 2)))
        sigma_beta_categorical_selection = theano.tensor.concatenate(
            [np.zeros((len(pd.unique(X_categorical_selection)), 1)), sigma_beta_categorical_selection_tmp], axis=1)

        # prior for non-centered random slope (categorical_selection)
        non_centered_selection = pm.Laplace('non_centered_selection', 
                                           mu=0.05, 
                                           b=1., 
                                           shape=(len(pd.unique(X_categorical_selection)), len(pd.unique(X_categorical_selection))))
           
        #random slopes (categorical_selection)
        beta_categorical_eq_selection = mu_beta_categorical_selection + pm.math.dot(non_centered_selection, sigma_beta_categorical_selection)
        beta_cat_deter_selection = pm.Deterministic('beta_cat_deter_selection', beta_categorical_eq_selection)
        beta_cat_tmp_selection = pm.Laplace('beta_cat_tmp_selection', mu=beta_cat_deter_selection, b=1.,
                                                  shape=(len(pd.unique(X_categorical_selection)), (variant_df.shape[0] - 1)))
        beta_cat_selection = theano.tensor.concatenate([np.zeros((len(pd.unique(X_categorical_selection)), 1)), beta_cat_tmp_selection], axis=1)
                
        #######################################################################
        
        # hyperpriors for slopes (categorical_gender)
        mu_beta_categorical_gender_tmp = pm.Laplace('mu_beta_categorical_gender_tmp', mu=0.05, b=1.,
                                                 shape=(len(pd.unique(X_categorical_gender)), (variant_df.shape[0] - 2)))
        mu_beta_categorical_gender = theano.tensor.concatenate([np.zeros((len(pd.unique(X_categorical_gender)), 1)), 
                                                                   mu_beta_categorical_gender_tmp], axis=1)
        sigma_beta_categorical_gender_tmp = pm.HalfNormal('sigma_beta_categorical_gender_tmp', sigma=1.,
                                                   shape=(len(pd.unique(X_categorical_gender)), (variant_df.shape[0] - 2)))
        sigma_beta_categorical_gender = theano.tensor.concatenate(
            [np.zeros((len(pd.unique(X_categorical_gender)), 1)), sigma_beta_categorical_gender_tmp], axis=1)

        # prior for non-centered random slope (categorical_gender)
        non_centered_gender = pm.Laplace('non_centered_gender', 
                                           mu=0.05, 
                                           b=1., 
                                           shape=(len(pd.unique(X_categorical_gender)), len(pd.unique(X_categorical_gender))))
        
        #random slopes (categorical_gender)
        beta_categorical_eq_gender = mu_beta_categorical_gender + pm.math.dot(non_centered_gender, sigma_beta_categorical_gender)
        beta_cat_deter_gender = pm.Deterministic('beta_cat_deter_gender', beta_categorical_eq_gender)
        beta_cat_tmp_gender = pm.Laplace('beta_cat_tmp_gender', mu=beta_cat_deter_gender, b=1.,
                                                  shape=(len(pd.unique(X_categorical_gender)), (variant_df.shape[0] - 1)))
        beta_cat_gender = theano.tensor.concatenate([np.zeros((len(pd.unique(X_categorical_gender)), 1)), beta_cat_tmp_gender], axis=1)

        # hyperpriors for slopes (categorical_audience)
        mu_beta_categorical_audience_tmp = pm.Laplace('mu_beta_categorical_audience_tmp', mu=0.05, b=1.,
                                                 shape=(len(pd.unique(X_categorical_audience)), (variant_df.shape[0] - 2)))
        mu_beta_categorical_audience = theano.tensor.concatenate([np.zeros((len(pd.unique(X_categorical_audience)), 1)), 
                                                                  mu_beta_categorical_audience_tmp], axis=1)
        sigma_beta_categorical_audience_tmp = pm.HalfNormal('sigma_beta_categorical_audience_tmp', sigma=1.,
                                                   shape=(len(pd.unique(X_categorical_audience)), (variant_df.shape[0] - 2)))
        sigma_beta_categorical_audience = theano.tensor.concatenate(
            [np.zeros((len(pd.unique(X_categorical_audience)), 1)), sigma_beta_categorical_audience_tmp], axis=1)

        # prior for non-centered random slope (categorical_audience)
        non_centered_audience = pm.Laplace('non_centered_audience', 
                                          mu=0.05, 
                                          b=1., 
                                          shape=(len(pd.unique(X_categorical_audience)), len(pd.unique(X_categorical_audience))))
        
        #random slopes (categorical_audience)
        beta_categorical_eq_audience = mu_beta_categorical_audience + pm.math.dot(non_centered_audience, sigma_beta_categorical_audience)
        beta_cat_deter_audience = pm.Deterministic('beta_cat_deter_audience', beta_categorical_eq_audience)
        beta_cat_tmp_audience = pm.Laplace('beta_cat_tmp_audience', mu=beta_cat_deter_audience, b=1.,
                                                  shape=(len(pd.unique(X_categorical_audience)), (variant_df.shape[0] - 1)))
        beta_cat_audience = theano.tensor.concatenate([np.zeros((len(pd.unique(X_categorical_audience)), 1)), beta_cat_tmp_audience], axis=1)
        
        #######################################################################
        
        # hyperpriors for slopes (categorical_browser)
        mu_beta_categorical_browser_tmp = pm.Laplace('mu_beta_categorical_browser_tmp', mu=0.05, b=1.,
                                                 shape=(len(pd.unique(X_categorical_browser)), (variant_df.shape[0] - 2)))
        mu_beta_categorical_browser = theano.tensor.concatenate([np.zeros((len(pd.unique(X_categorical_browser)), 1)), 
                                                                 mu_beta_categorical_browser_tmp], axis=1)
        sigma_beta_categorical_browser_tmp = pm.HalfNormal('sigma_beta_categorical_browser_tmp', sigma=1.,
                                                   shape=(len(pd.unique(X_categorical_browser)), (variant_df.shape[0] - 2)))
        sigma_beta_categorical_browser = theano.tensor.concatenate(
            [np.zeros((len(pd.unique(X_categorical_browser)), 1)), sigma_beta_categorical_browser_tmp], axis=1)

        # prior for non-centered random slope (categorical_browser)
        non_centered_browser = pm.Laplace('non_centered_browser', 
                                       mu=0.05, 
                                       b=1., 
                                       shape=(len(pd.unique(X_categorical_browser)), len(pd.unique(X_categorical_browser))))
        
        #random slopes (categorical_browser)
        beta_categorical_eq_browser = mu_beta_categorical_browser + pm.math.dot(non_centered_browser, sigma_beta_categorical_browser)
        beta_cat_deter_browser = pm.Deterministic('beta_cat_deter_browser', beta_categorical_eq_browser)
        beta_cat_tmp_browser = pm.Laplace('beta_cat_tmp_browser', mu=beta_cat_deter_browser, b=1.,
                                                  shape=(len(pd.unique(X_categorical_browser)), (variant_df.shape[0] - 1)))
        beta_cat_browser = theano.tensor.concatenate([np.zeros((len(pd.unique(X_categorical_browser)), 1)), beta_cat_tmp_browser], axis=1)
        
        #######################################################################
        
        # hyperpriors for slopes (categorical_city)
        mu_beta_categorical_city_tmp = pm.Laplace('mu_beta_categorical_city_tmp', mu=0.05, b=1.,
                                                 shape=(len(pd.unique(X_categorical_city)), (variant_df.shape[0] - 2)))
        mu_beta_categorical_city = theano.tensor.concatenate([np.zeros((len(pd.unique(X_categorical_city)), 1)), mu_beta_categorical_city_tmp], axis=1)
        sigma_beta_categorical_city_tmp = pm.HalfNormal('sigma_beta_categorical_city_tmp', sigma=1.,
                                                   shape=(len(pd.unique(X_categorical_city)), (variant_df.shape[0] - 2)))
        sigma_beta_categorical_city = theano.tensor.concatenate(
            [np.zeros((len(pd.unique(X_categorical_city)), 1)), sigma_beta_categorical_city_tmp], axis=1)

        # prior for non-centered random slope (categorical_city)
        non_centered_city = pm.Laplace('non_centered_city', 
                                    mu=0.05, 
                                    b=1., 
                                    shape=(len(pd.unique(X_categorical_city)), len(pd.unique(X_categorical_city))))
        
        #random slopes (categorical_city)
        beta_categorical_eq_city = mu_beta_categorical_city + pm.math.dot(non_centered_city, sigma_beta_categorical_city)
        beta_cat_deter_city = pm.Deterministic('beta_cat_deter_city', beta_categorical_eq_city)
        beta_cat_tmp_city = pm.Laplace('beta_cat_tmp_city', mu=beta_cat_deter_city, b=1.,
                                                  shape=(len(pd.unique(X_categorical_city)), (variant_df.shape[0] - 1)))
        beta_cat_city = theano.tensor.concatenate([np.zeros((len(pd.unique(X_categorical_city)), 1)), beta_cat_tmp_city], axis=1)
        
        #######################################################################
        
        # hyperpriors for slopes (categorical_device)
        mu_beta_categorical_device_tmp = pm.Laplace('mu_beta_categorical_device_tmp', mu=0.05, b=1.,
                                                 shape=(len(pd.unique(X_categorical_device)), (variant_df.shape[0] - 2)))
        mu_beta_categorical_device = theano.tensor.concatenate([np.zeros((len(pd.unique(X_categorical_device)), 1)), mu_beta_categorical_device_tmp], axis=1)
        sigma_beta_categorical_device_tmp = pm.HalfNormal('sigma_beta_categorical_device_tmp', sigma=1.,
                                                   shape=(len(pd.unique(X_categorical_device)), (variant_df.shape[0] - 2)))
        sigma_beta_categorical_device = theano.tensor.concatenate(
            [np.zeros((len(pd.unique(X_categorical_device)), 1)), sigma_beta_categorical_device_tmp], axis=1)

        # prior for non-centered random slope (categorical_device)
        non_centered_device = pm.Laplace('non_centered_device', 
                                        mu=0.05, 
                                        b=1., 
                                        shape=(len(pd.unique(X_categorical_device)), len(pd.unique(X_categorical_device))))
        
        #random slopes (categorical_device)
        beta_categorical_eq_device = mu_beta_categorical_device + pm.math.dot(non_centered_device, sigma_beta_categorical_device)
        beta_cat_deter_device = pm.Deterministic('beta_cat_deter_device', beta_categorical_eq_device)
        beta_cat_tmp_device = pm.Laplace('beta_cat_tmp_device', mu=beta_cat_deter_device, b=1.,
                                                  shape=(len(pd.unique(X_categorical_device)), (variant_df.shape[0] - 1)))
        beta_cat_device = theano.tensor.concatenate([np.zeros((len(pd.unique(X_categorical_device)), 1)), beta_cat_tmp_device], axis=1)
        # theano.printing.Print('vector', attrs=['shape'])(beta_cat_device)
        
        #######################################################################
        
        # hyperpriors for epsilon      
        sigma_epsilon = pm.HalfNormal('sigma_epsilon', sigma=1., 
                                        shape=(variant_df.shape[0]))
        
        # epsilon
        epsilon = pm.HalfNormal('epsilon', sigma=sigma_epsilon, # not working
                              shape=(variant_df.shape[0]))
        
        #######################################################################
        
        y_hat_tmp = (alpha 
                     + dot_product_continuos
                     + beta_cat_selection[X_categorical_selection_var, :]                     
                     + beta_cat_gender[X_categorical_gender_var, :]
                     + beta_cat_audience[X_categorical_audience_var, :] 
                     + beta_cat_browser[X_categorical_browser_var, :] 
                     + beta_cat_city[X_categorical_city_var, :] 
                     + beta_cat_device[X_categorical_device_var, :]
                     + epsilon)

        # softmax
        y_hat = theano.tensor.nnet.softmax(y_hat_tmp)
        # theano.printing.Print('vector', attrs=['shape'])(y_hat)

        # likelihood
        y_likelihood = pm.Categorical('y_likelihood', p=y_hat, observed=y_data)

    # dump trace model
    joblib.dump(varying_intercept_slope_noncentered, os.path.sep.join([BASE_DIR_OUTPUT, output_file_name_21]))

    return varying_intercept_slope_noncentered


# run MCMC sampling
with model_factory(X_continuos, 
                  X_categorical_selection,
                  X_categorical_gender,
                  X_categorical_audience,
                  X_categorical_browser,
                  X_categorical_city,
                  X_categorical_device,
                  y_data, 
                  variables_to_be_used, 
                  variant_df) as train_model:

    arviz_inference = pm.sample(draws=draws,
                                tune=tune,
                                chains=chains,
                                cores=cores,
                                target_accept=0.95,
                                discard_tuned_samples=True,
                                return_inferencedata=True)


###############################################################################
## 7. PICKLING THE MCMC TRACE BY JOBLIB
# re-load model
varying_intercept_slope_noncentered = joblib.load(os.path.sep.join([BASE_DIR_INPUT, output_file_name_21]))
  
# save the trace by Arviz as .hdf5 (huge file)
with varying_intercept_slope_noncentered:
    az.to_netcdf(data=arviz_inference,
                 filename=os.path.sep.join([BASE_DIR_OUTPUT, output_file_name_22]))


# end time according to computer clock
end_time = time.time()

# calculate total execution time
total_execution_time = pd.Series(np.round((end_time - start_time), 2)).rename('total_training_runtime_seconds')

# save folder's name as .csv file
total_execution_time.to_csv(os.path.sep.join([BASE_DIR_OUTPUT, 
                                     output_file_name_29]), index = False)

# shows run-time's timestamps + total execution time
print('start time (unix timestamp):{}'.format(start_time))
print('end time (unix timestamp):{}'.format(end_time))
print('total execution time (seconds):{}'.format(total_execution_time.iloc[0]))

# configure notification sending to server when the script has finishes
# url ="test_fake" # TEST
payload = {'ModelName': str(folder_name[0]), 
           'ModelOutputResult': str('The "BAYESIAN MODEL" training script has finished!')}
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






 

