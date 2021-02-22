"""
TITLE: "Compute MCC score for the bayesian model"
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
from sklearn.metrics import matthews_corrcoef, precision_score, recall_score, accuracy_score
import arviz as az
import time
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
#     BASE_DIR_INPUT = ('/home/paolo/Dropbox/Univ/scripts/python_scripts/tensorflow/wright_keith/801_bayesian_hbm_20201102')
#     BASE_DIR_OUTPUT = BASE_DIR_INPUT
#     INPUT_FOLDER = BASE_DIR_INPUT


# set input/output file names
input_file_name_03 = ('input/dictionary_variants_id.csv')
input_file_name_05 = ('input/config/run_modality.csv')
input_file_name_06 = ('input/user_activities_data_set_test.csv')
output_file_name_02 = ('output/403_analysis_saved_trace')
output_file_name_12 = ('output/403_analysis_mcc_score.csv')
output_file_name_21 = ('output/403_analysis_trace_model.joblib')
output_file_name_22 = ('output/403_analysis_trace.h5')
output_file_name_32 = ('output/403_variational_inference_saved_trace.joblib')


###############################################################################
## 3. LOADING DATA-SET 
# setting testing mode 
RELEASE = platform.release()

# start clocking time
start_time = time.time()

# loading the .csv files with raw data 
user_activities = pd.read_csv(os.path.sep.join([INPUT_FOLDER, 
                                         input_file_name_06]), header = 0)
    
variant_df = pd.read_csv(os.path.sep.join([INPUT_FOLDER, 
                                         input_file_name_03]), header = 0)

# run_modality, either: 
# - 'testing' : without grid-search => quick; 
# - 'production' : with grid-search => slow;
run_modality = pd.read_csv(os.path.sep.join([INPUT_FOLDER, 
                                         input_file_name_05]), header = None, 
                                     dtype = 'str')

# load model
varying_intercept_slope_noncentered = joblib.load(os.path.sep.join([BASE_DIR_INPUT, output_file_name_21]))

# load ArviZ NetCDF data-set containig MCMC samples
arviz_inference = az.from_netcdf(filename=os.path.sep.join([BASE_DIR_OUTPUT, output_file_name_22]))


################################################################################
## 4. PRE-PROCESSING
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
y_tmp = user_activities_concat_05.pop('best_video')

# rename column
y = y_tmp.rename('truth')

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
## 4. LOAD OUT-OF-BAG SET
# Build hold-out-set (Pandas Series)
y_data_1 = pd.Series(np.zeros(y.shape[0])).rename('y_data')
y_data = y_data_1.to_numpy(dtype="int")
X_continuos = X.loc[:, variables_to_be_used[0]]
X_categorical_selection = X.loc[:, variables_to_be_used[1]]
X_categorical_gender = X.loc[:, variables_to_be_used[2]]
X_categorical_audience = X.loc[:, variables_to_be_used[3]]
X_categorical_browser = X.loc[:, variables_to_be_used[4]]
X_categorical_city = X.loc[:, variables_to_be_used[5]]
X_categorical_device = X.loc[:, variables_to_be_used[6]]


###############################################################################
## 5. RE-BUILD HIERARCHICAL BAYESIAN MODEL
# select configuration settings
if run_modality.iloc[0, 0] == 'testing':

    # samples from the posterior distribution
    samples = 100

    # number of bootstrap iterations
    n_iterations = 3

    # set 'parallel' = False only for testing on the local computer
    parallel = False

elif run_modality.iloc[0, 0] == 'production':

    # samples from the posterior distribution
    samples = 1000

    # # ORIGINAL: number of bootstrap iterations
    # n_iterations = 30

    # WORKAROUND: forcing a reduction of bootstrap iterations
    n_iterations = 5
    
    # WARNING since conflict between 'joblib' + 'Theano' + 'mkl' library, 
    # the code id forced to work by single-core only.
    # ORIGINAL: enabling multi-threading
    #parallel = True
    
    # WORKAROUND: forcing single-core
    parallel = False
    
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


# function for building the bayesian model + predictions by using such a model
def model_factory(X_continuos, 
                  X_categorical_selection,
                  X_categorical_gender,
                  X_categorical_audience,
                  X_categorical_browser,
                  X_categorical_city,
                  X_categorical_device,
                  y_data, 
                  variables_to_be_used,
                  variant_df,
                  arviz_inference,
                  samples):
    
    """ please check run_model_oob's function docstring below for a description  
        of the inputs.
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

        # predicting new values from the posterior distribution of the previously trained model
        # Check whether the predicted output is correct (e.g. if we have 4 classes to be predicted,
        # then there should be present the numbers 0, 1, 2, 3 ... no more, no less!)
        post_pred_big_tmp = pm.sample_posterior_predictive(trace=arviz_inference,
                                                           samples=samples)

    return post_pred_big_tmp


###############################################################################
## 6. PREDICTION PERFORMANCE
# function for parallelizing the computation of MCC scores
def run_model_oob(X_continuos, 
                  X_categorical_selection,
                  X_categorical_gender,
                  X_categorical_audience,
                  X_categorical_browser,
                  X_categorical_city,
                  X_categorical_device,
                  y_data, 
                  variables_to_be_used,
                  variant_df,
                  arviz_inference,
                  samples,
                  y, 
                  mcc_metric):
    
    """Re-build Bayesian model in PyMC3 + compute MCC score on test-set.
      Args:
        x_02_data: matrix with all the not-hierachical variables (one for each 
            column) to be analyzed.
        x_05_data: column with the hierarchical variable.
        y_data: target variable/output.
        variables_to_be_used: used for getting the number of columns for
            x_02_data. That number informs the shape of the probability 
            distribution.
        variables_df: number of target variable/output (e.g. control + variants).
            That number informs the shape of the probability distribution.   
        arviz_inference: it is the MCMC traces
            generated by the training step before. 
        samples: number of samples of the posterior predictive (e.g. ideally, 
            it is better to have ~ 10000 samples).
        y : target variable/output. 
        mcc_metric: mertrics used for computing model's predictive performance.
        
        
      Returns:
        mcc_metric: MCC scores (and other metrics).  
    """
    
    # predicting new values from the posterior distribution of the previously trained model
    post_pred_big = model_factory(X_continuos, 
                                  X_categorical_selection,
                                  X_categorical_gender,
                                  X_categorical_audience,
                                  X_categorical_browser,
                                  X_categorical_city,
                                  X_categorical_device,
                                  y_data, 
                                  variables_to_be_used,
                                  variant_df,
                                  arviz_inference,
                                  samples)

    # print when the iteration is completed
    print('iteration completed')

    # convert predictions into a DataFrame
    transposed_output_01 = pd.DataFrame(post_pred_big['y_likelihood'])

    # compute the mode (most frequent class) of the predicted outputs
    transposed_output_02 = transposed_output_01.mode(axis=0, dropna=False)

    # transpose the first row of the mode's output
    transposed_output_03 = transposed_output_02.iloc[0, :].T

    # rename column
    transposed_output_04 = transposed_output_03.rename('predicted')

    # combine predicted + truth into one table
    y_predicted = pd.concat([transposed_output_04, y], axis = 1)

    # compute and print several classification metrics
    MCC_metric = pd.Series(matthews_corrcoef(y_predicted.loc[:, 'truth'],
                                             y_predicted.loc[:, 'predicted'])).rename('MCC_metric')
    accuracy_metric = pd.Series(accuracy_score(y_predicted.loc[:, 'truth'],
                                               y_predicted.loc[:, 'predicted'])).rename('accuracy_metric')
    precision_metric = pd.Series(precision_score(y_predicted.loc[:, 'truth'],
                                                 y_predicted.loc[:, 'predicted'],
                                                 average='micro')).rename('precision_metric')
    recall_metric = pd.Series(recall_score(y_predicted.loc[:, 'truth'],
                                           y_predicted.loc[:, 'predicted'],
                                           average='micro')).rename('recall_metric')

    # compute MCC classification metric
    mcc_metric_concatenate = pd.concat([MCC_metric,
                                        accuracy_metric,
                                        precision_metric,
                                        recall_metric],
                                       axis=1)

    mcc_metric = mcc_metric.append(mcc_metric_concatenate,
                                   ignore_index=True,
                                   sort=False)

    return (mcc_metric)


#  initialize DataFrame()
mcc_metric = pd.DataFrame()

# running either parallel or single-core computation.
if parallel:
    # execute computations in parallel (multi-threading)
    executor = joblib.Parallel(n_jobs=int(round((cpu_count()), 0) - 1),
                               backend='loky')
    tasks = (joblib.delayed(run_model_oob)(X_continuos, 
                                          X_categorical_selection,
                                          X_categorical_gender,
                                          X_categorical_audience,
                                          X_categorical_browser,
                                          X_categorical_city,
                                          X_categorical_device,
                                          y_data, 
                                          variables_to_be_used,
                                          variant_df,
                                          arviz_inference,
                                          samples,
                                          y, 
                                          mcc_metric) for i in range(n_iterations))
    output = executor(tasks)

else:
    # execute computations by single core
    output = [run_model_oob(X_continuos, 
                            X_categorical_selection,
                            X_categorical_gender,
                            X_categorical_audience,
                            X_categorical_browser,
                            X_categorical_city,
                            X_categorical_device,
                            y_data, 
                            variables_to_be_used,
                            variant_df,
                            arviz_inference,
                            samples,
                            y, 
                            mcc_metric) for i in range(n_iterations)]
    

###############################################################################
## 7. SAVE BEST SCORES IN A PANDAS DATAFRAME AND PLOT THEIR BOOTSTRAPPING
## DISTRIBUTION
results_02 = pd.DataFrame()

# collect and save all CV results for plotting
for counter in range(0, len(output)):
    # collect cross-validation results (e.g. multiple metrics etc.)
    results_01 = pd.DataFrame()
    results_01 = pd.DataFrame(output[counter])
    results_02 = results_02.append(results_01, ignore_index=True)

# print
print('Number iterations: {}'.format(len(output)))

# absolute values on the MCC scores (dirty workaround, but it is ok for now!)
results_03 = results_02.abs()

# take median
results_04 = pd.DataFrame(results_03.median(axis = 0).round(decimals = 4)).T

# save cleaned DataFrame as .csv file
results_04.to_csv(os.path.sep.join([BASE_DIR_OUTPUT,
                                    output_file_name_12]),
                                   index = False)

# deep copy
summary_table = results_03.copy()

# confidence intervals for MCC_metric
alpha = 0.95
p = ((1.0 - alpha) / 2.0) * 100
lower = max(0.0, np.percentile(summary_table.loc[:, 'MCC_metric'], p))
p = (alpha + ((1.0 - alpha) / 2.0)) * 100
upper = min(1.0, np.percentile(summary_table.loc[:, 'MCC_metric'], p))
median = np.median(summary_table.loc[:, 'MCC_metric'])
print('{} % confidence interval {} and {}'.format(alpha * 100, np.round(lower, decimals = 3),
                                                  np.round(upper, decimals = 3)))
print('Median MCC_metric {}'.format(np.round(median, decimals = 3)))


# end time according to computer clock
end_time = time.time()

# shows run-time's timestamps + total execution time
print('start time (unix timestamp):{}'.format(start_time))
print('end time (unix timestamp):{}'.format(end_time))
print('total execution time (seconds):{}'.format(np.round((end_time - start_time), 2)))






  

