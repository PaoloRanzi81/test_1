"""
TITLE: "ONNX conversion of gradient boosting classifier for predicting four 
different video trailers: loading .onnx file + computing MCC score"
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
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import matthews_corrcoef, precision_score, recall_score, accuracy_score
import time
import onnxruntime as rt
import matplotlib.pyplot as plt
import seaborn as sns
from pikepdf import Pdf

   
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
# LOCAL_COMPUTER ='4.9.0-14-amd64'
LOCAL_COMPUTER = '5.4.0-77-generic'


# TEST:
if RELEASE == LOCAL_COMPUTER: # Linux laptop
    # BASE_DIR_INPUT = ('/home/paolo/Dropbox/Univ/scripts/python_scripts/wright_keith/production_safety_test/758_grad_boost')
    BASE_DIR_INPUT = ('/home/paolo/Dropbox/Univ/scripts/python_scripts/wright_keith/758_grad_boost_20210502')
    BASE_DIR_OUTPUT = BASE_DIR_INPUT
    INPUT_FOLDER = BASE_DIR_INPUT


# set input/output file names
input_file_name_03 = ('input/dictionary_variants_id.csv')
input_file_name_05 = ('input/config/run_modality.csv')
input_file_name_06 = ('252_SPLITTING/output/input_cross_validation/test_set.csv')
output_file_name_08 = ('output/758_analysis.onnx') 
output_file_name_12 = ('output/758_analysis_mcc_score.csv')
output_file_name_16 = ('output/758_feat_imp_permutation.joblib') 
output_file_name_30 = ('output/752_predicted_probability_distribution.pdf') 
output_file_name_31 = ('output/752_mean_of_predicted_probability_distribution.pdf') 
output_file_name_32 = ('output/752_mean_of_ground_truth.pdf') 


###############################################################################
## 3. LOADING DATA-SET 
# setting testing mode 
RELEASE = platform.release()

# start clocking time
start_time = time.time()

# loading the .csv files with raw data 
try: 
    user_activities = pd.read_csv(os.path.sep.join([INPUT_FOLDER, 
                                              input_file_name_06]), header = 0)
except OSError:
    # print("OS error: {0}".format(err))
    print("\n"
          "\n"
          "WARNING: The test-set was not found!!!!! \n"
          "The test-set MUST be present in the followig folder: 252_SPLITTING/output/input_cross_validation/. \n"
          "Please run module 252 before running the present module. \n"
          "\n"
          "\n")
    
variant_df = pd.read_csv(os.path.sep.join([INPUT_FOLDER, 
                                         input_file_name_03]), header = 0)
 
# run_modality, either: 
# - 'testing' : without grid-search => quick; 
# - 'production' : with grid-search => slow;
run_modality = pd.read_csv(os.path.sep.join([INPUT_FOLDER, 
                                         input_file_name_05]), header = None, 
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
X_test = user_activities_concat_05.iloc[:, selected_columns]
y_test_1 = user_activities_concat_05.loc[:, 'best_video']

# rename
y_test = y_test_1.rename('truth')

# convert Pandas DataFrame to Numpy array
test_data = X_test.to_numpy()
   

###############################################################################
## 5. SET ONNX MODEL
# load ONNX model 
sess = rt.InferenceSession(os.path.sep.join([BASE_DIR_OUTPUT, 
                                         output_file_name_08]))

# set inputs/outputs 
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name # (as for 20210701, no more useful?)

# generate predictions on hold-out-set 
onnx_predictions_tmp = sess.run(
    None, {input_name: test_data.astype(np.float32)})

# predictions (classes) 
onnx_predictions_classes = onnx_predictions_tmp[0]

# predictions (probabilities) 
onnx_predictions_probabilities = pd.DataFrame.from_dict(onnx_predictions_tmp[1])

# build dictionary with more meaningful columns' names
replace_column_name = dict(zip(onnx_predictions_probabilities.columns,
                               variant_df.loc[: , 'variant_description'].to_list()))

# TEST:
# onnx_predictions_probabilities['truth'] = y_test
# onnx_predictions_probabilities['predicted']  = onnx_predictions_classes

# replace old columns name with new ones
onnx_predictions_probabilities.rename(columns = replace_column_name, inplace = True)

# re-set index
onnx_predictions_probabilities.reset_index(drop = False, inplace = True)

# transform data-set from wide to long 
probabilities_long = pd.melt(onnx_predictions_probabilities, 
                             id_vars=['index'], 
                             value_vars=variant_df.loc[: , 'variant_description'].to_list())

# drop useles column
probabilities_long.drop(columns = ['index'], inplace = True)

# small_data= probabilities_long.loc[probabilities_long.loc[:, "variable"].str.contains("test1"), "value"]
# penguins = sns.load_dataset("penguins")
# sns.displot(data=penguins, x="flipper_length_mm", kind = "kde")

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

# plot KDE of predicted probability distributions
pdf_01 = plt.figure(figsize=(19, 6))
pdf_01 = sns.set(font_scale = 0.5)
pdf_01 = sns.displot(data=probabilities_long,
                  x="value",
                  hue="variable",                 
                  kind = "kde", 
                  palette = colors_variant_id
                  )
pdf_01 = plt.title('758: predicted probability distributions', fontsize=5)
pdf_01 = plt.xlabel('best videos', fontsize=10)
pdf_01 = plt.ylabel('probability', fontsize=10)
plt.tight_layout()

# save plot as .pdf file
plt.savefig(os.path.sep.join([BASE_DIR_OUTPUT, output_file_name_30]))

# close pic in order to avoid overwriting with previous pics
plt.clf()

# compute mean across "variable" column
probabilities_long_mean = probabilities_long.groupby(by=["variable"]).mean()

# re-set index
probabilities_long_mean.reset_index(drop = False, inplace = True)

# plot bar-plot of the mean of predicted probability distributions
pdf_02 = plt.figure(figsize=(19, 6))
pdf_02 = sns.set(font_scale = 0.5)
pdf_02 = sns.catplot(x="variable", 
                 y="value", 
                 data= probabilities_long_mean,
                 kind="bar", 
                 palette = colors_variant_id
                  )
pdf_02 = plt.title('758: mean of predicted probability distributions', fontsize=5)
pdf_02 = plt.xlabel('best videos', fontsize=10)
pdf_02 = plt.ylabel('probability', fontsize=10)
plt.tight_layout()

# save plot as .pdf file
plt.savefig(os.path.sep.join([BASE_DIR_OUTPUT, output_file_name_31]))

# close pic in order to avoid overwriting with previous pics
plt.clf()

# set ground truth
ground_truth = user_activities_concat_05.loc[:, ['session_id', 'best_video']]

# drop duplicates
ground_truth.drop_duplicates(inplace = True)

# duplicate column
ground_truth['target_variable'] = ground_truth.loc[:, 'best_video']

# aggregate data by Pandas groupby
aggregate_most_liked = ground_truth.groupby(by = ['best_video']).count()

# reset index
aggregate_most_liked.reset_index(drop = False, inplace = True)

# build a dictionary in order to replace video_id with more meaningful names
replace_third_level_var = dict(zip(np.arange(variant_df.loc[:, 'variant_description'].shape[0]), variant_df.loc[:, 'variant_description']))

# convert dummy strings to more meaningful strings
aggregate_most_liked.loc[:,'best_video'].replace(to_replace = replace_third_level_var, 
            inplace = True)

# plot bar-plot of the ground truth from hold-out-set
pdf_03 = plt.figure(figsize=(19, 6))
pdf_03 = sns.set(font_scale = 0.5)
pdf_03 = sns.catplot(x = 'best_video', 
                             y = 'session_id', 
                             data = aggregate_most_liked,  
                             kind = 'bar', 
                             orient = 'v', 
                             palette = colors_variant_id)
pdf_03 = plt.title('758: ground truth from hold-out-set', fontsize=5)
pdf_03 = plt.xlabel('best videos', fontsize=10)
pdf_03 = plt.ylabel('percentage of total submissions', fontsize=10)
plt.tight_layout()

# save plot as .pdf file
plt.savefig(os.path.sep.join([BASE_DIR_OUTPUT, output_file_name_32]))

# close pic in order to avoid overwriting with previous pics
plt.clf()


###############################################################################
## 6. PREDICTION PERFORMANCE
# predict class each instance belogs to
y_predicted_tmp = onnx_predictions_classes

# convert predictions into a DataFrame 
transposed_output_04 = pd.Series(y_predicted_tmp).rename('predicted')
    
# combine predicted + truth into one table
y_predicted = pd.concat([transposed_output_04, y_test], axis = 1)

# compute and print classification metrics (Matthew Correlation Coefficient (MCC), precision and recall)                        
MCC_metric = pd.Series(matthews_corrcoef(y_predicted.loc[:, 'truth'], 
                                         y_predicted.loc[:, 'predicted'])).round(decimals = 4).rename('MCC_metric')
accuracy_metric = pd.Series(accuracy_score(y_predicted.loc[:, 'truth'], 
                                             y_predicted.loc[:, 'predicted'])).round(decimals = 4).rename('accuracy_metric')
precision_metric = pd.Series(precision_score(y_predicted.loc[:, 'truth'], 
                                             y_predicted.loc[:, 'predicted'], 
                                   average = 'micro')).round(decimals = 4).rename('precision_metric')
recall_metric = pd.Series(recall_score(y_predicted.loc[:, 'truth'], 
                                       y_predicted.loc[:, 'predicted'], 
                                   average = 'micro')).round(decimals = 4).rename('recall_metric')

# print metrics    
print_output = ('\n\ MCC_score = {},\n\
                accuracy = {},\n\
                precision = {},\n\
                recall = {},\n\  \n \n \n'.format(
                MCC_metric[0].round(decimals = 3), 
                accuracy_metric[0].round(decimals = 3), 
                precision_metric[0].round(decimals = 3), 
                precision_metric[0].round(decimals = 3)))  
print(print_output)

# compute MCC classification metric
mcc_metric_concatenate = pd.concat([MCC_metric, 
                                    accuracy_metric, 
                                    precision_metric, 
                                    recall_metric], axis = 1)

# save MCC scores as .csv files
mcc_metric_concatenate.to_csv(os.path.sep.join([BASE_DIR_OUTPUT, 
                                     output_file_name_12]), index= False)


###############################################################################
## 7. MERGING ALL PDFs IN "OUTPUT" FOLDER INTO A SINGLE PDF 
# generate list of .pdf files contained in the current folder   
full_path = [] 
for root, dirs, files in os.walk(os.path.sep.join([BASE_DIR_INPUT,'output'])):
# for root, dirs, files in os.walk(BASE_DIR_INPUT):
    for file in files:
        if file.endswith('.pdf'):
            full_path.append(os.path.join(root, file))    
            # # TEST:  
            # print(file)

# initialize pikepdf's object
pdf = Pdf.new()

# loop thorugh all pdf list
for index_01, single_pdf in enumerate(full_path):
    # print(index_01, single_pdf)
    src = Pdf.open(single_pdf)
    # pdf.resize(32,32) # TEST: not working
    # pdf.page_size(32,32) # TEST: not working
    # pdf.pages.page_size(32,32) # TEST: not working
    pdf.pages.extend(src.pages)
                       
# save merged file
pdf.save(os.path.sep.join([str(BASE_DIR_OUTPUT + '/' + output_file_name_16[0:6]), ('{}_merging_all_pdfs.pdf'.format(output_file_name_16[7:10]))]))    
          

# end time according to computer clock
end_time = time.time()

# shows run-time's timestamps + total execution time
print('start time (unix timestamp):{}'.format(start_time))
print('end time (unix timestamp):{}'.format(end_time))
print('total execution time (seconds):{}'.format(np.round((end_time - start_time), 2)))



  

