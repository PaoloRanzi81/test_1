"""
TITLE: "Master script for running a hierachical bayesian model from start to finish"
AUTHOR: Paolo Ranzi 
PYTHON VERSION: 3.6.9

DESCRIPTION: 
Further, please change the following sections according to your individidual input preferences:
    - 'SETTING PATHS AND KEYWORDS'; 
    - 'PARAMETERS TO BE SET!!!'
    - TRICK: at each run of the script below re-start the Python kernel (e.g. re-start Spyder or PyCharm)

"""

###############################################################################
## 1. IMPORTING LIBRARIES
# import required Python libraries
import platform
import os
import numpy as np
import sys
import time


###############################################################################
## 2. SETTING PATHS AND KEYWORDS
# set the correct pathways/folders
BASE_DIR_INPUT = ('.')
BASE_DIR_OUTPUT = BASE_DIR_INPUT

# # TEST:
# RELEASE = platform.release()

# if RELEASE == '5.3.0-62-generic': # Linux laptop
#    BASE_DIR_INPUT = ('/home/paolo/Dropbox/Univ/scripts/python_scripts/tensorflow/wright_keith/714_grad_boost_20200618')
#    BASE_DIR_OUTPUT = BASE_DIR_INPUT

# else:
#    BASE_DIR_INPUT = ('/home/paolo/raw_data')
#    BASE_DIR_OUTPUT = ('/home/paolo/outputs')


###############################################################################
## 3. RUNNING SCRIPTS SEQUENTIALLY

# set local path where Python scripts are located. 
sys.path.insert(0, BASE_DIR_INPUT) 

# TEST: check current directory
# print(sys.path)

# start clocking time
start_time = time.time()

# running the following scripts sequentially
from slave_scripts import bayesian_hbm_403_01_training
from slave_scripts import bayesian_hbm_403_02_plot_feat_imp


# end time according to computer clock
end_time = time.time()

# shows run-time's timestamps + total execution time
print('start time (unix timestamp):{}'.format(start_time))
print('end time (unix timestamp):{}'.format(end_time))
print('total execution time (seconds):{}'.format(np.round((end_time - start_time), 2)))
