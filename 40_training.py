from subprocess import run
from subprocess import call
import os
import time



print(os.getcwd())
inverse_ML_training_script = f'41_ann_all.py'

# you want to execute the following block of code for each set of features you care about
# feel free to loop through multiple feature_filename values

feature_filename = 'augmented_features_8x_100mV'
print(f'RUNNING: {feature_filename}')
start_time = time.time()
run(['python',inverse_ML_training_script, feature_filename])
print('\n')
print(f'training NN on {feature_filename} took {time.time()-start_time} seconds')
print('\n')
