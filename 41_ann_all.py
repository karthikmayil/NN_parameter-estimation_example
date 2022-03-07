import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Activation
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV
from keras.wrappers.scikit_learn import KerasRegressor
import math
from keras_tuner.tuners import RandomSearch
from sklearn.preprocessing import StandardScaler
from tensorflow.random import set_seed
import time
import joblib
set_seed(67)
import shutil
import sys
import matplotlib.pyplot as plt
import os

# load features. change filepath to use different features
feature_filename = sys.argv[1]
X_df = pd.read_csv(f'30_sim_features/{feature_filename}.csv',sep=',',index_col=0)


print('loaded training features')

def get_iteration(filepath):
    iterations = []
    for file in os.listdir(filepath):
        if file.startswith('Simulation_Parameters'):
            iterations.append(int(file.split('_I')[1].split('.')[0]))
    return np.max(iterations)

it = get_iteration(os.getcwd()+'/10_training_set_params/')

# load targets
param_df = pd.read_csv(f'10_training_set_params/targets_I{it}.csv',sep=',',index_col=0)
param_df = param_df.loc[param_df.index.isin(X_df.index)]


# multiply outputs by N if using data augmentaiton
if 'augmented' in feature_filename:
    N = int(feature_filename.split('x')[0][-1])
    param_df = param_df.loc[X_df.index.values,:]

print('loaded training simulation parameters')

# split

X_train, X_test, y_train, y_test = train_test_split(X_df.values, param_df, test_size=0.20, random_state = 42)
print('split into test and train set')


# define search space for ANN optimization
def build_model(hp):
    # callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
    # callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2, patience=50, min_lr=0.0001, min_delta=0.001)
    model = Sequential()
    model.add(Dense(units = len(X_df.values.T), activation = "relu"))
    for i in range(hp.Int('Layers', 2,32, step = 4)):
        model.add(Dense(
            units = hp.Int('Unit'+str(i), 2, 128, step = 8),
            activation = 'relu', kernel_initializer='random_uniform', bias_initializer='zeros'))
    model.add(Dense(units = len(y_train.columns), activation = 'linear', kernel_initializer='random_uniform',
                bias_initializer='zeros'))
    model.compile('adam',loss = 'mean_squared_error', metrics = ['mean_squared_error'])
    return model

# deletes hyperparameter tuning files
hyperparam_tuning_dirname = '43_ann_hyperparameter_tuning_dir'
def del_tuner_files(p):
    dir_path = f'{hyperparam_tuning_dirname}/{p}'

    try:
        shutil.rmtree(dir_path)
    except OSError as e:
        print("Error: %s : %s" % (dir_path, e.strerror))


optimizer = tf.keras.optimizers.Adam(0.001)
optimizer.learning_rate.assign(0.01)

p = 'all'
scaler = MinMaxScaler().fit(y_test.values.reshape(len(y_test),len(y_test.columns)))
scaler_filename = f"42_models/{p}/scaler_{feature_filename}.save"
joblib.dump(scaler, scaler_filename)



y_train_norm = scaler.transform(y_train)
y_test_norm = scaler.transform(y_test)

print('    Hyperparaemter tuning')
del_tuner_files(p)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2, patience=50, min_lr=0.0001, min_delta=0.0001)
tuner = RandomSearch(build_model, objective='val_loss', max_trials = 3, executions_per_trial = 2, directory = f'{hyperparam_tuning_dirname}/{p}', project_name = 'Inverse_Model')
tuner.search(X_train, y_train_norm, batch_size = 128, epochs = 1200, validation_data = (X_test, y_test_norm),verbose=0, callbacks=[reduce_lr])

# build, fit, and save optimized ANN
print('    ANN training with optimized Hyperparameters')
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
ann_model = tuner.hypermodel.build(best_hps)
ann_model.compile(optimizer = 'Adam', loss = 'mean_squared_error',metrics=['mae'])
history = ann_model.fit(X_train, y_train_norm, batch_size = 128, epochs = 1200,verbose=0, callbacks=[reduce_lr],validation_split=0.2)
ann_model.save(f"42_models/{p}/ANN_{feature_filename}.h5")

plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.savefig(f"42_models/{p}/ANN_{feature_filename}_training.png")

# collect metrics to quantify performance of ANN on train and test sets
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import max_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score

model_path = f'42_models/all/ANN_{feature_filename}.h5'
model = keras.models.load_model(model_path)

scaler_path = f"42_models/all/scaler_{feature_filename}.save"
scaler = joblib.load(scaler_path)

p_training_results = {'metric_name':[],'metric_value':[],'parameter':[],'set':[]}


for XX,yy,dataset in zip([X_train,X_test],[y_train.values,y_test.values],['Train','Test']):
    y_pred = model.predict(XX)
    y_pred = scaler.inverse_transform(y_pred)

    for i,p in enumerate(param_df.columns):

        p_training_results['metric_name'].append('MAE')
        p_training_results['metric_value'].append(mean_absolute_error(yy.T[i],y_pred.T[i]))

        p_training_results['metric_name'].append('MaxE')
        p_training_results['metric_value'].append(max_error(yy.T[i],y_pred.T[i]))

        p_training_results['metric_name'].append('RMSE')
        p_training_results['metric_value'].append(np.sqrt(mean_squared_error(yy.T[i],y_pred.T[i])))

        p_training_results['metric_name'].append('MAPE')
        p_training_results['metric_value'].append(mean_absolute_percentage_error(yy.T[i],y_pred.T[i]))

        p_training_results['metric_name'].append('R2')
        p_training_results['metric_value'].append(r2_score(yy.T[i],y_pred.T[i]))

        for j in range(5):
            p_training_results['parameter'].append(p)
            p_training_results['set'].append(dataset)


p_training_results = pd.DataFrame(p_training_results)
p_training_results = p_training_results[['parameter','set','metric_name','metric_value']]

p_training_results.to_csv(f'42_models/ANN_all_training_metrics_{feature_filename}.csv',sep=',')
print('saved evaluation metrics of trained model')
