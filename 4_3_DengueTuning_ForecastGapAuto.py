#!/usr/bin/env python
# coding: utf-8

# # 4_3_DengueTuning_ForecastGapAuto
# Here, we work on tuning the dengue forecasting model using the "Forecast the gap" approach. This means that we forecast across the 15-week gap between the training/hold data and the target.

# In[1]:


import pandas as pd
import numpy as np
import sys
import torch 
import os
import json

from darts import TimeSeries, concatenate
from darts.utils.callbacks import TFMProgressBar
from darts.metrics import mape, smape, mse, rmse, mae, mql, ql
from darts.dataprocessing.transformers import Scaler
from darts.dataprocessing.transformers import StaticCovariatesTransformer
from sklearn.preprocessing import MinMaxScaler
import optuna
from optuna.samplers import TPESampler
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from darts.models import (
    RandomForestModel,
    LinearRegressionModel,
    XGBModel,
    TFTModel,
    TiDEModel
)
from darts.utils.likelihood_models import GaussianLikelihood, PoissonLikelihood, QuantileRegression
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import Callback, EarlyStopping
import pickle
import helpers as hp

torch.set_float32_matmul_precision("medium")
pd.options.mode.copy_on_write = True

include_stringency_index = False
accel = "gpu"

quantiles = [0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99]


# # 1. Reading in Data

# In[2]:


n_optuna_trials = int(sys.argv[1])
gpu_id = int(sys.argv[2])
tuning_name = sys.argv[3]
run_name = sys.argv[4]
random_seed = int(sys.argv[5])

base_dir = os.getcwd()
model_input_dir = os.path.join(base_dir, "ModelInput")
tuning_dir = os.path.join(base_dir, "Dengue Tuning", tuning_name)
climate_tuning_dir = os.path.join(base_dir, "Climate Tuning", "Notebook_24_07_2025")
climate_model_name = "RandomForest"
climate_forecast_dir = os.path.join(climate_tuning_dir, climate_model_name, "Forecasts")

os.makedirs(tuning_dir, exist_ok = True)

weekly_cal = pd.read_csv(os.path.join(model_input_dir, "Calendar.csv"))
weekly_cal = weekly_cal.iloc[1:]
climate_df = pd.read_csv(os.path.join(model_input_dir, "TimeVaryingCovs.csv"))
dengue_df = pd.read_csv(os.path.join(model_input_dir, "DengueCases.csv"))

#Remove uf = "ES" since there is no dengue data for Espirito Santo
climate_df = climate_df[climate_df["uf"] != "ES"]
dengue_df = dengue_df[dengue_df["uf"] != "ES"]

dengue_df = dengue_df[["epiweek", "Year", "Week", "uf", "uf_name", "Cases"]]


# In[3]:


#Read the climate forecasts
clim_forecast_dict = hp.load_pickle(os.path.join(climate_forecast_dir, "forecast_dict.pkl"))
clim_forecast_log_dict = hp.load_pickle(os.path.join(climate_forecast_dir, "forecast_dict_log.pkl"))


# In[4]:


clim_tuning_config = hp.read_json(os.path.join(climate_tuning_dir,"tuning_details.json"))
clim_var_names = clim_tuning_config["clim_var_names"]

#If boolean says we should include stringency index, then we add it into the climate variable names
if include_stringency_index:
    clim_var_names.append("StringencyIndex")


# In[5]:

#Read a configuration file with details regarding the hyperparameter tuning process
den_tuning_config = hp.read_json(os.path.join(tuning_dir, "tuning_config.json"))
global_fit = den_tuning_config["global_fit"]
den_tuning_config


# In[6]:


set_info = weekly_cal[["epiweek", "Year", "Week", "WeekStart", "WeekMid", "WeekEnd"]]


#Set up all the set information for splitting the dengue and climate data

#For tuning training 1a,b,c - val1a,b,c
set_info["den_train1a"] = set_info["epiweek"] <= 201825
set_info["den_train1b"] = set_info["epiweek"] <= 201925
set_info["den_train1c"] = set_info["epiweek"] <= 202025

set_info["den_gap1a"] = (set_info["epiweek"] >= 201826) & (set_info["epiweek"] <= 201840)
set_info["den_gap1b"] = (set_info["epiweek"] >= 201926) & (set_info["epiweek"] <= 201940)
set_info["den_gap1c"] = (set_info["epiweek"] >= 202026) & (set_info["epiweek"] <= 202040)

set_info["den_val1a"] = (set_info["epiweek"] >= 201841) & (set_info["epiweek"] <= 201940)
set_info["den_val1b"] = (set_info["epiweek"] >= 201941) & (set_info["epiweek"] <= 202040)
set_info["den_val1c"] = (set_info["epiweek"] >= 202041) & (set_info["epiweek"] <= 202140)

set_info["den_ex"] = (set_info["epiweek"] >= 202141) & (set_info["epiweek"] <= 202225)

#Train - Val - Test for model evaluation
set_info["den_train1"] = set_info["epiweek"] <= 202110
set_info["den_train2"] = set_info["epiweek"] <= 202210
set_info["den_train3"] = set_info["epiweek"] <= 202310

set_info["den_val1"] = (set_info["epiweek"] >= 202126) & (set_info["epiweek"] <= 202225)
set_info["den_val2"] = (set_info["epiweek"] >= 202226) & (set_info["epiweek"] <= 202325)
set_info["den_val3"] = (set_info["epiweek"] >= 202326) & (set_info["epiweek"] <= 202425)

set_info["den_targ1"] = (set_info["epiweek"] >= 202241) & (set_info["epiweek"] <= 202340)
set_info["den_targ2"] = (set_info["epiweek"] >= 202341) & (set_info["epiweek"] <= 202440)
set_info["den_targ3"] = (set_info["epiweek"] >= 202441) & (set_info["epiweek"] <= 202540)



# In[7]:


uf_list = climate_df["uf"].unique()
uf_mapper = climate_df[["uf", "uf_name"]].copy().drop_duplicates()
uf_dict = dict(zip(uf_mapper["uf"], uf_mapper["uf_name"]))
uf_name_list = [uf_dict[curr_uf] for curr_uf in uf_list]


# # 2. Data Preparation

# In[8]:


#Log transformed climate (just the precipitation columns)
log_trans_cols = ["precip_min", "precip_med", "precip_max"]
proc_climate_df = hp.transform_df_cols(climate_df, log_trans_cols, np.log1p)

#Log transformed dengue cases
proc_dengue_df = hp.transform_df_cols(dengue_df, ["Cases"], np.log1p)


# In[9]:


#Create all the TimeSeries objects we need for tuning first
group_col = "uf"

#Dengue case TimeSeries
train1a = hp.create_ts_list(dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_train1a")
train1b = hp.create_ts_list(dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_train1b")
train1c = hp.create_ts_list(dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_train1c")

val1a = hp.create_ts_list(dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_val1a")
val1b = hp.create_ts_list(dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_val1b")
val1c = hp.create_ts_list(dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_val1c")

#Log transformed version of the cases TimeSeries for actual training
train1a_log = hp.create_ts_list(proc_dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_train1a")
train1b_log = hp.create_ts_list(proc_dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_train1b")
train1c_log = hp.create_ts_list(proc_dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_train1c")

gap1a_log = hp.create_ts_list(proc_dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_gap1a")
gap1b_log = hp.create_ts_list(proc_dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_gap1b")
gap1c_log = hp.create_ts_list(proc_dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_gap1c")

val1a_log = hp.create_ts_list(proc_dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_val1a")
val1b_log = hp.create_ts_list(proc_dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_val1b")
val1c_log = hp.create_ts_list(proc_dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_val1c")

#Climate values to use as covariates 
clim_train1a_log = hp.create_ts_list(proc_climate_df, clim_var_names, group_col, uf_list, set_info = set_info, set_column = "den_train1a")
clim_train1b_log = hp.create_ts_list(proc_climate_df, clim_var_names, group_col, uf_list, set_info = set_info, set_column = "den_train1b")
clim_train1c_log = hp.create_ts_list(proc_climate_df, clim_var_names, group_col, uf_list, set_info = set_info, set_column = "den_train1c")

#These are just for checking, and won't be used in the actual forecasting since we use forecasted variables in the validation set
clim_val1a_log = hp.create_ts_list(proc_climate_df, clim_var_names, group_col, uf_list, set_info = set_info, set_column = "den_val1a")
clim_val1b_log = hp.create_ts_list(proc_climate_df, clim_var_names, group_col, uf_list, set_info = set_info, set_column = "den_val1b")
clim_val1c_log = hp.create_ts_list(proc_climate_df, clim_var_names, group_col, uf_list, set_info = set_info, set_column = "den_val1c")

#Get the forecasts over the validation set to be used as future covariates
clim_preds_1a_log = clim_forecast_log_dict["clim_preds_1a"]
clim_preds_1b_log = clim_forecast_log_dict["clim_preds_1b"]
clim_preds_1c_log = clim_forecast_log_dict["clim_preds_1c"]


#Processing with COVID-19 stringency index (optional)
def intersect_stack(curr_shorter, curr_longer): 
    to_stack = curr_longer.slice_intersect(curr_shorter)
    return curr_shorter.stack(to_stack)

if include_stringency_index:
    #If including the StringencyIndex, then should at it to the climate forecasts
    #Since there is no StringencyIndex forecasting model, we assume it is 0 in forecasts
    temp_cal = weekly_cal.copy()
    temp_cal["WeekStart"] = pd.to_datetime(temp_cal["WeekStart"], format = "%Y-%m-%d")
    temp_cal = temp_cal.set_index("WeekStart")
    zero_series = TimeSeries.from_times_and_values(
        times = temp_cal.index,
        values = np.zeros(len(temp_cal)),
        columns=["StringencyIndex"]
    )

    #Add the zero series into the TimeSeries with predicted climate
    clim_preds_1a_log = [intersect_stack(curr_series, zero_series) for curr_series in clim_preds_1a_log]
    clim_preds_1b_log = [intersect_stack(curr_series, zero_series) for curr_series in clim_preds_1b_log]
    clim_preds_1c_log = [intersect_stack(curr_series, zero_series) for curr_series in clim_preds_1c_log]


# ## 2.1 Add Static Covariates

# In[10]:


stat_covs = pd.read_csv(os.path.join(model_input_dir, "StaticCovs.csv"))
stat_covs_names = stat_covs.columns[2:]

#generate min-max scaled version of the static covariates
stat_cov_scaler = MinMaxScaler()
stat_cov_scaler.fit(stat_covs[stat_covs_names])
stat_covs_vals_s = pd.DataFrame(stat_cov_scaler.transform(stat_covs[stat_covs_names]), columns = stat_covs_names)
stat_covs_s = pd.concat([stat_covs[["uf", "uf_name"]], stat_covs_vals_s], axis = 1)


# In[11]:


train1a_log_sc = hp.add_stat_covs(train1a_log, stat_covs_s, uf_list, stat_covs_names)
train1b_log_sc = hp.add_stat_covs(train1b_log, stat_covs_s, uf_list, stat_covs_names)
train1c_log_sc = hp.add_stat_covs(train1c_log, stat_covs_s, uf_list, stat_covs_names)



# ## 2.3 Rescaling TimeSeries data

# In[12]:


#Rescale the cases and the covariates
den_tune_scaler = Scaler(global_fit = global_fit)
clim_tune_scaler = Scaler(global_fit = global_fit)

#Fit the scalers to the training sets from the first folds
den_tune_scaler.fit(train1a_log_sc)
clim_tune_scaler.fit(clim_train1a_log)

#Generate the scaled TimeSeries
train1a_log_sc_s = den_tune_scaler.transform(train1a_log_sc)
train1b_log_sc_s = den_tune_scaler.transform(train1b_log_sc)
train1c_log_sc_s = den_tune_scaler.transform(train1c_log_sc)

gap1a_log_s = den_tune_scaler.transform(gap1a_log)
gap1b_log_s = den_tune_scaler.transform(gap1b_log)
gap1c_log_s = den_tune_scaler.transform(gap1c_log)

val1a_log_s = den_tune_scaler.transform(val1a_log)
val1b_log_s = den_tune_scaler.transform(val1b_log)
val1c_log_s = den_tune_scaler.transform(val1c_log)


#Scaled climate covariates
clim_train1a_log_s = clim_tune_scaler.transform(clim_train1a_log)
clim_train1b_log_s = clim_tune_scaler.transform(clim_train1b_log)
clim_train1c_log_s = clim_tune_scaler.transform(clim_train1c_log)

clim_preds_1a_log_s = clim_tune_scaler.transform(clim_preds_1a_log)
clim_preds_1b_log_s = clim_tune_scaler.transform(clim_preds_1b_log)
clim_preds_1c_log_s = clim_tune_scaler.transform(clim_preds_1c_log)

#Merge the training and validation covariates to be passed into the predict function for validation loss evaluation
clim_train_val1a_log_s =  [concatenate([curr_train, curr_val]) for curr_train, curr_val in zip(clim_train1a_log_s, clim_preds_1a_log_s)]
clim_train_val1b_log_s =  [concatenate([curr_train, curr_val]) for curr_train, curr_val in zip(clim_train1b_log_s, clim_preds_1b_log_s)]
clim_train_val1c_log_s =  [concatenate([curr_train, curr_val]) for curr_train, curr_val in zip(clim_train1c_log_s, clim_preds_1c_log_s)]


# # 3. Modelling
# At this point, we have all the data prepared and we can start the model tuning process.

# ## 3.1 Some Helpers
# First, we create a model creation function that creates a Darts model based on input hyperparameters.

# In[13]:


def series_difference(curr_longer, curr_shorter, input_chunk_length = 0, replay = 0):
    """
    Gets the difference between two series, with one longer than the other (points in longer that are not in shorter).
    Additionally gets the last input_chunk_length + replay data points from the shorter series and adds them to the series difference.

    """
    shorter_end = curr_shorter.end_time()
    long_before, long_after = curr_longer.split_after(shorter_end)
    to_ret = long_after
    to_add = 0 
    to_add += input_chunk_length
    to_add += replay

    if to_add != 0:
        to_ret = concatenate([long_before[-to_add:], to_ret])
    return to_ret

def series_difference_list(curr_longer_list, curr_shorter_list, input_chunk_length = 0, replay = 0):
    """
    Applies the series difference function to two lists of TimeSeries, where one list always has longer series
    than the corresponding series in the other list. 
    """
    to_ret = [series_difference(curr_longer, curr_shorter, input_chunk_length = input_chunk_length, replay = replay) for curr_longer, curr_shorter in zip(curr_longer_list, curr_shorter_list)]
    return to_ret

def num_windows(curr_ts, input_chunk_length, output_chunk_length):
    """
    Returns number of input/output chunk pairs that could be derived from a TimeSeries assuming stride = 1
    """
    return len(curr_ts) - (input_chunk_length + output_chunk_length) + 1

def get_replay_length(curr_ts, input_chunk_length, output_chunk_length, p):
    """
    Assuming stride=1, answers the question "how many points should we add to the input TimeSeries
    such that the number of i/o chunk pairs is increased by p" where p is some percentage. 
    """
    s = len(curr_ts)
    win_size = (input_chunk_length + output_chunk_length) 
    return int(np.ceil((1+p) * (s - win_size + 1) + (win_size) - 1)) - s



# ## 3.2 Model Builder

# In[14]:


gap_length = 15
val_base_length = 53
forecast_length = gap_length + val_base_length
replay_fraction = den_tuning_config["replay_fraction"]

if den_tuning_config["val_metric"] == "mql":
    val_metric = mql
else:
    val_metric = mae


# In[15]:


def build_tft_model(model_name, batch_size, input_chunk_length, output_chunk_length, hidden_size, hidden_continuous_size, lstm_layers, num_attention_heads, dropout, learning_rate, output_chunk_shift = 0, force_reset = True):
    torch.manual_seed(random_seed)
    curr_encoders = {"datetime_attribute": {"future": ["month", "year"]}, 
                    "transformer": Scaler()}
    early_stopper = EarlyStopping("val_loss", min_delta = 0.0001, patience = 8)

    curr_model = TFTModel(
        model_name = model_name,
        batch_size = batch_size,
        input_chunk_length = input_chunk_length,
        output_chunk_length = output_chunk_length,
        output_chunk_shift = output_chunk_shift,
        hidden_size = hidden_size,
        hidden_continuous_size = hidden_continuous_size,
        lstm_layers = lstm_layers,
        num_attention_heads = num_attention_heads,
        dropout = dropout,
        n_epochs = 100,
        add_encoders = curr_encoders,
        likelihood = QuantileRegression(quantiles = quantiles), 
        optimizer_kwargs = {"lr": learning_rate},
        pl_trainer_kwargs = {"accelerator": accel,
                             "devices": [gpu_id],
                             "callbacks": [early_stopper]},
        show_warnings = True,
        save_checkpoints = True,
        use_static_covariates = True,
        force_reset = force_reset,
        random_state = random_seed
    )
    return curr_model


# In[16]:


# ## 3.3 Objective Function

# In[17]:


def objective(trial):
    model_name = "BrazilModel_" + den_tuning_config["ModelSuffix"] + "_" + str(run_name) + "_" + str(trial._trial_id)
    batch_size = trial.suggest_categorical("batch_size", den_tuning_config["batch_size"])
    input_chunk_length = trial.suggest_int("input_chunk_length", den_tuning_config["input_chunk_length_low"], den_tuning_config["input_chunk_length_high"])

    output_chunk_length_config = den_tuning_config["output_chunk_length_high"]
    if output_chunk_length_config[0] == "min_vs_input":
        #We can set the output_chunk_length upper end to ensure it is smaller than the proposed input_chunk_length 
        output_chunk_length_high = min(input_chunk_length, output_chunk_length_config[1])
    else: #If not labelled as min_vs_input, we take the value of output_chunk_length_high directly and the entire range is sampled (not dependent on input_chunk_length for the trial) 
        output_chunk_length_high = output_chunk_length_config[1]

    #If output_chunk_length_high is not fixed, then we have to tune it 
    if output_chunk_length_config[0] != "fixed":
        output_chunk_length = trial.suggest_int("output_chunk_length", den_tuning_config["output_chunk_length_low"], output_chunk_length_high)
    else:
        #Rather than tuning output_chunk_length, if the config file says it's fixed, the value is directly taken from output_chunk_length_high and the "low" value is ignored
        output_chunk_length = output_chunk_length_high

    hidden_size = trial.suggest_categorical("hidden_size", den_tuning_config["hidden_size"])
    hidden_continuous_size = hidden_size #We assume that hidden_continuous_size is the same as hidden_size
    lstm_layers = trial.suggest_int("lstm_layers",  den_tuning_config["lstm_layers_low"], den_tuning_config["lstm_layers_high"])
    num_attention_heads = trial.suggest_categorical("num_attention_heads",  den_tuning_config["num_attention_heads"])
    dropout = trial.suggest_float("dropout",  den_tuning_config["dropout_low"],  den_tuning_config["dropout_high"])
    learning_rate = trial.suggest_float("learning_rate",  den_tuning_config["learning_rate_low"],  den_tuning_config["learning_rate_high"], log = True)
    output_chunk_shift = 0

    start_model = build_tft_model(model_name = model_name,
                                 batch_size = batch_size,
                                 input_chunk_length = input_chunk_length,
                                 output_chunk_length = output_chunk_length, 
                                 output_chunk_shift = output_chunk_shift,
                                 hidden_size = hidden_size,
                                 hidden_continuous_size = hidden_size,
                                 lstm_layers = lstm_layers,
                                 num_attention_heads = num_attention_heads,
                                 dropout = dropout,
                                 learning_rate = learning_rate
                                )


    #Dynamic validation set generation - add the last input_chunk_length + gap_length data points from training set into the validation set
    dyn_val1a_log_sc_s = [concatenate([curr_train[-(input_chunk_length):], curr_gap, curr_val]) for curr_train, curr_gap, curr_val in zip(train1a_log_sc_s, gap1a_log_s, val1a_log_s)]
    dyn_val1b_log_sc_s = [concatenate([curr_train[-(input_chunk_length):], curr_gap, curr_val]) for curr_train, curr_gap, curr_val in zip(train1b_log_sc_s, gap1b_log_s, val1b_log_s)]
    dyn_val1c_log_sc_s = [concatenate([curr_train[-(input_chunk_length):], curr_gap, curr_val]) for curr_train, curr_gap, curr_val in zip(train1c_log_sc_s, gap1c_log_s, val1c_log_s)]

    #Compute some values such as replay length that we use to create the fine tuning series
    curr_longer = train1b_log_sc_s[0]
    curr_shorter = train1a_log_sc_s[0]
    sample_diff = series_difference(curr_longer, curr_shorter, input_chunk_length = input_chunk_length)
    replay_length = get_replay_length(sample_diff, input_chunk_length, output_chunk_length, replay_fraction)

    #Create the series for training during fine tuning
    train1b_min_1a_log_sc_s = series_difference_list(train1b_log_sc_s, train1a_log_sc_s, input_chunk_length = input_chunk_length, replay = replay_length)
    train1c_min_1b_log_sc_s = series_difference_list(train1c_log_sc_s, train1b_log_sc_s, input_chunk_length = input_chunk_length, replay = replay_length)

    #Train the initial model on train1a
    start_model.fit(series = train1a_log_sc_s, 
                   val_series = dyn_val1a_log_sc_s, 
                   future_covariates = clim_train_val1a_log_s, 
                   val_future_covariates = clim_train_val1a_log_s, 
                   dataloader_kwargs = {"num_workers": 8, "persistent_workers": True})

    #Get the best model and predict
    best_model_1a = start_model.load_from_checkpoint(model_name, best = True)
    val1a_mod1a_predict_log_s = best_model_1a.predict(series = train1a_log_sc_s, future_covariates = clim_train_val1a_log_s, n = forecast_length, num_samples = 100, dataloader_kwargs = {"num_workers": 8, "persistent_workers": True})
    #val1b_mod1a_predict_log_s = best_model_1a.predict(series = train1b_log_sc_s, future_covariates = clim_train_val1b_log_s, n = forecast_length, num_samples = 100, dataloader_kwargs = {"num_workers": 8, "persistent_workers": True})
    #val1c_mod1a_predict_log_s = best_model_1a.predict(series = train1c_log_sc_s, future_covariates = clim_train_val1c_log_s, n = forecast_length, num_samples = 100, dataloader_kwargs = {"num_workers": 8, "persistent_workers": True})

    #Convert predictions to natural scale
    val1a_mod1a_predict = hp.return_to_nat_scale(val1a_mod1a_predict_log_s, ["Cases"], den_tune_scaler, trans_func = np.expm1)
    #val1b_mod1a_predict = hp.return_to_nat_scale(val1b_mod1a_predict_log_s, ["Cases"], den_tune_scaler, trans_func = np.expm1)
    #val1c_mod1a_predict = hp.return_to_nat_scale(val1c_mod1a_predict_log_s, ["Cases"], den_tune_scaler, trans_func = np.expm1)

    #Compute validation loss for 1a - val1a persus val1a_predict
    val_loss1a_mod1a = hp.norm_metric_list(val1a, val1a_mod1a_predict, train1a, series_summ_func = np.mean, metric = val_metric, quantiles = quantiles)
    #val_loss1b_mod1a = hp.norm_metric_list(val1b, val1b_mod1a_predict, train1a, series_summ_func = np.mean, metric = mql, quantiles = quantiles)
    #val_loss1c_mod1a = hp.norm_metric_list(val1c, val1c_mod1a_predict, train1a, series_summ_func = np.mean, metric = mql, quantiles = quantiles)

    #Build a new model for fine tuning with train1b
    model_1b = build_tft_model(model_name = model_name + "_1b",
                         batch_size = batch_size,
                         input_chunk_length = input_chunk_length,
                         output_chunk_length = output_chunk_length, 
                         output_chunk_shift = output_chunk_shift,
                         hidden_size = hidden_size,
                         hidden_continuous_size = hidden_size,
                         lstm_layers = lstm_layers,
                         num_attention_heads = num_attention_heads,
                         dropout = dropout,
                         learning_rate = learning_rate,
                         force_reset = True
                        )

    #Load the best weights from the training on 1a
    model_1b.load_weights_from_checkpoint(model_name = model_name, best = True) #Load the weights from the best model

    #Fit on train1b
    model_1b.fit(series = train1b_min_1a_log_sc_s, 
                   val_series = dyn_val1b_log_sc_s, 
                   future_covariates = clim_train_val1b_log_s, 
                   val_future_covariates = clim_train_val1b_log_s, 
                   dataloader_kwargs = {"num_workers": 8, "persistent_workers": True})

    #Build another model to load the best weights - there seems to be a bug in Darts with just loading the checkpoint straight
    best_model_1b = build_tft_model(model_name = model_name + "_best_1b",
                         batch_size = batch_size,
                         input_chunk_length = input_chunk_length,
                         output_chunk_length = output_chunk_length, 
                         output_chunk_shift = output_chunk_shift,
                         hidden_size = hidden_size,
                         hidden_continuous_size = hidden_size,
                         lstm_layers = lstm_layers,
                         num_attention_heads = num_attention_heads,
                         dropout = dropout,
                         learning_rate = learning_rate,
                         force_reset = True
                        )

    #Load the best model fine tuned on train_1b
    best_model_1b.load_weights_from_checkpoint(model_name = model_name + "_1b", best = True) #Load the weights from the best model

    #Generate predictions using the best model fine tuned on train1b
    val1b_mod1b_predict_log_s = best_model_1b.predict(series = train1b_log_sc_s, future_covariates = clim_train_val1b_log_s, n = forecast_length, num_samples = 100, dataloader_kwargs = {"num_workers": 8, "persistent_workers": True})
    #val1c_mod1b_predict_log_s = best_model_1b.predict(series = train1c_log_sc_s, future_covariates = clim_train_val1c_log_s, n = forecast_length, num_samples = 100, dataloader_kwargs = {"num_workers": 8, "persistent_workers": True})

    #Convert the predictions to natural scale
    val1b_mod1b_predict = hp.return_to_nat_scale(val1b_mod1b_predict_log_s, ["Cases"], den_tune_scaler, trans_func = np.expm1)
    #val1c_mod1b_predict = hp.return_to_nat_scale(val1c_mod1b_predict_log_s, ["Cases"], den_tune_scaler, trans_func = np.expm1)

    #Compute the validation loss
    val_loss1b_mod1b = hp.norm_metric_list(val1b, val1b_mod1b_predict, train1a, series_summ_func = np.mean, metric = val_metric, quantiles = quantiles)
    #val_loss1c_mod1b = hp.norm_metric_list(val1c, val1c_mod1b_predict, train1a, series_summ_func = np.mean, metric = mql, quantiles = quantiles)

    #Create a new model to fine tune on 1c
    model_1c = build_tft_model(model_name = model_name + "_1c",
                         batch_size = batch_size,
                         input_chunk_length = input_chunk_length,
                         output_chunk_length = output_chunk_length, 
                         output_chunk_shift = output_chunk_shift,
                         hidden_size = hidden_size,
                         hidden_continuous_size = hidden_size,
                         lstm_layers = lstm_layers,
                         num_attention_heads = num_attention_heads,
                         dropout = dropout,
                         learning_rate = learning_rate,
                         force_reset = True
                        )

    #Load the weights from the best model trained on 1b
    model_1c.load_weights_from_checkpoint(model_name = model_name + "_1b", best = True) #Load the weights from the best model

    #Fit on train1c
    model_1c.fit(series = train1c_min_1b_log_sc_s, 
                   val_series = dyn_val1c_log_sc_s, 
                   future_covariates = clim_train_val1c_log_s, 
                   val_future_covariates = clim_train_val1c_log_s, 
                   dataloader_kwargs = {"num_workers": 8, "persistent_workers": True})

    #Create another model to load the best weights from the fine tuned model on 1c
    best_model_1c = build_tft_model(model_name = model_name + "_best_1c",
                         batch_size = batch_size,
                         input_chunk_length = input_chunk_length,
                         output_chunk_length = output_chunk_length, 
                         output_chunk_shift = output_chunk_shift,
                         hidden_size = hidden_size,
                         hidden_continuous_size = hidden_size,
                         lstm_layers = lstm_layers,
                         num_attention_heads = num_attention_heads,
                         dropout = dropout,
                         learning_rate = learning_rate,
                         force_reset = True
                        )

    #Load the best model fine tuned on 1c
    best_model_1c.load_weights_from_checkpoint(model_name = model_name + "_1c", best = True) #Load the weights from the best model

    #Generate predictions on val 1c
    val1c_mod1c_predict_log_s = best_model_1c.predict(series = train1c_log_sc_s, future_covariates = clim_train_val1c_log_s, n = forecast_length, num_samples = 100, dataloader_kwargs = {"num_workers": 8, "persistent_workers": True})
    #Convert predictions to natural scale
    val1c_mod1c_predict = hp.return_to_nat_scale(val1c_mod1c_predict_log_s, ["Cases"], den_tune_scaler, trans_func = np.expm1)
    #Compute validation loss
    val_loss1c_mod1c = hp.norm_metric_list(val1c, val1c_mod1c_predict, train1a, series_summ_func = np.mean, metric = val_metric, quantiles = quantiles)

    #Put together the loss values for the three validation sets
    losses = np.array([val_loss1a_mod1a, val_loss1b_mod1b, val_loss1c_mod1c])

    #We return the mean validation loss across the three sets
    loss = np.mean(losses)
    return(loss)


# ## 3.4 Tuning

# In[18]:


#Create the Optuna journal storage file for the tuning process
storage = JournalStorage(
    JournalFileBackend(os.path.join(tuning_dir, den_tuning_config["OptunaLog"]))
)

multivariate_tpe = False
constant_liar_tpe = False

if("Multivariate_TPE" in den_tuning_config):
    multivariate_tpe = den_tuning_config["Multivariate_TPE"]
if("ConstantLiar_TPE" in den_tuning_config):
    constant_liar_tpe = den_tuning_config["ConstantLiar_TPE"]


#Create a sampler 
tuning_sampler = TPESampler(seed = random_seed, multivariate = multivariate_tpe, constant_liar = constant_liar_tpe)

#We then create the Optuna study, or load it if it exists
tuning_study = optuna.create_study(
    study_name = den_tuning_config["StudyName"],
    direction = "minimize",
    sampler = tuning_sampler,
    storage = storage,
    load_if_exists = True
)


# In[19]:


#For printing 
def print_callback(study, trial):
    print(f"Current Trial: {trial._trial_id}")
    print(f"Current value: {trial.value}, Current params: {trial.params}")
    print(f"Best value: {study.best_value}, Best params: {study.best_trial.params}")

import logging
import warnings
logging.getLogger("darts.models").setLevel(logging.CRITICAL)
logging.getLogger("pytorch_lightning").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore")

tuning_study.optimize(objective, callbacks = [print_callback], n_trials = n_optuna_trials)


# In[22]:


to_output = {"Study": tuning_study, "best_params": tuning_study.best_params, "trial_log_df": tuning_study.trials_dataframe(), "tuning_config": den_tuning_config, "sampler": tuning_sampler, "seed": random_seed}
hp.save_pickle(to_output, os.path.join(tuning_dir, "TFT_Tuning_StudyName=" + den_tuning_config["StudyName"] + "_GPU=" + str(gpu_id) + "_" + run_name + ".pkl"))
