
import pandas as pd
import numpy as np
import torch 
import os
import json
import properscoring as ps
from darts import TimeSeries, concatenate
from darts.utils.callbacks import TFMProgressBar
from darts.metrics import mape, smape, mse, rmse, mae, mql, ql
from darts.dataprocessing.transformers import Scaler
from darts.dataprocessing.transformers import StaticCovariatesTransformer
from sklearn.preprocessing import MinMaxScaler

from darts.utils.likelihood_models import GaussianLikelihood, PoissonLikelihood, QuantileRegression
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import Callback, EarlyStopping
import pickle

def create_ts(data_df, val_cols, set_info = None, set_column = None, time_col = "WeekStart", time_format = "%Y-%m-%d"):
    """
    Function creates a TimeSeries object from a DataFrame using only rows where set_column is true

    Parameters:
    data_df (DataFrame): DataFrame containing TimeSeries data (assumed in chronological order)
    val_cols (list or string): Name(s) of the column(s) in data_df containing the values to make a TimeSeries (will be multivariate if a list with more than 1 element)
    set_info (DataFrame): DataFrame to merge into data_df (assumed with Year, Week, and epiweek columns) containg set information
    set_column (string): Name of the column containing booleans to be used for inclusion/exclusion in the output TimeSeries
    time_col (string): Name of the column to be used as the time-axis of the output series
    time_format (string): Format of the time column

    Returns:
    curr_ts: TimeSeries filtered based on set_column with values based on val_cols
    """
    data_df[time_col] = pd.to_datetime(data_df[time_col], format = time_format)
    data_df = data_df.set_index(time_col)

    if set_info is not None:
        #We assume the set_info DataFrame has Year, Week, and epiweek columns we can join on 
        data_df = pd.merge(data_df, set_info, how = "left", on = ["Year", "Week", "epiweek"]) 
    if set_column is not None:
        data_df = data_df[data_df[set_column]]
    
    curr_ts = TimeSeries.from_times_and_values(times = data_df.index, values = data_df[val_cols], columns = val_cols)
    return curr_ts


def create_ts_list(data_df, val_cols, group_col, group_vals, set_info = None, set_column = None, time_col = "WeekStart", time_format = "%Y-%m-%d"):
    """
    Function creates a list TimeSeries objects from a DataFrame. Each list element is a TimeSeries for a particular unit from group_vals, based on the group_col

    Parameters:
    data_df (DataFrame): DataFrame containing TimeSeries data (assumed in chronological order)
    val_cols (list or string): Name(s) of the column(s) in data_df containing the values to make a TimeSeries (will be multivariate if a list with more than 1 element)
    group_col (string): Name of the column to match with the units from group_vals
    group_vals (list): List of values in the group_col to make a TimeSeries for. The list returned by this function will have a TimeSeries for each element in group_vals.
    set_info (DataFrame): DataFrame to merge into data_df (assumed with Year, Week, and epiweek columns) containg set information
    set_column (string): Name of the column containing booleans to be used for inclusion/exclusion in the output TimeSeries
    time_col (string): Name of the column to be used as the time-axis of the output series
    time_format (string): Format of the time column

    Returns:
    to_ret: list of TimeSeries, one for each element in group_vals
    """
    to_ret = []
    if set_info is not None:
        data_df = pd.merge(data_df, set_info, how = "left", on = ["Year", "Week", "epiweek"])
        
    for curr_unit in group_vals:
        curr_unit_df = data_df[data_df[group_col] == curr_unit]
        curr_unit_ts = create_ts(curr_unit_df, val_cols, set_column = set_column, time_col = time_col, time_format = time_format) #No need to pass set_info here since we have already merged in this function
        to_ret.append(curr_unit_ts)
    return to_ret

def transform_df_cols(input_df, cols_to_trans, trans_func):
    """
    Returns a version of input_df where selected columns cols_to_trans have been transformed using trans_func
    """
    proc_df = input_df.copy()

    for curr_col in cols_to_trans:
        proc_df[curr_col] = trans_func(input_df[curr_col])

    return proc_df


def plot_mv_series(obs_ts, preds_ts, curr_fig_dims, curr_val_cols, width, height, train_ts = None, train_ts_length = None):
    fig, axs = plt.subplots(curr_fig_dims[0], curr_fig_dims[1], figsize = (width,  height))
    for curr_col, curr_ax in zip(curr_val_cols, axs.flatten()):
        if train_ts is not None:
            if train_ts_length is not None:
                train_ts[curr_col][-train_ts_length:].plot(label = "Training", ax = curr_ax)
            else:
                train_ts[curr_col].plot(label = "Training", ax = curr_ax)
        obs_ts[curr_col].plot(label = "Observed", ax = curr_ax)
        preds_ts[curr_col].plot(label = "Forecasted", ax = curr_ax)
        curr_ax.set_title(curr_col)
        curr_ax.set_xlabel("")


def log_bound(x, lower, upper, plus_1 = True, eps = 1e-6):
    """
    Transforms some value x such that when we forecast it, it will always be between lower and upper
    """
    if plus_1: 
        x += 1
        lower += 1
        upper += 1
        
    y = np.log1p((x - lower)/(upper - x + eps))
    return y

def inv_log_bounds(y, lower, upper, plus_1 = True, eps = 1e-6):
    """
    Inverse of the log_bounds transformation above
    """
    exp_y = np.exp(y)
    numerator = (exp_y - 1) * (upper + eps) + lower
    denominator = exp_y
    return numerator / denominator

def generate_preds_obs_plot(train_list, obs_list, preds_list, uf_list, uf_sub_list, clim_var_names, label_list = None, width = 20, height = 42, show_train = True, train_limit = None):
    """
    Generates a plot to compare observed and predicted values, while optionally also showing the training series - used for multivariate series like climate.
    """
    to_vis_train = [curr_ts for curr_ts, curr_uf in zip(train_list, uf_list) if curr_uf in uf_sub_list]
    to_vis_preds_list = [curr_ts for curr_ts, curr_uf in zip(preds_list, uf_list) if curr_uf in uf_sub_list]
    to_vis_obs_list = [curr_ts for curr_ts, curr_uf in zip(obs_list, uf_list) if curr_uf in uf_sub_list]
    fig, axs = plt.subplots(len(clim_var_names), len(uf_sub_list), figsize = (width,  height))
    if label_list is None:
        label_list = uf_sub_list
    else:
        label_list = [curr_label for curr_label, curr_uf in zip(label_list, uf_list) if curr_uf in uf_sub_list]
    for ind1, (curr_train, curr_obs, curr_preds, curr_label) in enumerate(zip(to_vis_train, to_vis_obs_list, to_vis_preds_list, label_list)):
        for ind2, curr_var in enumerate(clim_var_names):
            if show_train:
                if train_limit is None:
                    curr_train[curr_var].plot(ax = axs[ind2][ind1], label = "Training")
                else:
                    curr_train[curr_var][-train_limit:].plot(ax = axs[ind2][ind1], label = "Training")
            curr_obs[curr_var].plot(ax = axs[ind2][ind1], label = "Observed")
            curr_preds[curr_var].plot(ax = axs[ind2][ind1], label = "Forecasted")
            axs[ind2][ind1].set_xlabel("")
            axs[ind2][ind1].set_ylabel(curr_var)
            axs[ind2][ind1].set_title(curr_label)

def generate_preds_obs_den_plot(train_list, obs_list, preds_list, uf_list, uf_sub_list, curr_fig_dims = (7,4), label_list = None, width = 20, height = 36, show_train = True, train_limit = None, low_quantile = 0.025, high_quantile = 0.975 ):
    """
    Generates a plot to compare observed and predicted values, while optionally also showing the training series - here used for single variable series like the dengue cases.
    """
    to_vis_train = [curr_ts for curr_ts, curr_uf in zip(train_list, uf_list) if curr_uf in uf_sub_list]
    to_vis_preds_list = [curr_ts for curr_ts, curr_uf in zip(preds_list, uf_list) if curr_uf in uf_sub_list]
    to_vis_obs_list = [curr_ts for curr_ts, curr_uf in zip(obs_list, uf_list) if curr_uf in uf_sub_list]
    fig, axs = plt.subplots(curr_fig_dims[0], curr_fig_dims[1], figsize = (width,  height))
    if label_list is None:
        label_list = uf_sub_list
    else:
        label_list = [curr_label for curr_label, curr_uf in zip(label_list, uf_list) if curr_uf in uf_sub_list]
        
    for ind1, (curr_train, curr_obs, curr_preds, curr_label, curr_ax) in enumerate(zip(to_vis_train, to_vis_obs_list, to_vis_preds_list, label_list, axs.flatten())):
        if show_train:
            if train_limit is None:
                curr_train.plot(ax = curr_ax, label = "Training")
            else:
                curr_train[-train_limit:].plot(ax = curr_ax, label = "Training")
        curr_obs.plot(ax = curr_ax, label = "Observed")
        curr_preds.plot(ax = curr_ax, label = "Forecasted", low_quantile = low_quantile, high_quantile = high_quantile)
        curr_ax.set_xlabel("")
        curr_ax.set_title(curr_label)

# def norm_metric(curr_obs, curr_preds, curr_normer, summ_func = None, metric = mae):
#     """
#     Computes a normalised version of some Darts metric given an observed, predicted, and normalising TimeSeries.
#     Note that this function assumes that all inputs are on their natural scales (should invert any log transforms or min-max scaling)

#     Parameters:
#     curr_obs (TimeSeries): Darts TimeSeries of observed values to compute metrics against
#     curr_preds (TimeSeries): Darts TimeSeries forecasted values to commpute metrics with
#     curr_normer (TimeSeries): Darts TimeSeries that serves as the basis of min-max values to normalise metrics (usually the training set)
#     summ_func (function): Function to summarise the normalised per component metrics
#     metric (function): Darts metric function (see: https://unit8co.github.io/darts/generated_api/darts.metrics.metrics.html)

#     Returns:
#     Value of the computed metric summarised (if summ_func is not None) or an array containing the metric per component (if summ_func is None)
#     """
#     curr_preds = curr_preds.slice_intersect(curr_obs) #Get the part of the forecast that intersects with the observation
#     assert len(curr_preds) == len(curr_obs) #Check that the forecasts and observations match in length
    
#     bounds_list = [] 
#     for comp in curr_normer.components: #Use the normer series to find the min and max per component
#         values = curr_normer[comp].values()
#         comp_min = values.min()
#         comp_max = values.max()
#         bounds_list.append((comp_min, comp_max))
    
#     met_per_comp = metric(curr_obs, curr_preds, component_reduction = None) #Get the metric value per component
#     norm_met_vals = []
#     for met, min_max in zip(met_per_comp, bounds_list):
#         curr_min = min_max[0]
#         curr_max = min_max[1]
#         norm_met_vals.append(met / (curr_max - curr_min)) #Normalise the component error based on the min and max for each component
#     norm_met_vals = np.array(norm_met_vals)

#     #Whether to summarise the metric values or to just return the whole array
#     if summ_func is None:
#         return norm_met_vals
#     else:
#         return summ_func(norm_met_vals)

# def norm_metric_list(obs_list, preds_list, normer_list, comp_summ_func = None, series_summ_func = None, metric = mae):
#     """
#     Computes a normalised metric using a list of observed, predicted, and normalising series. Function assumes that all series are on
#     their natural scales. 

#     Parameters:
#     obs_list (list): List of Darts TimeSeries containing observed values to compute metrics against
#     preds_list (list): List of Darts TimeSeries containing forecasted values to commpute metrics with
#     normer_list (list): List of Darts TimeSeries that serve as the basis of min-max values to normalise metrics (usually the training set)
#     comp_summ_func (function): Function to summarise the normalised per component metrics
#     series_summ_func (function): Function to summarise metrics across different series
#     metric (function): Darts metric function (see: https://unit8co.github.io/darts/generated_api/darts.metrics.metrics.html)
#     """
#     result = []
#     for curr_obs, curr_preds, curr_normer in zip(obs_list, preds_list, normer_list):
#         met_val = norm_metric(curr_obs, curr_preds, curr_normer, summ_func = comp_summ_func, metric = metric)
#         result.append(met_val)
#     result = np.array(result)

#     if series_summ_func is None:
#         return result
#     else:
#         return series_summ_func(result)
    
def norm_metric(curr_obs, curr_preds, curr_normer, summ_func = None, metric = mae, quantiles = None):
    """
    Computes a normalised version of some Darts metric given an observed, predicted, and normalising TimeSeries.
    Note that this function assumes that all inputs are on their natural scales (should invert any log transforms or min-max scaling)

    Parameters:
    curr_obs (TimeSeries): Darts TimeSeries of observed values to compute metrics against
    curr_preds (TimeSeries): Darts TimeSeries forecasted values to commpute metrics with
    curr_normer (TimeSeries): Darts TimeSeries that serves as the basis of min-max values to normalise metrics (usually the training set)
    summ_func (function): Function to summarise the normalised per component metrics
    metric (function): Darts metric function (see: https://unit8co.github.io/darts/generated_api/darts.metrics.metrics.html)

    Returns:
    Value of the computed metric summarised (if summ_func is not None) or an array containing the metric per component (if summ_func is None)
    """
    curr_preds = curr_preds.slice_intersect(curr_obs) #Get the part of the forecast that intersects with the observation
    assert len(curr_preds) == len(curr_obs) #Check that the forecasts and observations match in length
    
    bounds_list = [] 
    for comp in curr_normer.components: #Use the normer series to find the min and max per component
        values = curr_normer[comp].values()
        comp_min = values.min()
        comp_max = values.max()
        bounds_list.append((comp_min, comp_max))

    #If we're not computing MQL, we do not need the quantiles
    if metric is not mql:
        quantiles = None

    if metric is mql and (quantiles is not None):
        met_per_comp = metric(curr_obs, curr_preds, component_reduction = None, q = quantiles)
        met_per_comp = [np.mean(curr_errs) for curr_errs in met_per_comp]
    elif metric is ps.crps_ensemble:
        #when computing crps, we assume that there is only one component in the series since we use this when
        #evaluating dengue forecasts. Changes should be made here to make it applicable to multivariate forecasts
        curr_obs_vals = curr_obs.all_values().squeeze()
        curr_preds_vals = curr_preds.all_values().squeeze(1)
        met_per_comp = [np.mean(ps.crps_ensemble(curr_obs_vals, curr_preds_vals))]
    else:
        met_per_comp = metric(curr_obs, curr_preds, component_reduction = None) #Get the metric value per component
    if len(curr_normer.components) == 1:
        met_per_comp = [met_per_comp]
    norm_met_vals = []
    for met, min_max in zip(met_per_comp, bounds_list):
        curr_min = min_max[0]
        curr_max = min_max[1]
        norm_met_vals.append(met / (curr_max - curr_min)) #Normalise the component error based on the min and max for each component
    norm_met_vals = np.array(norm_met_vals)

    #Whether to summarise the metric values or to just return the whole array
    if summ_func is None:
        return norm_met_vals
    else:
        return summ_func(norm_met_vals)

def norm_metric_list(obs_list, preds_list, normer_list, comp_summ_func = None, series_summ_func = None, metric = mae, quantiles = None):
    """
    Computes a normalised metric using a list of observed, predicted, and normalising series. Function assumes that all series are on
    their natural scales. 

    Parameters:
    obs_list (list): List of Darts TimeSeries containing observed values to compute metrics against
    preds_list (list): List of Darts TimeSeries containing forecasted values to commpute metrics with
    normer_list (list): List of Darts TimeSeries that serve as the basis of min-max values to normalise metrics (usually the training set)
    comp_summ_func (function): Function to summarise the normalised per component metrics
    series_summ_func (function): Function to summarise metrics across different series
    metric (function): Darts metric function (see: https://unit8co.github.io/darts/generated_api/darts.metrics.metrics.html)
    """
    result = []
    for curr_obs, curr_preds, curr_normer in zip(obs_list, preds_list, normer_list):
        met_val = norm_metric(curr_obs, curr_preds, curr_normer, summ_func = comp_summ_func, metric = metric, quantiles = quantiles)
        result.append(met_val)
    result = np.array(result)

    if series_summ_func is None:
        return result
    else:
        return series_summ_func(result)

#def generate_component_metrics_table(obs_list, preds_list, label_list, label_name, component_names, normer_list = None, metrics = [mae], metric_names = ["MAE"], disp_mode = False):
def generate_component_metric_table(obs_list, preds_list, label_list, label_name, component_names, curr_metric, normer_list = None, disp_mode = False):
    """
    Function generates a table meant to display the error per state per component
    """
    table_builder = []
    if normer_list is not None: 
        metric_res = norm_metric_list(obs_list, preds_list, normer_list, metric = curr_metric)
    else:
        metric_res = curr_metric(obs_list, preds_list, component_reduction = None)
    
    for curr_label, curr_errors in zip(label_list, metric_res):
        curr_dict = {label_name: curr_label}
        for curr_comp, curr_comp_error in zip(component_names, curr_errors):
            curr_dict[curr_comp] = curr_comp_error
        table_builder.append(curr_dict)
        
    curr_table = pd.DataFrame(table_builder)
    
    if disp_mode:
        disp_table = curr_table.copy()
        summ_row = {label_name: "Summary"}
        for curr_comp in component_names:
            central = np.mean(curr_table[curr_comp])
            lower = np.min(curr_table[curr_comp])
            upper = np.max(curr_table[curr_comp])
            summ_row[curr_comp] = "%.3f (%.3f to %.3f)" % (central, lower, upper)
            disp_table[curr_comp] = disp_table.apply(lambda row: "%.3f" % row[curr_comp], axis = 1)
        disp_table = pd.concat([disp_table, pd.DataFrame([summ_row])])
        return disp_table
    else:
        summ_row = {label_name: "Summary"}
        for curr_comp in component_names:
            central = np.mean(curr_table[curr_comp])
            summ_row[curr_comp] = central
        curr_table = pd.concat([curr_table, pd.DataFrame([summ_row])])
    return curr_table

def ts_list_crps(obs_list, preds_list):
    """
    Takes in two lists of TimeSeries, one with observed and one with predicted values and returns a list of
    CRPS values, one for each pair of series
    """
    results = []
    for curr_obs, curr_preds in zip(obs_list, preds_list):
        curr_preds = curr_preds.slice_intersect(curr_obs)
        curr_obs_vals = curr_obs.all_values().squeeze()
        curr_preds_vals = curr_preds.all_values().squeeze(1)
        curr_res = np.mean(ps.crps_ensemble(curr_obs_vals, curr_preds_vals))
        results.append(curr_res)
    return results
    
def generate_series_metrics_table(obs_list, preds_list, label_list, label_name, metrics = [mae, rmse], metric_names = ["MAE", "RMSE"], normer_list = None, disp_mode = False):
    """
    Function displays multiple error metrics (each the mean across components) per state
    """
    if normer_list is not None:
        metric_names = ["Norm_" + curr_name for curr_name in metric_names]
    curr_table = pd.DataFrame([{label_name: curr_label} for curr_label in label_list])
    for curr_metric, curr_metric_name in zip(metrics, metric_names):
        if normer_list is not None: 
            curr_res = norm_metric_list(obs_list, preds_list, normer_list, comp_summ_func = np.mean, metric = curr_metric)
        else:
            if curr_metric is ps.crps_ensemble:
                curr_res = ts_list_crps(obs_list, preds_list)
            else:
                curr_res = curr_metric(obs_list, preds_list)
        curr_table[curr_metric_name] = curr_res
    
    if disp_mode:
        disp_table = curr_table.copy()
        summ_row = {label_name: "Summary"}
        for curr_metric_name in metric_names:
            central = np.mean(curr_table[curr_metric_name])
            lower = np.min(curr_table[curr_metric_name])
            upper = np.max(curr_table[curr_metric_name])
            summ_row[curr_metric_name] = "%.3f (%.3f to %.3f)" % (central, lower, upper)
            disp_table[curr_metric_name] = disp_table.apply(lambda row: "%.3f" % row[curr_metric_name], axis = 1)
            
        disp_table = pd.concat([disp_table, pd.DataFrame([summ_row])])
        return disp_table
    else:
        summ_row = {label_name: "Summary"}
        for curr_metric_name in metric_names:
            central = np.mean(curr_table[curr_metric_name])
            summ_row[curr_metric_name] = central
        curr_table = pd.concat([curr_table, pd.DataFrame([summ_row])])
        
    return curr_table

def plot_ts_list(plot_list, label_list, plot_comps = None, dims = (7,4), width = 18, height = 24, low_quantile = 0.025, high_quantile = 0.975, prefixes = None):
    """
    Plot a list of Darts TimeSeries objects 
    """
    fig, axs = plt.subplots(dims[0], dims[1], figsize = (width,  height))

    if prefixes is None:
        prefixes = ["" for temp in plot_list]

    for ts_list, curr_prefix in zip(plot_list, prefixes):
        for curr_ts, curr_label, curr_ax in zip(ts_list, label_list, axs.flatten()):
            if plot_comps is not None:
                to_plot = curr_ts[plot_comps]
            else:
                to_plot = curr_ts
        
            to_plot.plot(ax = curr_ax, label = curr_prefix, low_quantile = low_quantile, high_quantile = high_quantile)
            curr_ax.set_title(curr_label)
            curr_ax.set_xlabel("")

def transform_comps_ts(curr_ts, comps_to_trans, trans_func = np.expm1):
    """
    Transforms selected components from a TimeSeries using trans_func (by default, used to reverse log transforms)
    """
    curr_builder = []
    comp_names = curr_ts.components
    for curr_var in comp_names:
        temp = curr_ts[curr_var] 
        if curr_var in comps_to_trans:
            temp = temp.map(trans_func)
            
        curr_builder.append(temp)
    
    result_ts = concatenate(curr_builder, axis = 1)
    return result_ts

def transform_comps_ts_list(curr_ts_list, comps_to_trans, trans_func = np.expm1):
    """
    Transforms selected components for each TimeSeries in a given list using trans_func (by default, used to reverse log transforms)
    """
    to_ret = []
    for curr_ts in curr_ts_list:
        result = transform_comps_ts(curr_ts, comps_to_trans = comps_to_trans, trans_func = trans_func)
        to_ret.append(result)
    return to_ret

def return_to_nat_scale(curr_ts_list_log_s, comps_to_trans, curr_scaler, trans_func = np.expm1):
    """
    Takes a list of TimeSeries, inverse the min-max scaling, then transform certain columns (usually to invert log scaling)
    """
    curr_ts_list_log = curr_scaler.inverse_transform(curr_ts_list_log_s) #Invert the scaler transform
    curr_ts_list = transform_comps_ts_list(curr_ts_list_log, comps_to_trans, trans_func = trans_func) #Invert log transforms
    return curr_ts_list

def save_dict_as_json(curr_json, path):
    with open(path, "w") as f:
        json.dump(curr_json, f, indent = 4)

def read_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data

def save_pickle(to_save, save_path):
    with open(save_path, "wb") as file:
        pickle.dump(to_save, file)

def load_pickle(load_path):
    to_ret = None 
    with open(load_path, "rb") as file:
        to_ret = pickle.load(file)
    return(to_ret)

def df_list_to_excel(df_list, path, sheet_names = None):
    """
    Writes a list of DataFrames as separate sheets in one Excel file given some path.
    """
    if sheet_names is None:
        sheet_names = ["Sheet " + i for i in range(len(df_list))]
    with pd.ExcelWriter(path, engine = "openpyxl") as writer:
        for curr_df, curr_sheet in zip(df_list, sheet_names):
            curr_df.to_excel(writer, sheet_name = curr_sheet, index = False)
            
def ts_to_df(curr_ts, labels, label_names):
    """
    Converts a TimeSeries into a DataFrame and labels it
    """
    curr_df = curr_ts.to_dataframe()
    time_ax_name = curr_df.index.name
    curr_df = curr_df.reset_index()
    comp_names = list(curr_ts.components)
    for curr_label, curr_label_name in zip(labels, label_names):
        curr_df[curr_label_name] = curr_label
        
    curr_df[[time_ax_name] + label_names + comp_names]
    return curr_df

def ts_list_to_df(curr_ts_list, labels, label_names):
    """
    Converts list of TimeSeries into a single DataFrame with proper label columns

    Parameters:
    curr_ts_list (list): List of TimeSeries to convert into a DataFrame
    labels (list): List of lists, where each component list has the same length as curr_ts_list. These are then used to label rows from the generated DataFrame.
    label_names (list): List of label names (should be same length as labels)
    """
    builder = [] 
    for ind, curr_ts in enumerate(curr_ts_list):
        curr_labels = [curr_list[ind] for curr_list in labels]
        curr_df = ts_to_df(curr_ts, curr_labels, label_names)
        builder.append(curr_df)

    comp_names = list(curr_ts_list[0].components)
    time_ax_name = curr_ts_list[0].time_index.name
    to_ret = pd.concat(builder)
    to_ret = to_ret[[time_ax_name] + label_names + comp_names]
    return to_ret

def add_stat_covs(curr_ts_list, stat_covs, matcher_name_list, stat_covs_names, matcher_col = "uf"):
    """
    Takes a list of Darts TimeSeries and returns a list of the same series but with static covariates added in.
    """
    builder = [] 
    for curr_ts, curr_uf in zip(curr_ts_list, matcher_name_list): 
        curr_stat_covs_df = stat_covs[stat_covs[matcher_col] == curr_uf]
        curr_stat_covs_vals = curr_stat_covs_df[stat_covs_names].iloc[0]
        res = curr_ts.with_static_covariates(curr_stat_covs_vals)
        builder.append(res)
        
    return builder
    
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

def ts_list_concatenate(series_lists):
    """
    Returns a list of TimeSeries concatenating each set of series from the input list of lists (aligning on index)
    This assumes all input lists are of the same length
    """
    to_ret = []
    list_len = len(series_lists[0]) #Assume that all the input lists have the same length
    for i in range(list_len):
        to_concat = [curr_list[i] for curr_list in series_lists]
        to_ret.append(concatenate(to_concat))
    return to_ret