# I. Team and Contributors 
### Team Name: Imperial College London

### Members
* [Hadrian Jules Ang](https://profiles.imperial.ac.uk/h.ang21) - School of Public Health, Imperial College London
* [Garyfallos Konstantinoudis](https://profiles.imperial.ac.uk/g.konstantinoudis) - The Grantham Institute for Climate Change, Imperial College London
* [Anna Vicco](https://profiles.imperial.ac.uk/a.vicco21) - School of Public Health, Imperial College London
* [Clare McCormack](https://profiles.imperial.ac.uk/c.mccormack14) - School of Public Health, Imperial College London
* [Ilaria Dorigatti](https://profiles.imperial.ac.uk/i.dorigatti) - School of Public Health, Imperial College London

# II. Repository Structure
<pre>
imperial-mosqlimate-sprint2025/                           
├── Climate Tuning/                         #Tuning results for climate model   
│   ├── Notebook_24_07_2025/  
│   │   ├── LinearRegression/  
│   │   ├── RandomForest/  
│   │   ├── XGBoost/
│   │   ├── tuning_details.json             #Settings for climate model tuning      
├── Dengue Tuning/                          #Tuning and forecast results for dengue model 
│   ├── Tuning_30_07_2025/  
├── ModelInput/                             #Contains CSV files with pre-processed data
├── 0_DataDownloader_PY.ipynb               #Code to download climate data using API
├── 1_DataProcessing.ipynb                  #Code to prepare data for forecasts
├── 2_1_ClimateTuning.ipynb                 #Code to tune climate forecasting models
├── 3_ClimateForecasting.ipynb              #Code to forecast climate used as covariates
├── 4_3_DengueTuning_ForecastGapAuto.py     #Code to tune the dengue forecasting model
├── 5_DengueForecastGeneration.ipynb        #Code to generate dengue forecasts
├── environment.yml                         #Export of conda environment used 
└── README.md   
</pre>

# III. Libraries and Dependencies
Data pre-processing was initially done using the R programming language with the tidyverse (v2.0.0) package. Processed data was then passed to Python for modelling. See the included file environment.yml, which contains an export of the conda environment used to generate all the results in this repository (including all packages necessary and their versions).

# IV. Data and Variables
We used data primarily from five files provided by the Infodengue-Mosqlimate sprint organisers: 
1. *dengue.csv.gz* - for dengue case data, the primary forecast target.
2. *climate.csv.gz* - for climate data used as time-varying covariates with 12 values overall (min, med, max) + (temperature, precipitation, pressure, relative humidity).
3. *environ_vars.csv.gz* - information on Koppen climate classification and Brazilian biomes as static covariates.
4. *datasus_population_2001_2024.csv.gz* - population values used to aggregate climate data to the state-level
5. *map_region_health.csv* - used as a lookup table to match municipalities to their states

*all data from the state ES excluded due to the data issues mentioned by sprint organisers

This repository does not contain copies of the provided 2025 sprint data. This should be downloaded from https://sprint.mosqlimate.org/data/ and placed in the data_sprint_2025 folder for 1_DataProcessing.ipynb to run.

Code in 0_DataDownloader_PY.ipynb was used to download additional climate data from 2025-01-01 to 2025-07-05 from the provided Mosqlimate API. Data from the Oxford COVID-19 government response tracker was also pre-processed, but was unused in the current submitted forecasts [1]. Processing code for this may still be found in 1_DataProcessing.ipynb though we have excluded the raw data from the repository (see https://github.com/OxCGRT/covid-policy-tracker to download the referenced "OxCGRT_compact_subnational_v1.csv" and place it in the "Other Data" directory to run data processing code in this repository). 

First, dengue case data was aggregated from the municipality-level to the state-level, the administrative level required for the dengue forecasts. Municipality-level climate data was aggregated to the state-level using a population weighting scheme. Suppose we have a state $s$ with $n$ municipalities, which we can write as set $s = \{m_1, m_2, m_3 ... m_n\}$. Let $C_{m,t}$ be the value of climate variable $C$ in admin unit $m$ at time $t$, while $Pop\left(m,y(t)\right)$ is the population of admin unit $m$ during the year time $t$ is in, denoted by $y(t)$. Then we have the following equation for the state-level value $C_{s,t}$.

$$C_{s,t} = \frac{1}{Pop(s, y\left(t\right))} \sum_{m_k \in s} Pop\left(m_k, y(t)\right)C_{m_k,t}$$

This equation was used to compute values for the 12 climate variables used as model covariates. Since there were three municipalities which did not have available climate data, we assumed they had the climate values of other municipalities to ensure that population weights in each state added up to 1. This was based on the following geocode matches.
1. 2916104 (Itaparica) = 2933208 (Vera Cruz)
2. 2919926 (Madre de Deus) = 2929206 (Sao Francisco do Conde)
3. 2605459 (Fernando de Noronha) = 2407500 (Maxaranguape)

We applied a similar population weighting scheme for environmental variables used as static covariates (climate classification and biomes), computing the proportion of the population (based on 2017 data) in each state living in each biome and climate classification. See the code in 1_DataProcessing.ipynb for the pre-processing.

No manual variable selection was applied as the temporal fusion transformer (TFT) model we used for dengue forecasting features internal variable selection networks. All precipitation (min, median, max) and dengue case values were transformed with $logp1(x) = log(x+1)$ prior to scaling or model input. Min-max scaling was then applied across all variables to facilitate training. The inverses of these transformations were applied to model outputs to get the final forecasts.

After processing, all the data was split into the following sets.

| Data Set    | Start     | End       |Notes                                                                        |
|-------------|-----------|-----------|-----------------------------------------------------------------------------|
| Hold 1      | 2010      | EW25-2022 |Sprint designated train1                                                     |
| Hold 2      | 2010      | EW25-2023 |Sprint designated train2                                                     |
| Hold 3      | 2010      | EW25-2024 |Sprint designated train3                                                     |
| Target 1    | EW41-2022 | EW40-2023 |Sprint designated target1                                                    |
| Target 2    | EW41-2023 | EW40-2024 |Sprint designated target2                                                    |
| Target 3    | EW41-2024 | EW40-2025 |Sprint designated target3                                                    |
| Train 1     | 2010      | EW10-2021 |Training set for predicting over Target 1                                    |
| Train 2     | 2010      | EW10-2022 |Training set for predicting over Target 2                                    |
| Train 3     | 2010      | EW10-2023 |Training set for predicting over Target 3                                    |
| Val 1       | EW26-2021 | EW25-2022 |Used along with Train1 for early stopping                                    |
| Val 2       | EW26-2022 | EW25-2023 |Used along with Train2 for early stopping                                    |
| Val 3       | EW26-2023 | EW25-2024 |Used along with Train3 for early stopping                                    |
| Train 1a    | 2010      | EW25-2018 |Training set for 1st fold during dengue hyperparameter tuning                |
| Train 1b    | 2010      | EW25-2019 |Training set for 2nd fold during dengue hyperparameter tuning                |
| Train 1c    | 2010      | EW25-2020 |Training set for 3rd fold during dengueh yperparameter tuning                |
| Val 1a      | Data C    | Data D    |Used with Train 1a during tuning for early stopping and validation loss      |
| Val 2b      | Data A    | Data B    |Used with Train 1b during tuning for early stopping and validation loss      |
| Val 3c      | Data C    | Data D    |Used with Train 1c during tuning for early stopping and validation loss      |
| Clim 1a     | 2010      | EW10-2015 |Used during 1st fold to tune the climate forecasting model                   |
| Clim 1b     | 2010      | EW10-2016 |Used during 2nd fold to tune the climate forecasting model                   |
| Clim 1c     | 2010      | EW10-2017 |Used during 3rd fold to tune the climate forecasting model                   |
| ClimVal 1a  | EW26-2015 | EW25-2016 |Used with Clim 1a to evaluate validation loss during hyperparameter tuning   |
| ClimVal 1b  | EW26-2016 | EW25-2017 |Used with Clim 1b to evaluate validation loss during hyperparameter tuning   |
| ClimVal 1c  | EW26-2017 | EW25-2018 |Used with Clim 1c to evaluate validation loss during hyperparameter tuning   |



# V. Model Training
We trained two different models: (1) a random forest regression model to forecast climate variables for use as future covariates, and (2) a temporal fusion transformer (TFT), a deep-learning based model to forecast dengue [2]. For both these models, we used the implementation from the Darts Python package [3]. 

Code in 2_1_ClimateTuning.ipynb shows the process of selecting the best lags hyperparameter to generate forecasts of the weekly state-level climate variables with a random forest model. Tuning was done using the Tree-structured Parzen Estimator (TPE) sampler from the Optuna Python package with data from Clim 1a, 1b, 1c for training and validation loss evaluated using ClimVal 1a, 1b, 1c, respectively [4]. Normalised mean absolute error (MAE) averaged across the three validation sets was used to select the best value for lags through 64 trials. Once the ideal lags value was found (lags = 191), climate forecasts were generated using the code in 3_ClimateForecasting.ipynb.

Code in 4_3_DengueTuning_ForecastGapAuto.py reads observed and forecasted climate values and then uses them to tune hyperparameters for a dengue forecasting TFT model. Like the climate model, tuning was done using the Optuna Python package [4]. A total of 128 trials were executed using 4 TPE samplers running in parallel (each doing 32 trials with separate GPUs for training) while sharing a common results pool for generating hyperparameter proposals. Overall, 8 hyperparameters were tuned: input_chunk_length, output_chunk_length, hidden_size, lstm_layers, num_attention_heads, dropout, learning_rate, and batch_size (see Darts TFT documentation at https://unit8co.github.io/darts/generated_api/darts.models.forecasting.tft_model.html). 

In each trial, a TFT was initially fit to data from Train 1a using quantile regression with a maximum of 100 epochs. Val 1a (and the gap between Train and Val 1a) was used for early stopping with patience = 8, and min_delta = 0.0001 to avoid overfitting. A forecast of length 68 (15 week gap + 1 year validation set [52/53 weeks]) was then generated with Train 1a as the input sequence. Normalised mean quantile loss (MQL) was computed on the forecast versus its intersection with Val 1a. The final model trained on Train 1a was fine tuned using new data from Train 1b plus the last input chunk of 1a concatenated with some fraction of older data such that the number of input/output chunk pairs seen by the model is increased by 40% (to prevent model forgetting). Val 1b was used for early stopping and validation loss computation and this process was repeated with Train 1c and Val 1c. The hyperparameter set that minimised mean normalised MQL across Val 1a, b, and c was used for the main model.

To generate the forecasts, the tuning process was first mirrored. A TFT with the best hyperparameter set found was first trained on Train 1a, then fine tuned with Train 1b, and c (see 5_DengueForecastGeneration.ipynb). The resulting model was then further fine tuned using new data from Train 1, with Val 1 used for early stopping. Hold 1 was then used as an input sequence to generate the forecast for Target 1. This process was repeated, fine tuning on new data from Train 2, forecasting for Target 2 using Hold 2, then fine tuning on Train 3 and forecasting for Target 3 using Hold 3. The forecasts for the three target periods were then submitted after clipping negative values (all between -1 and 0) to 0.

# VI. References

1. Hale T, Angrist N, Goldszmidt R, et al. A global panel database of pandemic policies (Oxford COVID-19 Government Response Tracker). Nat Hum Behav 2021; 5: 529–38.
2. Lim B, Arık SÖ, Loeff N, Pfister T. Temporal Fusion Transformers for interpretable multi-horizon time series forecasting. Int J Forecast 2021; 37: 1748–64.
3. Herzen J, Lässig F, Piazzetta SG, et al. Darts: user-friendly modern machine learning for time series. J Mach Learn Res 2022; 23.
4. Akiba T, Sano S, Yanase T, Ohta T, Koyama M. Optuna: A Next-generation Hyperparameter Optimization Framework. In: Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. New York, NY, USA: Association for Computing Machinery: 2623–31.
  

  
  

# VII. Data Usage Restriction
All predictions were generated in accordance with the data usage restrictions (using only data up to EW 25 of year $t$ to forecast EW40 year $t$ to EW41 year $t+1$). Forecasts were generated through the gap to produce results for target evaluation (i.e. forecast from EW26 of year $t$ to EW41 to year $t+1$ for both dengue and climate).


# VIII. Predictive Uncertainty
The TFT model applied generates probabilistic forecasts using quantile regression. We used the Darts predict function to generate forecasts with 200 samples each, which then served as the basis of the reported uncertainty intervals. 