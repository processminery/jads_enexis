import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import pystan
import fbprophet
from fbprophet import Prophet

# Functie voor de MAPE (Mean Absolute Percent Error)
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Functie voor het aanmaken van features obv datum
def create_features(df, label=None):
    df['date'] = pd.to_datetime(df['DATUM']) # df['DATUM']
    df['dayofweek'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    #df['weekofyear'] = df['date'].dt.weekofyear
    df['weekofyear'] = df['date'].dt.isocalendar().week.astype('int64')

    X = df[['dayofweek', 'month', 'quarter', 'year', 'dayofyear', 'dayofmonth', 'weekofyear']]
    if label:
        y = df[label]
        return X, y
    return X


# Functie voor het trainen van een NEDU profiel model
def train_model_nedu_profielen_xgb(df_nedu, profile, from_date='2010-01-01', split_date='2018-01-01'):
    df_nedu_train = df_nedu.loc[(df_nedu["DATUM"] >= from_date) & (df_nedu["DATUM"] <= split_date)].copy()
    df_nedu_test = df_nedu.loc[df_nedu["DATUM"] > split_date].copy()
    print(f'Length full     dataset: {len(df_nedu)} samples')
    print(f'Length training dataset: {len(df_nedu_train)} samples')
    print(f'Length test     dataset: {len(df_nedu_test)} samples\n')

    # Create training and test datasets
    X_train, y_train = create_features(df_nedu_train, label='VERBRUIKS_FACTOR')
    X_test, y_test = create_features(df_nedu_test, label='VERBRUIKS_FACTOR')

    # Build XGBoost Model
    reg = xgb.XGBRegressor(num_leaves=200,
                           n_estimators=1000,
                           subsample=1.0,
                           min_child_weight=2,
                           max_depth=25,
                           learning_rate=0.05,
                           gamma=0,
                           colsample_bytree=0.95)

    reg.fit(X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=False)

    # Forecast on Test Set
    df_nedu_test['Forecast'] = reg.predict(X_test)
    df_nedu_all = pd.concat([df_nedu_test, df_nedu_train], sort=False)

    # Evaluate the Performance
    # Make predictions using the test set
    y_test_pred = reg.predict(X_test)
    print(f'Measures for Profile {profile}:')
    print(f'R^2   test: {(r2_score(y_test, y_test_pred)):.2f}')
    print(f'MSE   test: {(mean_squared_error(y_test, y_test_pred)):.2f}')
    print(f'RMSE  test: {(np.sqrt(mean_squared_error(y_test, y_test_pred))):.2f}')
    print(f'MAE   test: {(mean_absolute_error(y_test, y_test_pred)):.2f}')
    print(f'MAPE  test: {(mean_absolute_percentage_error(y_test, y_test_pred)):.2f}')
    print(f'MAPA  test: {100 - mean_absolute_percentage_error(y_test, y_test_pred):.2f}%\n')

    # save the trained model as pickle file
    filename_model = 'NEDU_XGB_' + profile + '_' + split_date + '.pkl'
    models_location = '../../trained_models'
    if 'trained_models' not in os.getcwd():
        os.chdir(models_location)
    joblib.dump(reg, filename_model)
    print(f'Model saved: {filename_model}\n')


# Functie voor het maken van een prediction voor een NEDU profiel
def predict_nedu_profielen_xgb(model, start='2022-01-01', end='2024-12-31', predict_type='mid'):
    # load model
    if 'trained_models' not in os.getcwd():
        os.chdir(models_location)
    reg = joblib.load(model)
    # Forecast into future
    df_forecast = pd.DataFrame(pd.date_range(start=start, end=end, freq='D'),
                               columns=['DATUM']).sort_values(by='DATUM', ascending=True)
    df_forecast['VERBRUIKS_FACTOR'] = 0.0
    X_forecast, _ = create_features(df_forecast, label='VERBRUIKS_FACTOR')
    df_forecast['VERBRUIKS_FACTOR'] = reg.predict(X_forecast)

    return df_forecast


# Functie voor training Prophet model
def train_model_nedu_profielen_prophet(df_nedu, profile, from_date='2010-01-01', split_date='2018-01-01'):
    # rename columns
    df_nedu = df_nedu.rename(columns={"DATUM": "ds", "VERBRUIKS_FACTOR": "y"})
    df_nedu["ds"] = pd.to_datetime(df_nedu['ds'])
    df_nedu_train = df_nedu.loc[(df_nedu["ds"] >= from_date) & (df_nedu["ds"] <= split_date)].copy()
    df_nedu_test = df_nedu.loc[df_nedu["ds"] > split_date].copy()
    print(f'Length full     dataset: {len(df_nedu)} samples')
    print(f'Length training dataset: {len(df_nedu_train)} samples')
    print(f'Length test     dataset: {len(df_nedu_test)} samples\n')

    # We fit the model by instantiating a new Prophet object.
    # Any settings to the forecasting procedure are passed into the constructor.
    # Then you call its fit method and pass in the historical dataframe. Fitting should take 1-5 seconds.
    m = Prophet(growth='linear',
                changepoints=None,
                n_changepoints=2,  # 25,
                changepoint_range=0.4,  # 0.8,
                yearly_seasonality=True,  # 'auto',
                weekly_seasonality=True,  # 'auto',
                daily_seasonality=False,  # 'auto',
                holidays=None,  # holidays,
                seasonality_mode='additive',
                seasonality_prior_scale=10.0,
                holidays_prior_scale=10.0,
                changepoint_prior_scale=0.05,
                mcmc_samples=0,
                interval_width=0.8,
                uncertainty_samples=1000,
                stan_backend=None)

    m.add_country_holidays(country_name='NL')  # holidays van Nederland toevoegen

    m.fit(df_nedu_train)

    # Forecast on Test Set
    X_test = pd.DataFrame(df_nedu_test['ds'])
    forecast = m.predict(X_test)
    df_nedu_test = pd.merge(df_nedu_test, forecast, on='ds', how='left')
    df_nedu_all = pd.concat([df_nedu_test, df_nedu_train], sort=False)

    # Evaluate the Performance
    # Make predictions using the test set
    y_test = df_nedu_test['y']
    y_test_pred = df_nedu_test['yhat']

    print(f'Measures for Profile {profile}:')
    print(f'R^2   test: {(r2_score(y_test, y_test_pred)):.2f}')
    print(f'MSE   test: {(mean_squared_error(y_test, y_test_pred)):.2f}')
    print(f'RMSE  test: {(np.sqrt(mean_squared_error(y_test, y_test_pred))):.2f}')
    print(f'MAE   test: {(mean_absolute_error(y_test, y_test_pred)):.2f}')
    print(f'MAPE  test: {(mean_absolute_percentage_error(y_test, y_test_pred)):.2f}')
    print(f'MAPA  test: {100 - mean_absolute_percentage_error(y_test, y_test_pred):.2f}%\n')

    # save the trained model as pickle file
    filename_model = 'NEDU_PRO_' + profile + '_' + split_date + '.pkl'
    models_location = '../../trained_models'
    if 'trained_models' not in os.getcwd():
        os.chdir(models_location)
    joblib.dump(m, filename_model)
    print(f'Model saved: {filename_model}\n')


# Functie voor het maken van een prediction voor een NEDU profiel
def predict_nedu_profielen_prophet(model, start='2022-01-01', end='2024-12-31', predict_type='mid'):
    # load model
    if 'trained_models' not in os.getcwd():
        os.chdir(models_location)
    m = joblib.load(model)
    # Forecast into future
    df = pd.DataFrame(pd.date_range(start=start, end=end, freq='D'),
                      columns=['ds']).sort_values(by='ds', ascending=True)
    df_forecast = m.predict(df)

    if predict_type == 'low':
        df = df_forecast[['ds', 'yhat_lower']].rename(columns={"ds": "DATUM", "yhat_lower": "VERBRUIKS_FACTOR"})

    if predict_type == 'mid':
        df = df_forecast[['ds', 'yhat']].rename(columns={"ds": "DATUM", "yhat": "VERBRUIKS_FACTOR"})

    if predict_type == 'high':
        df = df_forecast[['ds', 'yhat_upper']].rename(columns={"ds": "DATUM", "yhat_upper": "VERBRUIKS_FACTOR"})

    return df