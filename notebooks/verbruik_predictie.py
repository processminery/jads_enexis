import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plot
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from datetime import datetime

# The MAPE (Mean Absolute Percent Error) measures the size of the error in percentage terms.
def vp_mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    y_true[
        y_true == 0
    ] = 1e-22  # vervang nul waardes door een heel klein getal om delen door 0 te voorkomen
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def calc_accuracy_lr(df_input, split_date=2018):
    """ Bepaal de R2, RMSE, MSE, MAE en MAPE

    Bereken de verschillende nauwkeurigheidsmaten op basis van de input die wordt gesplitst in een train en test set
    gebaseerd op de split datum

    Args:
        df_input: dataframe met de kolommen SJV_TOTAAL, E1A_TOTAAL, E1B_TOTAAL, E1C_TOTAAL, LEVERINGSPERCENTAGE, PC4 en JAAR
        split_date: het jaar waarop gesplitst moet worden (op de JAAR kolom), bijvoorbeeld '2018'

    Returns:
        Een dictionary met de nauwkeurigheidsmaten voor de train en test set
    """
    df_train = df_input[df_input.JAAR < split_date]
    X_train = df_train.JAAR.values
    y_train = df_train.SJV_TOTAAL.values

    df_test = df_input[df_input.JAAR >= split_date]
    X_test = df_test.JAAR.values
    y_test = df_test.SJV_TOTAAL.values

    regressor = linear_model.Lasso()
    regressor.fit(np.array(X_train.reshape(-1, 1)), np.array(y_train.reshape(-1, 1)))
    y_train_pred = regressor.predict(np.array(X_train.reshape(-1, 1)))
    y_test_pred = regressor.predict(np.array(X_test.reshape(-1, 1)))

    return {
        "r2_train": r2_score(y_train, y_train_pred),
        "r2_test": r2_score(y_test, y_test_pred),
        "MSE_train": mean_squared_error(y_train, y_train_pred),
        "MSE test": mean_squared_error(y_test, y_test_pred),
        "RMSE train": np.sqrt(mean_squared_error(y_train, y_train_pred)),
        "RMSE test": np.sqrt(mean_squared_error(y_test, y_test_pred)),
        "MAE train": mean_absolute_error(y_train, y_train_pred),
        "MAE test": mean_absolute_error(y_test, y_test_pred),
        "MAPE train": vp_mean_absolute_percentage_error(y_train, y_train_pred),
        "MAPE test": vp_mean_absolute_percentage_error(y_test, y_test_pred),
    }


def predict_verbruik_lr(df_input, predict_type="mid"):
    """ Voorspel verbruik obv Lineaire regressie

    Deze predictiefunctie traint een lineaire regressiemodel en voorspelt de waarde van SJV_TOTAAL, E1A_TOTAAL, E1B_TOTAAL, E1C_TOTAAL en LEVERINGSPERCENTAGE.

    Args:
        df_input: dataframe met de kolommen SJV_TOTAAL, E1A_TOTAAL, E1B_TOTAAL, E1C_TOTAAL, LEVERINGSPERCENTAGE, PC4 en JAAR
        predict_type: Het predictie type dat wordt bepaald. Low markeert de onzekerheid aan de onderkant en high aan de bovenkant

    Returns:
        Een dataframe met de voorspelling voor 2021-2023

    Raises:
        ValueError: als predict_type niet low, mid of high is, wordt er een ValueError teruggegeven
    """
    # Check of het meegegeven predict_type bekend is
    valid_types = ["low", "mid", "high"]
    if not predict_type in valid_types:
        raise ValueError("predict_type not correctly specified (low, mid, high")

    # Creëer het lineaire regressie model
    regressor = linear_model.Lasso()

    # Creëer een nieuw dataframe wat we zullen vullen met de voorspelling
    df_output = pd.DataFrame(
        columns=[
            "SJV_TOTAAL",
            "E1A_TOTAAL",
            "E1B_TOTAAL",
            "E1C_TOTAAL",
            "AANSLUITINGEN_AANTAL",
            "LEVERINGSRICHTING_PERC",
            "PC4",
            "JAAR",
        ]
    )

    # Creëer een lijst van alle pc4's
    list_of_pc4 = df_input["PC4"].unique()

    # Voorspel voor de jaren 2021-2023
    X_pred = np.array([2021, 2022, 2023]).reshape(-1, 1)

    # Bandbreedte voor de voorspellingen
    low_decr = 0.99
    high_incr = 1.01

    # Bandbreedte bepaling
    multiplier = 1
    if predict_type == "low":
        multiplier = low_decr
    if predict_type == "high":
        multiplier = high_incr

    # Loop door alle pc4's en voorspel 2021-2023
    for pc4 in tqdm(list_of_pc4):
        df_pc4 = df_input[df_input.PC4 == pc4]

        # Skip deze pc4 als er minder dan 9 jaren in zitten
        if len(df_pc4) < 9:
            continue

        # Train het model en maak een forecast voor SJV_TOTAAL
        X = np.array(df_pc4.JAAR.values.reshape(-1, 1))
        y = np.array(df_pc4.SJV_TOTAAL.values.reshape(-1, 1))
        regressor.fit(X, y)
        forecast_totaal = regressor.predict(X_pred)

        # Train het model en maak een forecast voor E1A_TOTAAL
        y = np.array(df_pc4.E1A_TOTAAL.values.reshape(-1, 1))
        regressor.fit(X, y)
        forecast_e1a = regressor.predict(X_pred)

        # Train het model en maak een forecast voor E1B_TOTAAL
        y = np.array(df_pc4.E1B_TOTAAL.values.reshape(-1, 1))
        regressor.fit(X, y)
        forecast_e1b = regressor.predict(X_pred)

        # Train het model en maak een forecast voor E1C_TOTAAL
        y = np.array(df_pc4.E1C_TOTAAL.values.reshape(-1, 1))
        regressor.fit(X, y)
        forecast_e1c = regressor.predict(X_pred)

        # Train het model en maak een forecast voor AANSLUITINGEN_AANTAL
        y = np.array(df_pc4.AANSLUITINGEN_AANTAL.values.reshape(-1, 1))
        regressor.fit(X, y)
        forecast_aantal = regressor.predict(X_pred)

        # Train het model en maak een forecast voor LEVERINGSRICHTING_PERC
        y = np.array(df_pc4.LEVERINGSRICHTING_PERC.values.reshape(-1, 1))
        regressor.fit(X, y)
        forecast_perc = regressor.predict(X_pred)

        # Voeg de voorspellingen toe aan het output dataframe
        for index, jaar in enumerate(X_pred):
            m = pow(multiplier, index + 1)
            df_output = df_output.append(
                {
                    "SJV_TOTAAL": forecast_totaal[index] * m,
                    "E1A_TOTAAL": forecast_e1a[index] * m,
                    "E1B_TOTAAL": forecast_e1b[index] * m,
                    "E1C_TOTAAL": forecast_e1c[index] * m,
                    "AANSLUITINGEN_AANTAL": forecast_aantal[index] * m,
                    "LEVERINGSRICHTING_PERC": forecast_perc[index] * m,
                    "PC4": pc4,
                    "JAAR": jaar[0],
                },
                ignore_index=True,
            )
    df_output.JAAR = df_output.JAAR.astype("int")
    df_output.PC4 = df_output.PC4.astype("int")

    return df_output
