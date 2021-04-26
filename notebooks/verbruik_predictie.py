import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plot
from sklearn import linear_model


def predict_verbruik_lr(df_input, predict_type="mid"):
    """ Voorspel verbruik obv Lineaire regressie

    Deze predictiefunctie traint een lineaire regressiemodel en voorspelt de waarde van SJV_TOTAAL, E1A_TOTAAL, E1B_TOTAAL, E1C_TOTAAL en LEVERINGSPERCENTAGE.

    Args:
        df_input: dataframe met de kolommen SJV_TOTAAL, E1A_TOTAAL, E1B_TOTAAL, E1C_TOTAAL, LEVERINGSPERCENTAGE, PC4 en JAAR
        predict_type:

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
            df_output = df_output.append(
                {
                    "SJV_TOTAAL": forecast_totaal[index],
                    "E1A_TOTAAL": forecast_e1a[index],
                    "E1B_TOTAAL": forecast_e1b[index],
                    "E1C_TOTAAL": forecast_e1c[index],
                    "AANSLUITINGEN_AANTAL": forecast_aantal[index],
                    "LEVERINGSRICHTING_PERC": forecast_perc[index],
                    "PC4": pc4,
                    "JAAR": jaar[0],
                },
                ignore_index=True,
            )

    df_output.JAAR = df_output.JAAR.astype("int")
    df_output.PC4 = df_output.PC4.astype("int")
    return df_output
