import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plot
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from datetime import datetime
import math
from fbprophet import Prophet
from fbprophet.plot import plot_plotly, plot_components_plotly
from multiprocessing import Pool
from multiprocessing import Process
from tqdm import trange

sum = 0


def vp_mean_absolute_percentage_error(y_true, y_pred):
    global sum
    for i, y in enumerate(y_true):
        sum = sum if i > 0 else 0
        y = y if y > 0 else 1e-12
        portion = abs((y - y_pred[i]) / y)
        sum = sum + abs((y - y_pred[i]) / y)
    N = i + 1
    return sum / N * 100


def vp_add_feature_lr(X, power: int = 2):
    """ Voeg feature toe aan de X

    Voeg een feature toe op basis van de bestaande X-en. Voorbeelden:
    
    X = np.array([2021, 2022, 2023])
    X = vp_add_feature_lr(X, feature = None)

    geeft [[2021], [2022], [2023]]

    Args:
        X: np.array met daarin de X-en, bijvoorbeeld [2020, 2021, 2022]
        power:  integer die specificeert hoeveel termen er toegevoegd moeten worden aan de X.  
                Voorbeeld:  power = 2, dan wordt het array [x, x^2]  
                            power = 3, dan wordt het array [x, x^2, x^3]

    Returns:
        Een np.array met de vorm [[x,pow(x,2),power(x,3),...], [x,pow(x,2),power(x,3),...], ...]
    """
    X_out = []
    for x in X:
        if power:
            term = []
            for i in range(1, power + 1):
                term.append(pow(x, i))
            X_out.append(term)
        else:
            X_out.append([x])
    return np.array(X_out)


def calc_accuracy_lr(df_input, split_year=2019, power: int = 2, model="lasso"):
    """ Bepaal de R2, RMSE, MSE, MAE en MAPE

    Bereken de verschillende nauwkeurigheidsmaten voor lineaire regressie op basis van de input die wordt gesplitst in een train en test set
    gebaseerd op de split datum

    Args:
        df_input: dataframe met de kolommen SJV_TOTAAL, E1A_TOTAAL, E1B_TOTAAL, E1C_TOTAAL, LEVERINGSPERCENTAGE, PC4 en JAAR
        split_date: het jaar waarop gesplitst moet worden (op de JAAR kolom), bijvoorbeeld '2018'
        power:  Voeg extra termen toe in de vorm pow(x, power). power = 1 betekent dat alleen x wordt gebruikt in de reguliere expressie functie
        model:  None - Standaard lineaire regressie (least squares)
                Lasso - Lasso model

    Returns:
        Een object met als attributen de nauwkeurigheidsmaten voor de train en test set. Daarnaast bevat het object ook het gefitte model (model) 
        en de train en test sets (df_train en df_test)
    """
    df_train = df_input[df_input.JAAR < split_year]
    X_train = df_train.JAAR.values
    X_train = vp_add_feature_lr(X_train, power)
    y_train = df_train.SJV_TOTAAL.values

    df_test = df_input[df_input.JAAR >= split_year]
    X_test = df_test.JAAR.values
    X_test = vp_add_feature_lr(X_test, power)
    y_test = df_test.SJV_TOTAAL.values

    if model == "lasso":
        regressor = linear_model.Lasso()
    else:
        regressor = linear_model.LinearRegression()
    print(f"Using a {model} model", end=" ")
    if power:
        if power == 1:
            print(f"with 1 power term")
        else:
            print(f"with {power} power terms")
    else:
        print("with no additional feature")
    print("", flush=True)

    regressor = regressor.fit(X_train, y_train)
    y_train_pred = regressor.predict(X_train)
    df_train["SJV_TOTAAL_PRED"] = y_train_pred
    y_test_pred = regressor.predict(X_test)
    df_test["SJV_TOTAAL_PRED"] = y_test_pred

    # Creëer een object wat alle benodigde parameters bevat
    obj_out = type(
        "calculation_parameters",
        (),
        dict(
            R2_train=r2_score(y_train, y_train_pred),
            R2_test=r2_score(y_test, y_test_pred),
            MSE_train=mean_squared_error(y_train, y_train_pred),
            MSE_test=mean_squared_error(y_test, y_test_pred),
            RMSE_train=np.sqrt(mean_squared_error(y_train, y_train_pred)),
            RMSE_test=np.sqrt(mean_squared_error(y_test, y_test_pred)),
            MAE_train=mean_absolute_error(y_train, y_train_pred),
            MAE_test=mean_absolute_error(y_test, y_test_pred),
            MAPE_train=vp_mean_absolute_percentage_error(y_train, y_train_pred),
            MAPE_test=vp_mean_absolute_percentage_error(y_test, y_test_pred),
            df_train=df_train,
            df_test=df_test,
            model=regressor,
        ),
    )
    return obj_out


def predict_verbruik_lr(df_input, predict_type="mid", power: int = 2, model="lasso"):
    """ Voorspel verbruik obv Lineaire regressie

    Deze predictiefunctie traint een lineair regressiemodel en voorspelt de waarde van SJV_TOTAAL, E1A_TOTAAL, E1B_TOTAAL, E1C_TOTAAL en LEVERINGSPERCENTAGE.

    Args:
        df_input: dataframe met de kolommen SJV_TOTAAL, E1A_TOTAAL, E1B_TOTAAL, E1C_TOTAAL, LEVERINGSPERCENTAGE, PC4 en JAAR  
        predict_type: Het predictie type dat wordt bepaald. Low markeert de onzekerheid aan de onderkant en high aan de bovenkant  
        power:  Voeg extra termen toe in de vorm pow(x, power). power = 1 betekent dat alleen x wordt gebruikt in de reguliere expressie functie
        model:  None - Standaard lineaire regressie (least squares)  
                Lasso - Lasso model  

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
    if model == "lasso":
        regressor = linear_model.Lasso()
    else:
        regressor = linear_model.LinearRegression()
    print(f"Using a {model} model", end=" ")
    if power:
        if power == 1:
            print(f"with 1 power term")
        else:
            print(f"with {power} power terms")
    else:
        print("with no additional feature")
    print("", flush=True)

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
    # voeg feature toe
    X_pred = np.array([2021, 2022, 2023])
    X_pred = vp_add_feature_lr(X_pred, power)

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

        # Creëer de features
        X = df_pc4.JAAR.values
        X = vp_add_feature_lr(X, power)

        # Train het model en maak een forecast voor SJV_TOTAAL
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


def create_fb_prophet_obj():
    # Creëer een facebook prophet object
    return Prophet(
        growth="linear",
        changepoints=None,
        n_changepoints=2,  # 25,
        changepoint_range=0.4,  # 0.8,
        yearly_seasonality=True,  #'auto',
        weekly_seasonality=False,  #'auto',
        daily_seasonality=False,  #'auto',
        holidays=None,  # holidays,
        seasonality_mode="additive",
        seasonality_prior_scale=10.0,
        holidays_prior_scale=10.0,
        changepoint_prior_scale=0.05,
        mcmc_samples=0,
        interval_width=0.8,
        uncertainty_samples=1000,
        stan_backend=None,
    )


pool = None


def fit_en_predict(ds, pred, *y):
    """ Voer de fit en predict parallel uit voor de verschillende y variabelen

    Voor de paralle uitvoering wordt een Pool gebruikt uit de multiprocessing module

    Args:
        ds:     De datestamps voor het FB prophet model
        pred:   Een dataframe met een ds kolom waarin de te voorspellen datestamps zitten
        *y:     Een of meerder y-variabelen die gefit en voorspeld moeten worden

    """
    global pool

    # Gebruik meerdere threads om alle fits parallel uit te voeren
    # Een pool is relatief kostbaar in tijd om op te zetten dus herbruik de pool wanneer mogelijk
    pool = Pool(len(y)) if pool is None else pool

    # Gebruik een generator om alle functie calls te genereren die parallel worden uitgevoerd
    results = [pool.apply(internal_fit_en_predict, args=(ds, yvar, pred)) for yvar in y]

    # Geef de resultaten terug als een tuple van dataframes
    return results


def internal_fit_en_predict(ds, y, pred):
    """ Doe een fit en predict met FB prophet

    Args:
        ds:     De datestamps voor het FB prophet model
        y:      De y-variabelen om te fitten
        pred:   Een dataframe met een ds kolom waarin de te voorspellen datestamps zitten

    Returns:
        Het gevulde predictie dataframe zoals dit door het FB prophet model wordt gemaakt
    """
    # Bouw het input frame voor deze y-variabele
    df_input = pd.DataFrame(columns=["ds", "y"])
    df_input.ds = ds
    df_input.y = y

    # Doe de fit en maak de voorspelling
    model = create_fb_prophet_obj()
    model.fit(df_input)
    return model.predict(pred)


def predict_verbruik_fb_prophet(df_input):
    """ Maak een fb prophet voorspelling voor 2021-2023

    De functie maakt een voorspelling per PC4 en voegt deze samen in 1 dataframe. De voortgang wordt weergegeven mbv de tqdm module.
    Bij elke voorspelling zit ook een onderkant en bovenkant van de voorspelde waarde (yhat_lower, yhat_upper). Deze worden in 
    afzonderlijke dataframes teruggegeven

    Args:
        df_input:   Een kleinverbruik dataframe op PC4 niveau met de kolommen:  
                    PC4, JAAR, SJV_TOTAAL, E1A_TOTAAL, E1B_TOTAAL, E1C_TOTAAL, AANSLUITINGEN_AANTAL, LEVERINGSRICHTING_PERC

    Returns:
        Een tuple van 3 dataframes (low, mid, high) met daarin de voorspelde waardes voor bovenstaande y-kolommen voor 2021-2023
    """
    # Creëer een nieuw dataframe wat we zullen vullen met de voorspelling
    df_output_low = pd.DataFrame(
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
    df_output_mid = df_output_low.copy()
    df_output_high = df_output_low.copy()

    # Creëer een lijst van alle pc4's
    list_of_pc4 = df_input["PC4"].unique()

    # Voorspel voor de jaren 2021-2023
    X_pred = np.array([2021, 2022, 2023])
    df_predict = pd.DataFrame(columns=["ds"])
    df_predict.ds = pd.to_datetime(X_pred, format="%Y")

    t_pc4 = tqdm(list_of_pc4)
    for pc4 in t_pc4:
        # Voeg pc4 aan output toe, zodat we kunnen zien voor welke pc4 er een voorspelling gemaakt wordt
        t_pc4.set_description(f"PC4 = {pc4}")

        # Selecteer alle rijen voor deze pc4
        df_pc4 = df_input[df_input.PC4 == pc4]

        # Skip deze pc4 als er minder dan 9 jaren in zitten
        if len(df_pc4) < 9:
            continue

        # Fit het model en maak een voorspelling voor de 6 kolommen
        # De return values komen in een tuple van 6 dataframes terecht
        ds = pd.to_datetime(df_pc4.JAAR, format="%Y")
        (
            df_predict_totaal,
            df_predict_e1a,
            df_predict_e1b,
            df_predict_e1c,
            df_predict_aantal,
            df_predict_perc,
        ) = fit_en_predict(
            ds,
            df_predict,
            df_pc4.SJV_TOTAAL,
            df_pc4.E1A_TOTAAL,
            df_pc4.E1B_TOTAAL,
            df_pc4.E1C_TOTAAL,
            df_pc4.AANSLUITINGEN_AANTAL,
            df_pc4.LEVERINGSRICHTING_PERC,
        )

        # Bouw de output dataframes voor low, mid en high
        for index, jaar in enumerate(X_pred):
            df_output_low = df_output_low.append(
                {
                    "SJV_TOTAAL": df_predict_totaal.yhat_lower[index],
                    "E1A_TOTAAL": df_predict_e1a.yhat_lower[index],
                    "E1B_TOTAAL": df_predict_e1b.yhat_lower[index],
                    "E1C_TOTAAL": df_predict_e1c.yhat_lower[index],
                    "AANSLUITINGEN_AANTAL": df_predict_aantal.yhat_lower[index],
                    "LEVERINGSRICHTING_PERC": df_predict_perc.yhat_lower[index],
                    "PC4": pc4,
                    "JAAR": jaar,
                },
                ignore_index=True,
            )
            df_output_mid = df_output_mid.append(
                {
                    "SJV_TOTAAL": df_predict_totaal.yhat[index],
                    "E1A_TOTAAL": df_predict_e1a.yhat[index],
                    "E1B_TOTAAL": df_predict_e1b.yhat[index],
                    "E1C_TOTAAL": df_predict_e1c.yhat[index],
                    "AANSLUITINGEN_AANTAL": df_predict_aantal.yhat[index],
                    "LEVERINGSRICHTING_PERC": df_predict_perc.yhat[index],
                    "PC4": pc4,
                    "JAAR": jaar,
                },
                ignore_index=True,
            )
            df_output_high = df_output_high.append(
                {
                    "SJV_TOTAAL": df_predict_totaal.yhat_upper[index],
                    "E1A_TOTAAL": df_predict_e1a.yhat_upper[index],
                    "E1B_TOTAAL": df_predict_e1b.yhat_upper[index],
                    "E1C_TOTAAL": df_predict_e1c.yhat_upper[index],
                    "AANSLUITINGEN_AANTAL": df_predict_aantal.yhat_upper[index],
                    "LEVERINGSRICHTING_PERC": df_predict_perc.yhat_upper[index],
                    "PC4": pc4,
                    "JAAR": jaar,
                },
                ignore_index=True,
            )

    return df_output_low, df_output_mid, df_output_high

