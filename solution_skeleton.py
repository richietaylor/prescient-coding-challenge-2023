# %%
import numpy as np
import pandas as pd
import datetime
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from concurrent.futures import ProcessPoolExecutor
import plotly.express as px
import concurrent.futures
import darts
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
)
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import time
   # Train individual models for predicting returns of all stocks simultaneously
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from catboost import CatBoostRegressor

print("---Python script Start---", str(datetime.datetime.now()))

# %%

# data reads
df_returns_train = pd.read_csv("data/returns_train.csv")
df_returns_test = pd.read_csv("data/returns_test.csv")
df_returns_train["month_end"] = pd.to_datetime(arg=df_returns_train["month_end"]).apply(
    lambda d: d.date()
)
df_returns_test["month_end"] = pd.to_datetime(arg=df_returns_test["month_end"]).apply(
    lambda d: d.date()
)

# %%


def log(message):
    print(f"{datetime.datetime.now()} - {message}")


def equalise_weights(df: pd.DataFrame):
    """
    Function to generate the equal weights, i.e. 1/p for each active stock within a month

    Args:
        df: A return data frame. First column is month end and remaining columns are stocks

    Returns:
        A dataframe of the same dimension but with values 1/p on active funds within a month

    """

    # create df to house weights
    n_length = len(df)
    df_returns = df
    df_weights = df_returns[:n_length].copy()
    df_weights.set_index("month_end", inplace=True)

    # list of stock names
    list_stocks = list(df_returns.columns)
    list_stocks.remove("month_end")

    # assign 1/p
    df_weights[list_stocks] = 1 / len(list_stocks)

    return df_weights


# %%

def calculate_max_achievable_score(df_test, df_returns):
        max_score = 0

        for i in range(len(df_test)):
            # Get the data up to the current month in the test set
            df_latest = df_returns[(df_returns["month_end"] < df_test.loc[i, "month_end"])].drop(columns=["month_end"])
            actual_returns = df_test.iloc[i].drop("month_end").values

            # Sort the stocks by their actual returns
            sorted_indices = actual_returns.argsort()[::-1]  # Descending order
            sorted_returns = actual_returns[sorted_indices]

            # Select the top N stocks with the highest actual returns
            N = 10  # adjust this parameter as needed
            top_returns = sorted_returns[:N]

            # Calculate the maximum achievable cumulative return
            max_score += sum(top_returns)

        log(f"Maximum achievable score: {max_score}")
        return max_score


class EnsembleStockPredictor:
    def __init__(self, forecast_horizon=3, model_list=None):
        self.forecast_horizon = forecast_horizon
        self.models = model_list if model_list else []

    def fit(self, X_train, y_train):
        # Fit the models
        for model in self.models:
            model.fit(X_train, y_train)
        log("Models trained")

    def predict(self, current_df):
        # Predict returns using each model and average them
        predictions = [model.predict(current_df.iloc[[-1]])[0] for model in self.models]
        predicted_returns = sum(predictions) / len(predictions)
        return predicted_returns
    


def generate_portfolio(df_train: pd.DataFrame, df_test: pd.DataFrame):
    """
    Function to generate stocks weight allocation for time t+1 using historic data. Initial weights generated as 1/p for active stock within a month

    Args:
        df_train: The training set of returns. First column is month end and remaining columns are stocks
        df_test: The testing set of returns. First column is month end and remaining columns are stocks

    Returns:
        The returns dataframe and the weights
    """

    print(
        "---> training set spans",
        df_train["month_end"].min(),
        df_train["month_end"].max(),
    )
    print(
        "---> training set spans",
        df_test["month_end"].min(),
        df_test["month_end"].max(),
    )

    # initialise data
    n_train = len(df_train)
    df_returns = pd.concat(objs=[df_train, df_test], ignore_index=True)

    df_weights = equalise_weights(
        df_returns[:n_train]
    )  # df to store weights and create initial

    # list of stock names
    list_stocks = list(df_returns.columns)
    list_stocks.remove("month_end")

    # <<--------------------- YOUR CODE GOES BELOW THIS LINE --------------------->>
    # Modify the existing code to fix feature name warnings and IndexError

    # Assuming the model is already trained with the initial training set
    log("Training model...")
    X = df_train.drop(columns=["month_end"])
    y = df_train[list_stocks]

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

 

    model_list = [
        # Create individual models
        # RandomForestRegressor(n_estimators=100, random_state=43),
        CatBoostRegressor(n_estimators=100, random_state=42, objective="MultiRMSE",loss_function="MAE",learning_rate=0.1),
        # MLPRegressor(
        #     hidden_layer_sizes=[100],
        #     random_state=42,
        #     early_stopping=True,
        #     learning_rate="adaptive",
        #     max_iter=1000,
        # ),
    ]


    # Define the forecast horizon (e.g., 3 months into the future)
    forecast_horizon = 3
    # Create and train the ensemble predictor
    ensemble_predictor = EnsembleStockPredictor(forecast_horizon=forecast_horizon, model_list=model_list)
    ensemble_predictor.fit(X_train, y_train)


    # Loop through the test set to make weighted allocation decisions
    for i in range(len(df_test)):
        # Get the data up to the current month in the test set
        df_latest = df_returns[
            (df_returns["month_end"] < df_test.loc[i, "month_end"])
        ].drop(columns=["month_end"])

        log(f"Predicting for {df_test.loc[i, 'month_end']}")

        # Initialize a list to hold cumulative returns for each stock and track maximum values
        cumulative_returns = np.zeros(len(list_stocks))
        tracking_active = np.ones(
            len(list_stocks), dtype=bool
        )  # Track whether we're accumulating returns

        # Forecast multiple steps ahead
        current_df = df_latest.copy()
        for step in range(forecast_horizon):
            # Predict returns for all stocks simultaneously using the average of the ensemble models
            predicted_returns = ensemble_predictor.predict(current_df)

            # Accumulate the predicted returns if the stock is still in an "upward phase"
            for j in range(len(list_stocks)):
                if tracking_active[j] and predicted_returns[j] > 0:
                    cumulative_returns[j] += predicted_returns[j]
                elif predicted_returns[j] < 0:
                    # Stop tracking if there's a decrease after an increase
                    tracking_active[j] = False

            # Update the latest data by appending the predicted returns (for the next step)
            new_row = pd.DataFrame([predicted_returns], columns=list_stocks)
            current_df = pd.concat([current_df, new_row], ignore_index=True)

        # Sort the stocks by their cumulative returns
        sorted_indices = cumulative_returns.argsort()[::-1]  # Descending order
        sorted_stocks = [list_stocks[idx] for idx in sorted_indices]

        # Select the top N stocks with the highest cumulative returns
        N = 10  # adjust this parameter as needed
        top_stocks = sorted_stocks[:N]
        top_cumulative_returns = cumulative_returns[sorted_indices][:N]

        # Calculate weights based on normalized cumulative returns with a 10% cap per stock
        total_cumulative_return = sum(top_cumulative_returns)
        weights = [0] * len(list_stocks)
        for j, stock in enumerate(top_stocks):
            idx = list_stocks.index(stock)
            weight = min(top_cumulative_returns[j] / total_cumulative_return, 0.10)
            weights[idx] = weight

        # Adjust weights to ensure the total allocation is 100%
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]

        # Add the current weights to the DataFrame
        df_this = pd.DataFrame(
            data=[[df_test.loc[i, "month_end"]] + weights], columns=df_returns.columns
        )
        df_weights = pd.concat(objs=[df_weights, df_this], ignore_index=True)

    max_score = calculate_max_achievable_score(df_test, df_returns)
    log(f"Maximum achievable score: {max_score}")

    # The updated weights DataFrame (df_weights) now includes allocation decisions based on proportional cumulative returns of the top N stocks with a maximum cap of 10% per stock.
    # <<--------------------- YOUR CODE GOES ABOVE THIS LINE --------------------->>

    # 10% limit check
    # if len(
    #     np.array(df_weights[list_stocks])[np.array(df_weights[list_stocks]) > 0.101]
    # ):

    #     raise Exception(r"---> 10% limit exceeded")

    return df_returns, df_weights


# %%


def plot_total_return(
    df_returns: pd.DataFrame,
    df_weights_index: pd.DataFrame,
    df_weights_portfolio: pd.DataFrame,
):
    """
    Function to generate the two total return indices.

    Args:
        df_returns: Ascending date ordered combined training and test returns data.
        df_weights_index: Index weights. Equally weighted
        df_weights_index: Portfolio weights. Your portfolio should use equally weighted for the training date range. If blank will be ignored

    Returns:
        A plot of the two total return indices and the total return indices as a dataframe
    """

    # list of stock names
    list_stocks = list(df_returns.columns)
    list_stocks.remove("month_end")

    # replace nans with 0 in return array
    ar_returns = np.array(df_returns[list_stocks])
    np.nan_to_num(x=ar_returns, copy=False, nan=0)

    # calc index
    ar_rtn_index = np.array(df_weights_index[list_stocks]) * ar_returns
    ar_rtn_port = np.array(df_weights_portfolio[list_stocks]) * ar_returns

    v_rtn_index = np.sum(ar_rtn_index, axis=1)
    v_rtn_port = np.sum(ar_rtn_port, axis=1)

    # add return series to dataframe
    df_rtn = pd.DataFrame(data=df_returns["month_end"], columns=["month_end"])
    df_rtn["index"] = v_rtn_index
    df_rtn["portfolio"] = v_rtn_port
    df_rtn

    # create total return
    base_price = 100
    df_rtn.sort_values(by="month_end", inplace=True)
    df_rtn["index_tr"] = ((1 + df_rtn["index"]).cumprod()) * base_price
    df_rtn["portfolio_tr"] = ((1 + df_rtn["portfolio"]).cumprod()) * base_price
    df_rtn

    df_rtn_long = df_rtn[["month_end", "index_tr", "portfolio_tr"]].melt(
        id_vars="month_end", var_name="series", value_name="Total Return"
    )

    # plot
    fig1 = px.line(
        data_frame=df_rtn_long, x="month_end", y="Total Return", color="series"
    )

    return fig1, df_rtn


# %%


# running solution
df_returns = pd.concat(objs=[df_returns_train, df_returns_test], ignore_index=True)
df_weights_index = equalise_weights(df_returns)
df_returns, df_weights_portfolio = generate_portfolio(df_returns_train, df_returns_test)
fig1, df_rtn = plot_total_return(
    df_returns,
    df_weights_index=df_weights_index,
    df_weights_portfolio=df_weights_portfolio,
)
fig1.show()


# optimizer_kwargs = {
#     "lr": 0.001,
# }
# common_model_args = {
#     "optimizer_kwargs": optimizer_kwargs,
#     "likelihood": None,  # use a likelihood for probabilistic forecasts
#     "batch_size": 64,
#     "random_state": 42,
#     "hidden_size": 128,
# }
# model = TiDEModel(
#     input_chunk_length=4, output_chunk_length=1, n_epochs=5,  **common_model_args
# )
