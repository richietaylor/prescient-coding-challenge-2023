# %%
import numpy as np
import pandas as pd
import datetime
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from concurrent.futures import ProcessPoolExecutor
import plotly.express as px
import concurrent.futures

from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ...
# ETS-based prediction strategy
def predict_return(stock, df_latest, trend, damped_trend):
    model = ExponentialSmoothing(df_latest[stock], trend=trend, damped_trend=damped_trend)
    model_fit = model.fit()
    predicted_return = model_fit.predict(start=len(df_latest), end=len(df_latest))
    return predicted_return.iloc[0]


print('---Python script Start---', str(datetime.datetime.now()))

# %%

# data reads
df_returns_train = pd.read_csv('data/returns_train.csv')
df_returns_test = pd.read_csv('data/returns_test.csv')
df_returns_train['month_end'] = pd.to_datetime(arg=df_returns_train['month_end']).apply(lambda d: d.date())
df_returns_test['month_end'] = pd.to_datetime(arg=df_returns_test['month_end']).apply(lambda d: d.date())

# %%

def equalise_weights(df: pd.DataFrame):

    '''
        Function to generate the equal weights, i.e. 1/p for each active stock within a month

        Args:
            df: A return data frame. First column is month end and remaining columns are stocks

        Returns:
            A dataframe of the same dimension but with values 1/p on active funds within a month

    '''

    # create df to house weights
    n_length = len(df)
    df_returns = df
    df_weights = df_returns[:n_length].copy()
    df_weights.set_index('month_end', inplace=True)

    # list of stock names
    list_stocks = list(df_returns.columns)
    list_stocks.remove('month_end')

    # assign 1/p
    df_weights[list_stocks] = 1/len(list_stocks)

    return df_weights


# %%

def generate_portfolio(df_train: pd.DataFrame, df_test: pd.DataFrame):

    '''
        Function to generate stocks weight allocation for time t+1 using historic data. Initial weights generated as 1/p for active stock within a month

        Args:
            df_train: The training set of returns. First column is month end and remaining columns are stocks
            df_test: The testing set of returns. First column is month end and remaining columns are stocks

        Returns:
            The returns dataframe and the weights
    '''

    print('---> training set spans', df_train['month_end'].min(), df_train['month_end'].max())
    print('---> training set spans', df_test['month_end'].min(), df_test['month_end'].max())

    # initialise data
    n_train = len(df_train)
    df_returns = pd.concat(objs=[df_train, df_test], ignore_index=True)

    df_weights = equalise_weights(df_returns[:n_train]) # df to store weights and create initial

    # list of stock names
    list_stocks = list(df_returns.columns)
    list_stocks.remove('month_end')

    # <<--------------------- YOUR CODE GOES BELOW THIS LINE --------------------->>

    
    # ARIMA-based prediction strategy
    # Loop through the test set
    for i in range(len(df_test)):
        # Get the data up to the current month in the test set
        df_latest = df_returns[(df_returns['month_end'] < df_test.loc[i, 'month_end'])]

        # Initialize a dictionary to store the predicted returns
        predicted_returns = {}

        # Fit the ETS model and make predictions for each stock
        for stock in list_stocks:
            print(f"Processing {stock}...")
            predicted_return = predict_return(stock, df_latest, trend='add', damped_trend=True)
            predicted_returns[stock] = predicted_return

        # Sort the stocks by their predicted returns
        sorted_stocks = sorted(predicted_returns, key=predicted_returns.get, reverse=True)

        # Select the top N stocks with the highest predicted returns
        N = 10  # adjust this parameter as needed
        top_stocks = sorted_stocks[:N]

        # Assign equal weights to the top stocks and zero weight to the other stocks
        weights = [1/N if stock in top_stocks else 0 for stock in list_stocks]

        # Add the current weights to the DataFrame
        df_this = pd.DataFrame(data=[[df_test.loc[i, 'month_end']] + weights], columns=df_latest.columns)
        df_weights = pd.concat(objs=[df_weights, df_this], ignore_index=True)

    
    # <<--------------------- YOUR CODE GOES ABOVE THIS LINE --------------------->>
    
    # 10% limit check
    if len(np.array(df_weights[list_stocks])[np.array(df_weights[list_stocks]) > 0.101]):

        raise Exception(r'---> 10% limit exceeded')

    return df_returns, df_weights


# %%


def plot_total_return(df_returns: pd.DataFrame, df_weights_index: pd.DataFrame, df_weights_portfolio: pd.DataFrame):

    '''
        Function to generate the two total return indices.

        Args:
            df_returns: Ascending date ordered combined training and test returns data.
            df_weights_index: Index weights. Equally weighted
            df_weights_index: Portfolio weights. Your portfolio should use equally weighted for the training date range. If blank will be ignored

        Returns:
            A plot of the two total return indices and the total return indices as a dataframe
    '''

    # list of stock names
    list_stocks = list(df_returns.columns)
    list_stocks.remove('month_end')

    # replace nans with 0 in return array
    ar_returns = np.array(df_returns[list_stocks])
    np.nan_to_num(x=ar_returns, copy=False, nan=0)

    # calc index
    ar_rtn_index = np.array(df_weights_index[list_stocks])*ar_returns
    ar_rtn_port = np.array(df_weights_portfolio[list_stocks])*ar_returns

    v_rtn_index = np.sum(ar_rtn_index, axis=1)
    v_rtn_port = np.sum(ar_rtn_port, axis=1)

    # add return series to dataframe
    df_rtn = pd.DataFrame(data=df_returns['month_end'], columns=['month_end'])
    df_rtn['index'] = v_rtn_index
    df_rtn['portfolio'] = v_rtn_port
    df_rtn

    # create total return
    base_price = 100
    df_rtn.sort_values(by = 'month_end', inplace = True)
    df_rtn['index_tr'] = ((1 + df_rtn['index']).cumprod()) * base_price
    df_rtn['portfolio_tr'] = ((1 + df_rtn['portfolio']).cumprod()) * base_price
    df_rtn

    df_rtn_long = df_rtn[['month_end', 'index_tr', 'portfolio_tr']].melt(id_vars='month_end', var_name='series', value_name='Total Return')

    # plot
    fig1 = px.line(data_frame=df_rtn_long, x='month_end', y='Total Return', color='series')

    return fig1, df_rtn

# %%





# running solution
df_returns = pd.concat(objs=[df_returns_train, df_returns_test], ignore_index=True)
df_weights_index = equalise_weights(df_returns)
df_returns, df_weights_portfolio = generate_portfolio(df_returns_train, df_returns_test)
fig1, df_rtn = plot_total_return(df_returns, df_weights_index=df_weights_index, df_weights_portfolio=df_weights_portfolio)
fig1.show()
