import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from tensorflow.keras.layers import LSTM, Dropout, Dense, BatchNormalization
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.ensemble import GradientBoostingRegressor
from tensorflow.keras.models import Sequential
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
def load_csv(file_path):
    """
    Load a CSV file into a pandas dataframe
    :param file_path: a string representing the path to the CSV file
    :return: a pandas dataframe containing the data from the CSV file
    """
    return pd.read_csv(file_path, index_col=False)

def calculate_day_over_day_change_percent(df):
    """
    Calculate the day-over-day change percentage of a stock
    :param df: a pandas dataframe with a column named 'Price'
    :return: a pandas dataframe with an additional column named 'Change'
    """
    df['Change'] = (df['Price'].shift(1) - df['Price']) / df['Price'].shift(1) * 100
    return df
def add_boillinger_bands(df, period=20, num_std=2):
    """
    Add Bollinger Bands to a stock's data
    :param df: a pandas dataframe with a column named 'Price'
    :param period: a number representing the period to calculate the SMA
    :param num_std: a number representing the number of standard deviations to calculate the Bollinger Bands
    :return: a pandas dataframe with additional columns named 'SMA', 'Upper Band', and 'Lower Band'
    """
    df['SMA'] = df['Price'].rolling(window=period).mean()
    df['Upper Band'] = df['SMA'] + (df['Price'].rolling(window=period).std() * num_std)
    df['Lower Band'] = df['SMA'] - (df['Price'].rolling(window=period).std() * num_std)
    return df
def drop_nan_rows(df):
    """
    Drop rows with NaN values in the 'Price' column
    :param df: a pandas dataframe with a column named 'Price'
    :return: a pandas dataframe with rows containing NaN values in the 'Price' column removed
    """
    df = df.dropna(subset=['Price'])
    return df
def calculate_rsi(df, period=2):
    """
    Calculate the Relative Strength Index (RSI) of a stock
    :param df: a pandas dataframe with a column named 'Price'
    :param period: a number representing the period to calculate the RSI
    :return: a pandas dataframe with an additional column named 'RSI'
    """
    delta = df['Price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df


def create_model(input_shape):
    """
    Create an LSTM model
    :param input_shape: a tuple representing the input shape of the model
    :return: a compiled LSTM model
    """
    model = Sequential()
    model.add(LSTM(1000, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(500, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(250, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def create_random_forest_model(X_train, y_train, X_test, y_test):
    """
    Create a Random Forest model to predict the change in stock price
    :param X_train: a pandas dataframe with columns for technical indicators
    :param y_train: a pandas series with the target variable
    :param X_test: a pandas dataframe with columns for technical indicators
    :param y_test: a pandas series with the target variable
    :return: a trained Random Forest model
    """
    model = RandomForestRegressor(n_estimators=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    return model

def backtest_strategy(df, magnitude):
    """
    Backtest a trading strategy based on the model's predictions
    :param df: a pandas dataframe with columns for the actual and predicted values
    :param magnitude: a number representing the magnitude of the predicted change to trade on
    :return: a number representing the final capital after backtesting the strategy
    """
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[(df['Date'].dt.year > 2008)]
    df = df.sort_values(by='Date')
    capital = 10000
    value = [10000]
    wins = []
    losses = []
    for date, group in df.groupby('Date'):
        num_trades = len(group[abs(group['Predicted']) > magnitude])
        if num_trades == 0:
            continue
        capital_per_trade = capital / num_trades
        for i in range(len(group)):
            if (group.iloc[i]['Predicted']) > magnitude:
                position = capital_per_trade if group.iloc[i]['Predicted'] > 0 else -(capital_per_trade / 2)
                profit = position * (group.iloc[i]['Actual'] / 100)
                print(f"Date: {date}, Position: {position}, Profit: {profit}")
                capital += profit
                value.append(capital)
                wins.append(profit) if profit > 0 else losses.append(profit)
    print(f"Average win: {sum(wins) / len(wins)}")
    print(f"Average loss: {sum(losses) / len(losses)}")
    print("Win Loss Ratio: ", len(wins) / len(losses))
    print(f"Max Drawdown: {max([((value[i] - max(value[:i])) / max(value[:i])) for i in range(1, len(value))])}")
    return capital, value

def print_results(df_actual_vs_predicted, magnitude):
    """
    Print the results of the model's predictions
    :param df_actual_vs_predicted: a pandas dataframe with columns for the actual and predicted values
    :param magnitude: a number representing the magnitude of the predicted change to trade on
    :return: none
    """
    print(f"Predicted magnitude greater than {str(magnitude)}. Number of times: ",
          len(df_actual_vs_predicted[abs(df_actual_vs_predicted['Predicted']) > magnitude]))
    print(df_actual_vs_predicted[abs(df_actual_vs_predicted['Predicted']) > magnitude].to_string())
    print(
        f"Predicted magnitude greater than {str(magnitude)} but signs are the same and actual is less than {str(magnitude)}")
    print(df_actual_vs_predicted[(abs(df_actual_vs_predicted['Predicted']) > magnitude) & (
                abs(df_actual_vs_predicted['Actual']) < magnitude) & (
                                             df_actual_vs_predicted['Predicted'] * df_actual_vs_predicted[
                                         'Actual'] > 0)])
    print(f"Predicted magnitude greater than {str(magnitude)} but signs are different: (None if empty dataframe)")
    print(df_actual_vs_predicted[(abs(df_actual_vs_predicted['Predicted']) > magnitude) & (
                df_actual_vs_predicted['Predicted'] * df_actual_vs_predicted['Actual'] < 0)])
    print(f"Actual magnitude greater than {str(magnitude)}. Number of times: ",
          len(df_actual_vs_predicted[abs(df_actual_vs_predicted['Actual']) > magnitude]))
    print(f"Number of underpredictions based on magnitude (absolute val) of change: ", len(
        df_actual_vs_predicted[abs(df_actual_vs_predicted['Predicted']) < abs(df_actual_vs_predicted['Actual'])]))
    print(f"overpredictions: ", len(
        df_actual_vs_predicted[abs(df_actual_vs_predicted['Predicted']) > abs(df_actual_vs_predicted['Actual'])]))
    print(f"Number of times the model predicted the correct sign of the change: {len(df_actual_vs_predicted[df_actual_vs_predicted['Predicted'] * df_actual_vs_predicted['Actual'] > 0])}")
    backtest_result, portfolio_value = backtest_strategy(df_actual_vs_predicted, magnitude)
    plt.plot(portfolio_value)
    plt.title('Backtest Results Since 2015 with a 70/30 Train/Test Split')
    plt.xlabel('# Of Trades')
    plt.grid(True)
    plt.show()
    print(f"Backtest result: ${backtest_result} profit with magnitude {(magnitude)}")
    df_actual_vs_predicted['Difference'] = df_actual_vs_predicted['Actual'] - df_actual_vs_predicted['Predicted']
    sns.distplot(df_actual_vs_predicted['Difference'], bins=1000, kde=True)
    plt.title('Distribution of Differences between Actual and Predicted Values')
    plt.xlabel('Difference')
    plt.ylabel('Density')
    plt.show()

def build_model(df):
    """
    Build a Random Forest model to predict the change in stock price
    :param df: a pandas dataframe with columns for technical indicators and the target variable 'Change'
    :return: a trained Random Forest model, the test data, and the test target variable
    """
    df = df.dropna(subset=['Change'])
    X = df.drop(['Change', 'Date'], axis=1)
    y = df['Change']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=105, shuffle=False)
    model = create_random_forest_model(X_train, y_train, X_test, y_test)
    y_pred = model.predict(X_test)
    df_actual_vs_predicted = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    df_actual_vs_predicted['Date'] = df['Date'].iloc[-len(y_test):].values
    magnitude = 1.3
    print_results(df_actual_vs_predicted, magnitude)
    with open('model.pkl', 'wb') as file:
        pickle.dump(model, file)
    return model, X_test, y_test




def build_lstm_model(df):
    """
    Build an LSTM model to predict the change in stock price
    :param df: a pandas dataframe with columns for technical indicators and the target variable 'Change'
    :return: a trained LSTM model, the test data, and the test target variable
    """
    X = df.drop(['Change', 'Date'], axis=1)
    y = df['Change']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10, shuffle=False)
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)
    X_train_selected = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    X_test_selected = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
    model = create_model(X_train_selected.shape[1:])
    model.fit(X_train_selected, y_train, epochs=50, batch_size=32, verbose=1)
    loss = model.evaluate(X_test_selected, y_test, verbose=0)
    print(f'Test loss(MSE): {loss}')
    return model, X_test_selected, y_test


def build_gradient_boosting_model(df):
    """
    Build a Gradient Boosting model to predict the change in stock price
    :param df: a pandas dataframe with columns for technical indicators and the target variable 'Change'
    :return: a trained Gradient Boosting model, the test data, and the test target variable
    """
    X = df.drop(['Change', 'Date'], axis=1)
    y = df['Change']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=106, shuffle=False)
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0, loss='ls')
    model.fit(X_train_scaled, y_train)
    mse = mean_squared_error(y_test, model.predict(X_test_scaled))
    print(f'Test MSE: {mse}')

    return model, X_test_scaled, y_test
def add_macd_indicators(df):
    """
    Add Moving Average Convergence Divergence (MACD) indicators to a stock's data
    :param df: a pandas dataframe with a column named 'Price'
    :return: a pandas dataframe with additional columns named '12 Day EMA', '26 Day EMA', 'MACD', and 'Signal Line'
    """
    df['12 Day EMA'] = df['Price'].ewm(span=12, adjust=False).mean()
    df['26 Day EMA'] = df['Price'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['12 Day EMA'] - df['26 Day EMA']
    df['Signal Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    return df
def add_stochastic_oscillator(df):
    """
    Add Stochastic Oscillator indicators to a stock's data
    :param df: a pandas dataframe with a column named 'Price'
    :return: a pandas dataframe with additional columns named '14 High', '14 Low', '%K', and '%D'
    """
    df['14 High'] = df['Price'].rolling(window=7).max()
    df['14 Low'] = df['Price'].rolling(window=7).min()
    df['%K'] = ((df['Price'] - df['14 Low']) / (df['14 High'] - df['14 Low'])) * 100
    df['%D'] = df['%K'].rolling(window=3).mean()
    return df

def prep_data(data):
    """
    Prepare the data for training the model
    :param data: a pandas dataframe with a column named 'Price'
    :return: a pandas dataframe with additional columns for technical indicators
    """
    data = calculate_rsi(data)
    data = drop_nan_rows(data)
    data = add_boillinger_bands(data)
    data = add_macd_indicators(data)
    data = add_stochastic_oscillator(data)
    data = data.reindex(index=data.index[::-1])
    data = calculate_day_over_day_change_percent(data)
    return data
if __name__ == '__main__':
    data = []
    first_rows = []
    for file in ['nvda.csv', 'spy.csv', 'smci.csv', 'tsla.csv', 'aapl.csv', 'msft.csv', 'amd_data.csv', 'v.csv', 'abbv.csv', 'meta.csv', 'ma.csv', 'amzn.csv', 'goog.csv', 'orcl.csv', 'intc.csv', 'nflx.csv', 'adbe.csv']:
        print(prep_data(load_csv(file)).iloc[0])
        data.append(prep_data(load_csv(file)).iloc[1:])

    data = pd.concat(data)
    print(data)
    model, X_test, y_test = build_model(data)
    # uncomment the following if using LSTM nn
    '''
    loss = model.evaluate(X_test, y_test, verbose=0)
    print(f'Test loss: {loss}')
    y_pred = model.predict(X_test).flatten()
    mse = mean_squared_error(y_test, y_pred)
    print(f'Test MSE: {mse}')
    df_actual_vs_predicted = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    print(df_actual_vs_predicted.to_string())
    df_actual_vs_predicted['Date'] = data['Date'].iloc[-len(y_test):].values
    magnitude = 2
    print(f"Predicted magnitude greater than {str(magnitude)}. Number of times: ", len(df_actual_vs_predicted[abs(df_actual_vs_predicted['Predicted']) > magnitude]))
    print(df_actual_vs_predicted[abs(df_actual_vs_predicted['Predicted']) > magnitude].to_string())
    print(f"Predicted magnitude greater than {str(magnitude)} but signs are the same and actual is less than {str(magnitude)}")
    print(df_actual_vs_predicted[(abs(df_actual_vs_predicted['Predicted']) > magnitude) & (abs(df_actual_vs_predicted['Actual']) < magnitude) & (df_actual_vs_predicted['Predicted'] * df_actual_vs_predicted['Actual'] > 0)])
    print(f"Predicted magnitude greater than {str(magnitude)} but signs are different: (None if empty dataframe)")
    print(df_actual_vs_predicted[(abs(df_actual_vs_predicted['Predicted']) > magnitude) & (df_actual_vs_predicted['Predicted'] * df_actual_vs_predicted['Actual'] < 0)].to_string())
    print(f"Actual magnitude greater than {str(magnitude)}. Number of times: ", len(df_actual_vs_predicted[abs(df_actual_vs_predicted['Actual']) > magnitude]))
    '''
