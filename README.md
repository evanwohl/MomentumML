# MomentumML: Security Price Prediction Model

## Overview
This repository contains a machine learning model using random forest regression to predict the day-over-day change in the closing price of various securities. The model utilizes momentum indicators for 17 securities, and a trading strategy has been developed and backtested based on the model's predictions.

## Model Description
- **Algorithm**: Random Forest Regression
- **Dataset**: Includes momentum indicators for 17 securities.
- **Training Period**: Starts in 2010
- **Testing Period**: Starts in 2015
- **Data Split**: 70% training, 30% testing

## Model Performance
(39938 rows of data used for training)
- **Mean Squared Error**: 5.272332061904529
- **Mean Absolute Error**: 1.4767322521351225
## Actual % Change in Price vs Predicted % Change in Price
![image](https://github.com/evanwohl/MomentumML/assets/156111794/eed62714-6d34-48f1-8e36-2bab0d957eab)

## Strategy Backtest Results
The following metrics were obtained through backtesting the built-in trading strategy:
- **Average Win**: $341.57
- **Average Loss**: -$278.74
- **Win/Loss Ratio**: 1.211

Below are the visualizations of the portfolio values resulting from the backtested trading strategy:

![image](https://github.com/evanwohl/MomentumML/assets/156111794/2c0c2b0c-0169-4578-8ab1-1d0d51b1da56)


## Usage
Ensure that all required dependencies are installed in your Python virtual environment.

- Set the following variables to the desired values in the main conditional:

    **date_to_train_from = 2010**
  
    **date_to_test_from = 2015**
- Directly under the initialization of these variables, replace the values inside the list that is being iterated over with the location to the csv files of your data. It should have two columns, one for the date and one for the close price (Date, and Price respecitvely)
- Run **build_model.py**

## License 

This project is licensed under the MIT License - see the LICENSE file for details.
