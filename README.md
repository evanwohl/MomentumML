# MomentumML: Security Price Prediction Model

## Overview
This repository contains a machine learning model using random forest regression to predict the day-over-day change in the closing price of various securities. The model utilizes momentum indicators for 17 securities, and a trading strategy has been developed and backtested based on the model's predictions.

## Model Description
- **Algorithm**: Random Forest Regression
- **Dataset**: Includes momentum indicators for 17 securities.
- **Training Period**: Starts in 2010
- **Testing Period**: Starts in 2014
- **Data Split**: 70% training, 30% testing

## Model Performance
- **Mean Squared Error**: 5.269296510425714
- **Mean Absolute Error**: 1.4749345441781025
- **Average Prediction Error**: 1.47%

## Actual % Change in Price vs Predicted % Change in Price
![image](https://github.com/evanwohl/MomentumML/assets/156111794/eed62714-6d34-48f1-8e36-2bab0d957eab)

## Strategy Backtest Results
The following metrics were obtained through backtesting the built-in trading strategy:
- **Average Win**: $225.90
- **Average Loss**: -$196.80
- **Win/Loss Ratio**: 1.113

## Portfolio Valuation Plot
Below are the visualizations of the portfolio values resulting from the backtested trading strategy:

![Portfolio Value Plot](https://github.com/evanwohl/MomentumML/assets/156111794/d66c024f-a1e3-4991-a367-daf22ab5e6bf)


## Installation
Ensure that all required dependencies are installed in your Python virtual environment. You can install dependencies via the following command:
```bash
pip install -r requirements.txt
