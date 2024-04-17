Our final project for STAT486: Market Models consisted of creating a machine learning model to predict day over day change in close price of various different securities. Originally, we attempted to create a LSTM neural network, however it proved to be ineffective due to issues with redundancy as momentum indicators inherently represent time series data. Thus, we decided to create a random forest regression model and created/backtested a trading strategy based on the results produced. 
The following plot depicts the portfolio value resulting from backtesting the built in strategy, with training data starting in 2010 and test data starting in 2014 with a set magnitude of 1.3. (It is important to keep in mind that the test split contains only 30% of the total data, so it is hard to compute a number for alpha, sharpe ratio, etc.). 
![image](https://github.com/evanwohl/MomentumML/assets/156111794/d66c024f-a1e3-4991-a367-daf22ab5e6bf)

![image](https://github.com/evanwohl/MomentumML/assets/156111794/eed62714-6d34-48f1-8e36-2bab0d957eab)

