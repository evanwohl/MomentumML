Machine learning model to predict day over day change in close price of various different securities. A random forest regression model was trained over a dataset containing momentum indicatores for 17 seucurities, and a trading strategy was created/backtested based on the results produced. 

**Model Mean Squared Error: 5.269296510425714**

**Model Mean Absolute Error: 1.4749345441781025**

**The model's predictions are, on average, off by 1.47%.**

The following plot depicts the portfolio value resulting from backtesting the built in strategy, with training data starting in 2010 and test data starting in 2014 with a set magnitude of 1.3. (It is important to keep in mind that the test split contains only 30% of the total data, so it is hard to compute a number for alpha, sharpe ratio, etc.). 
![image](https://github.com/evanwohl/MomentumML/assets/156111794/d66c024f-a1e3-4991-a367-daf22ab5e6bf)

**Average win: 225.8955038527688**

**Average loss: -196.80185718120578**

**Win Loss Ratio:  1.1132075471698113**
![image](https://github.com/evanwohl/MomentumML/assets/156111794/eed62714-6d34-48f1-8e36-2bab0d957eab)

**Usage**

Ensure that all required dependencies are installed in your virtual enviroment.

Set the following variables to the desired values in the **if name == '__main___'** conditional:

**date_to_train_from = 2010**

**date_to_test_from = 2015**

Directly under the initialization of these variables, replace the values inside the list that is being iterated over with the location to the csv files of your data. It should have two columns, one for the date and one for the close price (called Date and Price respecitvely). 
