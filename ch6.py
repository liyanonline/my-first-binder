# !pip3 install pandas sklearn statsmodels dmba matplotlib

# %matplotlib inline

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge,LassoCV,BayesianRidge
import statsmodels.formula.api as sm

from dmba import regressionSummary, exhaustive_search
from dmba import backward_elimination, forward_selection,stepwise_selection
from dmba import adjusted_r2_score, AIC_score, BIC_score


# UserWarning: Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure. plt.show()
# sudo apt-get install python3-tk
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# reduce data frame to the top 1000 rows and select columns for regression analysis
# car_df = pd.read_csv('/opt/notebooks/Codes/codes/ToyotaCorolla.csv', encoding = 'unicode_escape') # by Yan
car_df = pd.read_csv('ToyotaCorolla.csv', encoding = 'unicode_escape') # by Yan
car_df = car_df.iloc[0:1000]
predictors = ['Age_08_04', 'KM', 'Fuel_Type', 'HP', 'Met_Color',
'Automatic', 'CC',
'Doors', 'Quarterly_Tax', 'Weight']
outcome = 'Price'
# partition data
X = pd.get_dummies(car_df[predictors], drop_first=True)
y = car_df[outcome]
train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.4, random_state=1)
car_lm = LinearRegression()
car_lm.fit(train_X, train_y)
# print coefficients
print(pd.DataFrame({'Predictor': X.columns, 'coefficient': car_lm.coef_}))
# print performance measures (training data)
regressionSummary(train_y, car_lm.predict(train_X))

# Use predict() to make predictions on a new set
car_lm_pred = car_lm.predict(valid_X)
result = pd.DataFrame({'Predicted': car_lm_pred, 'Actual': valid_y, 'Residual': valid_y - car_lm_pred})
print(result.head(20))
# print performance measures (validation data)
regressionSummary(valid_y, car_lm_pred)


# Use predict() to make predictions on a new set
car_lm_pred = car_lm.predict(valid_X)
result = pd.DataFrame({'Predicted': car_lm_pred, 'Actual': valid_y, 'Residual': valid_y - car_lm_pred})
print(result.head(20))
# print performance measures (validation data)
regressionSummary(valid_y, car_lm_pred)

car_lm_pred = car_lm.predict(valid_X)
all_residuals = valid_y - car_lm_pred
# Determine the percentage of datapoints with a residual in [-1406, 1406] = approx.
# 75%
print(len(all_residuals[(all_residuals > -1406) & (all_residuals < 1406)]) /len(all_residuals))

# pd.DataFrame('Residuals': all_residuals).hist(bins=25)
pd.DataFrame(all_residuals).hist(bins = 25) # by Yan

plt.show()
