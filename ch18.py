import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import statsmodels.formula.api as sm
from statsmodels.tsa import tsatools
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Load data and convert to time series
Amtrak_df = pd.read_csv('Amtrak.csv')
Amtrak_df['Date'] = pd.to_datetime(Amtrak_df.Month,
format='%d/%m/%Y')
ridership_ts = pd.Series(Amtrak_df.Ridership.values,
index=Amtrak_df.Date,
name='Ridership')
ridership_ts.index = pd.DatetimeIndex(ridership_ts.index,
freq=ridership_ts.index.inferred_freq)
# centered moving average with window size = 12
ma_centered = ridership_ts.rolling(12, center=True).mean()
# trailing moving average with window size = 12
ma_trailing = ridership_ts.rolling(12).mean()
# shift the average by one time unit to get the next day predictions
ma_centered = pd.Series(ma_centered[:-1].values,
index=ma_centered.index[1:])
ma_trailing = pd.Series(ma_trailing[:-1].values,
index=ma_trailing.index[1:])
fig, ax = plt.subplots(figsize=(8, 7))
ax = ridership_ts.plot(ax=ax, color='black', linewidth=0.25)
ma_centered.plot(ax=ax, linewidth=2)
ma_trailing.plot(ax=ax, style='--', linewidth=2)
ax.set_xlabel('Time')
ax.set_ylabel('Ridership')
ax.legend(['Ridership', 'Centered Moving Average', 'Trailing Moving Average'])
plt.show()

# partition the data
nValid = 36
nTrain = len(ridership_ts) - nValid
train_ts = ridership_ts[:nTrain]
valid_ts = ridership_ts[nTrain:]
# moving average on training
ma_trailing = train_ts.rolling(12).mean()
last_ma = ma_trailing[-1]
# create forecast based on last moving average in the training period
ma_trailing_pred = pd.Series(last_ma, index=valid_ts.index)
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(9, 7.5))
ma_trailing.plot(ax=axes[0], linewidth=2, color='C1')
ma_trailing_pred.plot(ax=axes[0], linewidth=2, color='C1',
linestyle='dashed')
residual = train_ts - ma_trailing
residual.plot(ax=axes[1], color='C1')
residual = valid_ts - ma_trailing_pred
residual.plot(ax=axes[1], color='C1', linestyle='dashed')
# graphLayout(axes, train_ts, valid_ts)

# Build a model with seasonality, trend, and quadratic trend
ridership_df = tsatools.add_trend(ridership_ts, trend='ct')
ridership_df['Month'] = ridership_df.index.month
# partition the data
train_df = ridership_df[:nTrain]
valid_df = ridership_df[nTrain:]
formula ='Ridership~trend + np.square(trend) + C(Month)'
ridership_lm_trendseason = sm.ols(formula=formula, data=train_df).fit()
# create single-point forecast
ridership_prediction = ridership_lm_trendseason.predict(valid_df.iloc[0, :])
# apply MA to residuals
ma_trailing = ridership_lm_trendseason.resid.rolling(12).mean()
print('Prediction', ridership_prediction[0])
print('ma_trailing', ma_trailing[-1])


residuals_ts = ridership_lm_trendseason.resid
residuals_pred = valid_df.Ridership -ridership_lm_trendseason.predict(valid_df)
fig, ax = plt.subplots(figsize=(9,4))
ridership_lm_trendseason.resid.plot(ax=ax, color='black', linewidth=0.5)
residuals_pred.plot(ax=ax, color='black', linewidth=0.5)
ax.set_ylabel('Ridership')
ax.set_xlabel('Time')
ax.axhline(y=0, xmin=0, xmax=1, color='grey', linewidth=0.5)
# run exponential smoothing
# with smoothing level alpha = 0.2
expSmooth = ExponentialSmoothing(residuals_ts, freq='MS')
expSmoothFit = expSmooth.fit(smoothing_level=0.2)
expSmoothFit.fittedvalues.plot(ax=ax)
expSmoothFit.forecast(len(valid_ts)).plot(ax=ax, style='--',
linewidth=2, color='C0')
# singleGraphLayout(ax, [-550, 550], train_df, valid_df)
# run exponential smoothing with additive trend and additive
expSmooth = ExponentialSmoothing(train_ts, trend='additive',
seasonal='additive',
seasonal_periods=12, freq='MS')
expSmoothFit = expSmooth.fit()
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(9, 7.5))
expSmoothFit.fittedvalues.plot(ax=axes[0], linewidth=2,
color='C1')
expSmoothFit.forecast(len(valid_ts)).plot(ax=axes[0],
linewidth=2, color='C1',
linestyle='dashed')
residual = train_ts - expSmoothFit.fittedvalues
residual.plot(ax=axes[1], color='C1')
residual = valid_ts - expSmoothFit.forecast(len(valid_ts))
residual.plot(ax=axes[1], color='C1', linestyle='dashed')
# graphLayout(axes, train_ts, valid_ts)

print(expSmoothFit.params)
print('AIC: ', expSmoothFit.aic)
print('AICc: ', expSmoothFit.aicc)
