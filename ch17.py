import math
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import statsmodels.formula.api as sm
from statsmodels.tsa import tsatools, stattools
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics import tsaplots

# code for creating Figure 17.1
# load data and convert to time series
Amtrak_df = pd.read_csv('Amtrak.csv')
# Amtrak_df.isnull().sum()
# Amtrak_df.dtypes
# Amtrak_df['Month'] = pd.to_datetime(Amtrak_df['Month'])
# Amtrak_df = Amtrak_df.set_index('Month')
# Amtrak_df.plot(grid = True)

Amtrak_df['Date'] = pd.to_datetime(Amtrak_df.Month) # , format='
# ridership_ts = pd.Series(Amtrak_df.Ridership.values,index=Amtrak_df.Date)
# ridership_ts = Amtrak_df[['Ridership', 'Date']]
ridership_ts = Amtrak_df
# ridership_ts.columns = ['Ridership', 'Month']
ridership_ts = ridership_ts.set_index('Date')

# fit a linear trend model to the time series
ridership_df = tsatools.add_trend(ridership_ts, trend='ct')
ridership_lm = sm.ols(formula='Ridership~trend', data=ridership_df).fit()
# shorter and longer time series
ax = ridership_ts.plot()
ax.set_xlabel('Time')
ax.set_ylabel('Ridership (in 000s)')
ax.set_ylim(1300, 2300)
ridership_lm.predict(ridership_df).plot(ax=ax)
# plt.show()
plt.savefig('fig17.1.png')

# code for creating Figure 17.2
# fit linear model using training set and predict on validation set

# ridership_df = tsatools.add_trend(ridership_ts, trend='c')
# ridership_df['Month'] = ridership_df.index.month
# partition the data
nValid = 36
nTrain = len(ridership_df) - nValid
train_df = ridership_df[:nTrain]
valid_df = ridership_df[nTrain:]
ridership_lm = sm.ols(formula='Ridership ~ trend', data=train_df).fit()
predict_df = ridership_lm.predict(valid_df)
# create the plot
def singleGraphLayout(ax, ylim, train_df, valid_df):
    ax.set_xlim('1990', '2004-6')
    ax.set_ylim(*ylim)
    ax.set_xlabel('Time')
    one_month = pd.Timedelta('31 days')
    xtrain = (min(train_df.index), max(train_df.index) -
    one_month)
    xvalid = (min(valid_df.index) + one_month,
    max(valid_df.index) - one_month)
    xtv = xtrain[1] + 0.5 * (xvalid[0] - xtrain[1])
    ypos = 0.9 * ylim[1] + 0.1 * ylim[0]
    ax.add_line(plt.Line2D(xtrain, (ypos, ypos), color='black',
    linewidth=0.5))
    ax.add_line(plt.Line2D(xvalid, (ypos, ypos), color='black',
    linewidth=0.5))
    ax.axvline(x=xtv, ymin=0, ymax=1, color='black',
    linewidth=0.5)
    ypos = 0.925 * ylim[1] + 0.075 * ylim[0]
    ax.text('1995', ypos, 'Training')
    ax.text('2002-3', ypos, 'Validation')
def graphLayout(axes, train_df, valid_df):
    singleGraphLayout(axes[0], [1300, 2550], train_df, valid_df)
    singleGraphLayout(axes[1], [-550, 550], train_df, valid_df)
    train_df.plot(y='Ridership', ax=axes[0], color='C0',
    linewidth=0.75)
    valid_df.plot(y='Ridership', ax=axes[0], color='C0',
    linestyle='dashed',
    linewidth=0.75)
    axes[1].axhline(y=0, xmin=0, xmax=1, color='black',
    linewidth=0.5)
    axes[0].set_xlabel('')
    axes[0].set_ylabel('Ridership (in 000s)')
    axes[1].set_ylabel('Forecast Errors')
    if axes[0].get_legend():
       axes[0].get_legend().remove()

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(9, 7.5))

# generate the naive and seasonal naive

ridership_lm.predict(train_df).plot(ax=axes[0], color='C1')
ridership_lm.predict(valid_df).plot(ax=axes[0], color='C1', linestyle='dashed')

residual = train_df.Ridership - ridership_lm.predict(train_df)
residual.plot(ax=axes[1], color='C1')
residual = valid_df.Ridership - ridership_lm.predict(valid_df)
residual.plot(ax=axes[1], color='C1', linestyle='dashed')
graphLayout(axes, train_df, valid_df)
plt.tight_layout()
plt.show()

# code for creating Figure 17.3
ridership_lm_linear = sm.ols(formula='Ridership ~ trend',data=train_df).fit()
predict_df_linear = ridership_lm_linear.predict(valid_df)
ridership_lm_expo = sm.ols(formula='np.log(Ridership) ~ trend',data=train_df).fit()
predict_df_expo = ridership_lm_expo.predict(valid_df)
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9,3.75))
train_df.plot(y='Ridership', ax=ax, color='C0', linewidth=0.75)
valid_df.plot(y='Ridership', ax=ax, color='C0',linestyle='dashed', linewidth=0.75)
singleGraphLayout(ax, [1300, 2600], train_df, valid_df)
ridership_lm_linear.predict(train_df).plot(color='C1')
ridership_lm_linear.predict(valid_df).plot(color='C1',linestyle='dashed')
ridership_lm_expo.predict(train_df).apply(lambda row:math.exp(row)).plot(color='C2')
ridership_lm_expo.predict(valid_df).apply(lambda row:math.exp(row)).plot(color='C2',linestyle='dashed')
ax.get_legend().remove()
plt.show()

# code for creating Figure 17.4
ridership_lm_poly = sm.ols(formula='Ridership ~ trend + np.square(trend)', data= train_df).fit()
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(9, 7.5))
ridership_lm_poly.predict(train_df).plot(ax=axes[0], color='C1')
ridership_lm_poly.predict(valid_df).plot(ax=axes[0], color='C1', linestyle='dashed')
residual = train_df.Ridership - ridership_lm_poly.predict(train_df)
residual.plot(ax=axes[1], color='C1')
residual = valid_df.Ridership - ridership_lm_poly.predict(valid_df)
residual.plot(ax=axes[1], color='C1', linestyle='dashed')
graphLayout(axes, train_df, valid_df)
plt.show()

# Table 17.5 Summary of output from fitting additive
# ridership_df = tsatools.add_trend(ridership_ts, trend='c')
# ridership_df['Month'] = ridership_df.index.month
# partition the data
# train_df = ridership_df[:nTrain]
# valid_df = ridership_df[nTrain:]
ridership_lm_season = sm.ols(formula='Ridership~C(Month)', data=train_df).fit()
ridership_lm_season.summary()

# Table 17.6 Summary of output from fitting trend and
formula = 'Ridership ~ trend + np.square(trend) + C(Month)'
ridership_lm_trendseason = sm.ols(formula=formula, data=train_df).fit()

# code for creating Figure 17.7
tsaplots.plot_acf(train_df['1991-01-01':'1993-01-01'].Ridership)
plt.show()


# code for running AR(1) model on residuals
formula = 'Ridership ~ trend + np.square(trend) + C(Month)'
train_lm_trendseason = sm.ols(formula=formula, data=train_df).fit()
train_res_arima = ARIMA(train_lm_trendseason.resid, order=(1, 0, 0), freq='MS').fit(trend='nc')
# train_res_arima = ARIMA(train_lm_trendseason.resid, order=(1, 0, 0)).fit(trend='nc')
forecast, _, conf_int = train_res_arima.forecast(1)

# code for creating Figure 17.9
ax = train_lm_trendseason.resid.plot(figsize=(9,4))
train_res_arima.fittedvalues.plot(ax=ax)
singleGraphLayout(ax, [-250, 250], train_df, valid_df)
plt.show()

# Output for AR(1) Model on S&P500 Monthly Closing Prices
sp500_df = pd.read_csv('SP500.csv')
# convert date to first of each month
sp500_df['Date'] = pd.to_datetime(sp500_df.Date, format='%d-%b-%y').dt.to_period('M')
sp500_ts = pd.Series(sp500_df.Close.values, index=sp500_df.Date,name='sp500')
sp500_arima = ARIMA(sp500_ts, order=(1, 0, 0)).fit(disp=0)
print(pd.DataFrame({'coef': sp500_arima.params, 'std err':sp500_arima.bse}))