import pandas as pd
import matplotlib.pylab as plt
import statsmodels.formula.api as sm
from statsmodels.tsa import tsatools

from matplotlib.backends.backend_pdf import PdfPages
pp = PdfPages('ch16.pdf')
	
Amtrak_df = pd.read_csv('Amtrak.csv')
# convert the date information to a datetime object
Amtrak_df['Date'] = pd.to_datetime(Amtrak_df.Month,
format='%d/%m/%Y')
# convert dataframe column to series (name is used to label the data)
ridership_ts = pd.Series(Amtrak_df.Ridership.values,
index=Amtrak_df.Date,
name='Ridership')
# define the time series frequency
ridership_ts.index = pd.DatetimeIndex(ridership_ts.index,
freq=ridership_ts.index.inferred_freq)
# plot the series
ax = ridership_ts.plot()
ax.set_xlabel('Time')
ax.set_ylabel('Ridership (in 000s)')
ax.set_ylim(1300, 2300)

pp.savefig()


# create short time series from 1997 to 1999 using a slice
ridership_ts_3yrs = ridership_ts['1997':'1999']
# create a data frame with additional predictors from time series
# the following command adds a constant term, a trend term and a quadratic trend term
ridership_df = tsatools.add_trend(ridership_ts, trend='ctt')
# fit a linear regression model to the time series
ridership_lm = sm.ols(formula='Ridership ~ trend + trend_squared',data=ridership_df).fit()
# shorter and longer time series
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10,6))
ridership_ts_3yrs.plot(ax=axes[0])
ridership_ts.plot(ax=axes[1])
for ax in axes:
  ax.set_xlabel('Time')
  ax.set_ylabel('Ridership (in 000s)')
  ax.set_ylim(1300, 2300)  
ridership_lm.predict(ridership_df).plot(ax=axes[1])

pp.savefig()
plt.show()


nValid = 36
nTrain = len(ridership_ts) - nValid
# partition the data
train_ts = ridership_ts[:nTrain]
valid_ts = ridership_ts[nTrain:]
# generate the naive and seasonal naive forecast
naive_pred = pd.Series(train_ts[-1], index=valid_ts.index)
last_season = train_ts[-12:]
seasonal_pred = pd.Series(pd.concat([last_season]*5)
[:len(valid_ts)].values,
index=valid_ts.index)
# plot forecasts and actual in the training and validationsets
ax = train_ts.plot(color='C0', linewidth=0.75, figsize=
(9,7))
valid_ts.plot(ax=ax, color='C0', linestyle='dashed',
linewidth=0.75)
ax.set_xlim('1990', '2006-6')
ax.set_ylim(1300, 2600)
ax.set_xlabel('Time')
ax.set_ylabel('Ridership (in 000s)')
naive_pred.plot(ax=ax, color='green')
seasonal_pred.plot(ax=ax, color='orange')
# determine coordinates for drawing the arrows and lines
one_month = pd.Timedelta('31 days')
xtrain = (min(train_ts.index), max(train_ts.index) -
one_month)
xvalid = (min(valid_ts.index) + one_month,
max(valid_ts.index) - one_month)
xfuture = (max(valid_ts.index) + one_month, '2006')
xtv = xtrain[1] + 0.5 * (xvalid[0] - xtrain[1])
xvf = xvalid[1] + 0.5 * (xfuture[0] - xvalid[1])
ax.add_line(plt.Line2D(xtrain, (2450, 2450), color='black',
linewidth=0.5))
ax.add_line(plt.Line2D(xvalid, (2450, 2450), color='black',
linewidth=0.5))
ax.add_line(plt.Line2D(xfuture, (2450, 2450), color='black',
linewidth=0.5))
ax.text('1995', 2500, 'Training')
ax.text('2001-9', 2500, 'Validation')
ax.text('2004-7', 2500, 'Future')
ax.axvline(x=xtv, ymin=0, ymax=1, color='black',
linewidth=0.5)
ax.axvline(x=xvf, ymin=0, ymax=1, color='black',
linewidth=0.5)

pp.savefig()
plt.show()
pp.close()

#regressionSummary(valid_ts, naive_pred)

# regressionSummary(valid_ts, seasonal_pred)






















