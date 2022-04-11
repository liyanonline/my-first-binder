%matplotlib inline
# replace with:
# pd.read_csv('
# pd.read_csv('

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from mord import LogisticIT
import matplotlib.pylab as plt
import seaborn as sns
from dmba import classificationSummary, gainsChart, liftChart
from dmba.metric import AIC_score


bank_df = pd.read_csv('/opt/notebooks/Codes/codes/UniversalBank.csv')
bank_df.drop(columns=['ID', 'ZIP Code'], inplace=True)
bank_df.columns = [c.replace(' ', '_') for c in
bank_df.columns]
# Treat education as categorical, convert to dummy variables
bank_df['Education'] = bank_df['Education'].astype('category')
new_categories = {1: 'Undergrad', 2: 'Graduate', 3:
'Advanced/Professional'}
bank_df.Education.cat.rename_categories(new_categories,
inplace=True)
bank_df = pd.get_dummies(bank_df, prefix_sep='_',
drop_first=True)
y = bank_df['Personal_Loan']
X = bank_df.drop(columns=['Personal_Loan'])
# partition data
train_X, valid_X, train_y, valid_y = train_test_split(X, y,
test_size=0.4, random_state=1)
# fit a logistic regression (set penalty=l2 and C=1e42 to avoid regularization)
logit_reg = LogisticRegression(penalty="l2", C=1e42,
solver='liblinear')
logit_reg.fit(train_X, train_y)
print('intercept ', logit_reg.intercept_[0])
print(pd.DataFrame({'coeff': logit_reg.coef_[0]},index=X.columns).transpose())
print('AIC', AIC_score(valid_y, logit_reg.predict(valid_X),
df = len(train_X.columns) + 1))

logit_reg_pred = logit_reg.predict(valid_X)
logit_reg_proba = logit_reg.predict_proba(valid_X)
logit_result = pd.DataFrame({'actual': valid_y,
'p(0)': [p[0] for p in
logit_reg_proba],
'p(1)': [p[1] for p in
logit_reg_proba],
'predicted': logit_reg_pred })
# display four different cases
interestingCases = [2764, 932, 2721, 702]
print(logit_result.loc[interestingCases])


# training confusion matrix
classificationSummary(train_y, logit_reg.predict(train_X))
# validation confusion matrix
classificationSummary(valid_y, logit_reg.predict(valid_X))

df = logit_result.sort_values(by=['p(1)'], ascending=False)
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
gainsChart(df.actual, ax=axes[0])
liftChart(df['p(1)'], title=False, ax=axes[1])
plt.show()

data = pd.read_csv('/opt/notebooks/Codes/codes/accidentsFull.csv')
outcome = 'MAX_SEV_IR'
predictors = ['ALCHL_I', 'WEATHER_R']
y = data[outcome]
X = data[predictors]
train_X, train_y = X, y
print('Nominal logistic regression')
logit = LogisticRegression(penalty="l2", solver='lbfgs',
C=1e24, multi_class='multinomial')
logit.fit(X, y)
print(' intercept', logit.intercept_)
print(' coefficients', logit.coef_)
print()
probs = logit.predict_proba(X)
results = pd.DataFrame({
'actual': y, 'predicted': logit.predict(X),
'P(0)': [p[0] for p in probs],
'P(1)': [p[1] for p in probs],
'P(2)': [p[2] for p in probs],
})
print(results.head())
print()
print('Ordinal logistic regression')
logit = LogisticIT(alpha=0)
logit.fit(X, y)
print(' theta', logit.theta_)
print(' coefficients', logit.coef_)
print()
probs = logit.predict_proba(X)
results = pd.DataFrame({
'actual': y, 'predicted': logit.predict(X),
'P(0)': [p[0] for p in probs],
'P(1)': [p[1] for p in probs],
'P(2)': [p[2] for p in probs],
})
print(results.head())

delays_df = pd.read_csv('/opt/notebooks/Codes/codes/FlightDelays.csv')
# Create an indicator variable
delays_df['isDelayed'] = [1 if status == 'delayed' else 0
for status in delays_df['Flight Status']]
def createGraph(group, xlabel, axis):
  groupAverage = delays_df.groupby([group])['isDelayed'].mean()
  if group == 'DAY_WEEK': # rotate so that display starts on Sunday
    groupAverage = groupAverage.reindex(index=np.roll(groupAverage.index,1))
    groupAverage.index = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
  ax = groupAverage.plot.bar(color='C0', ax=axis)
  ax.set_ylabel('Average Delay')
  ax.set_xlabel(xlabel)
  return ax
def graphDepartureTime(xlabel, axis):
  temp_df = pd.DataFrame({'CRS_DEP_TIME':
  delays_df['CRS_DEP_TIME'] // 100, 'isDelayed': delays_df['isDelayed']})
  groupAverage = temp_df.groupby(['CRS_DEP_TIME'])['isDelayed'].mean()
  ax = groupAverage.plot.bar(color='C0', ax=axis)
  ax.set_xlabel(xlabel); ax.set_ylabel('Average Delay')
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 10))
createGraph('DAY_WEEK', 'Day of week', axis=axes[0][0])
createGraph('DEST', 'Destination', axis=axes[0][1])
graphDepartureTime('Departure time', axis=axes[1][0])
createGraph('CARRIER', 'Carrier', axis=axes[1][1])
createGraph('ORIGIN', 'Origin', axis=axes[2][0])
createGraph('Weather', 'Weather', axis=axes[2][1])
plt.tight_layout()

agg = delays_df.groupby(['ORIGIN', 'DAY_WEEK','CARRIER']).isDelayed.mean()
agg = delays_df.groupby(['ORIGIN', 'DAY_WEEK','CARRIER']).isDelayed.mean()
agg = agg.reset_index()
# Define the layout of the graph
height_ratios = []
for i, origin in enumerate(sorted(delays_df.ORIGIN.unique())):
  height_ratios.append(len(agg[agg.ORIGIN == origin].CARRIER.unique()))
  gridspec_kw = {'height_ratios': height_ratios, 'width_ratios':[15, 1]}
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 6),gridspec_kw = gridspec_kw)
axes[0, 1].axis('off')
axes[2, 1].axis('off')
maxIsDelay = agg.isDelayed.max()
for i, origin in enumerate(sorted(delays_df.ORIGIN.unique())):
  data = pd.pivot_table(agg[agg.ORIGIN == origin],values='isDelayed', aggfunc=np.sum,index=['CARRIER'], columns= ['DAY_WEEK'])
  data = data[[7, 1, 2, 3, 4, 5, 6]] # Shift last columns to first
  ax = sns.heatmap(data, ax=axes[i][0], vmin=0,
vmax=maxIsDelay,
cbar_ax=axes[1][1],
cmap=sns.light_palette("navy"))
ax.set_xticklabels(['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'])
if i != 2:
  ax.get_xaxis().set_visible(False)
  ax.set_ylabel('Airport ' + origin)
plt.show()

delays_df = pd.read_csv('/opt/notebooks/Codes/codes/FlightDelays.csv')
# Create an indicator variable
delays_df['isDelayed'] = [1 if status == 'delayed' else 0
for status in delays_df['Flight Status']]
# convert to categorical
delays_df.DAY_WEEK = delays_df.DAY_WEEK.astype('category')
# create hourly bins departure time
delays_df.CRS_DEP_TIME = [round(t / 100) for t in
delays_df.CRS_DEP_TIME]
delays_df.CRS_DEP_TIME = delays_df.CRS_DEP_TIME.astype('category')
predictors = ['DAY_WEEK', 'CRS_DEP_TIME', 'ORIGIN', 'DEST',
'CARRIER', 'Weather']
outcome = 'isDelayed'
X = pd.get_dummies(delays_df[predictors], drop_first=True)
y = delays_df[outcome]
classes = ['ontime', 'delayed']
# split into training and validation
train_X, valid_X, train_y, valid_y = train_test_split(X, y,
test_size=0.4, random_state=1)
logit_full = LogisticRegression(penalty="l2", C=1e42,
solver='liblinear')
logit_full.fit(train_X, train_y)
print('intercept ', logit_full.intercept_[0])
print(pd.DataFrame({'coeff': logit_full.coef_[0]},
index=X.columns).transpose())
print('AIC', AIC_score(valid_y, logit_full.predict(valid_X),
df = len(train_X.columns) + 1))


logit_reg_pred = logit_full.predict_proba(valid_X)
full_result = pd.DataFrame({'actual': valid_y,
'p(0)': [p[0] for p in
logit_reg_pred],
'p(1)': [p[1] for p in
logit_reg_pred],
'predicted':
logit_full.predict(valid_X)})
full_result = full_result.sort_values(by=['p(1)'],
ascending=False)
# confusion matrix
classificationSummary(full_result.actual, full_result.predicted,
class_names=classes)
gainsChart(full_result.actual, figsize=[5, 5])
plt.show()


delays_df = pd.read_csv('/opt/notebooks/Codes/codes/FlightDelays.csv')
delays_df['isDelayed'] = [1 if status == 'delayed' else 0
for status in delays_df['Flight Status']]
delays_df['CRS_DEP_TIME'] = [round(t / 100) for t in
delays_df['CRS_DEP_TIME']]
delays_red_df = pd.DataFrame({
'Sun_Mon' : [1 if d in (1, 7) else 0 for d in
delays_df.DAY_WEEK],
'Weather' : delays_df.Weather,
'CARRIER_CO_MQ_DH_RU' : [1 if d in ("CO", "MQ", "DH",
"RU") else 0
for d in delays_df.CARRIER],
'MORNING' : [1 if d in (6, 7, 8, 9) else 0 for d in
delays_df.CRS_DEP_TIME],
'NOON' : [1 if d in (10, 11, 12, 13) else 0 for d in
delays_df.CRS_DEP_TIME],
'AFTER2P' : [1 if d in (14, 15, 16, 17, 18) else 0 for d
in delays_df.CRS_DEP_TIME],
'EVENING' : [1 if d in (19, 20) else 0 for d in
delays_df.CRS_DEP_TIME],
'isDelayed' : [1 if status == 'delayed' else 0 for
status in delays_df['Flight Status']],
})
X = delays_red_df.drop(columns=['isDelayed'])
y = delays_red_df['isDelayed']
classes = ['ontime', 'delayed']
# split into training and validation
train_X, valid_X, train_y, valid_y = train_test_split(X, y,
test_size=0.4,
random_state=1)
logit_red = LogisticRegressionCV(penalty="l1", solver='liblinear', cv=5)
logit_red.fit(train_X, train_y)
print('intercept ', logit_red.intercept_[0])
print(pd.DataFrame({'coeff': logit_red.coef_[0]},
index=X.columns).transpose())
print('AIC', AIC_score(valid_y, logit_red.predict(valid_X),
df=len(train_X.columns) + 1))
# confusion matrix
classificationSummary(valid_y, logit_red.predict(valid_X),
class_names=classes)


''' 
from Table 10.11 Logistic regression model for loan acceptance using Statmodels
# same initial preprocessing and creating dummies
# add constant column
bank_df = sm.add_constant(bank_df, prepend=True)
y = bank_df['Personal_Loan']
X = bank_df.drop(columns=['Personal_Loan'])
# partition data
train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.4,
random_state=1)
# use GLM (general linear model) with the binomial family to fit a logistic logit_reg = sm.GLM(train_y, train_X, family=sm.families.Binomial())
logit_result = logit_reg.fit()
logit_result.summary()
'''

