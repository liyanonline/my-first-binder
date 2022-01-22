# %matplotlib inline

import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
pp = PdfPages('ch8.pdf')

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
# UserWarning: Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure. plt.show()

from dmba import classificationSummary, gainsChart
# delays_df = pd.read_csv('FlightDelays.csv')
# delays_df = pd.read_csv('/opt/notebooks/Codes/codes/FlightDelays.csv')
delays_df = pd.read_csv('FlightDelays.csv')

# delays_df = pd.read_html('https://cl1p.net/uqzdiybhxgeoi')
# convert to categorical
delays_df.DAY_WEEK = delays_df.DAY_WEEK.astype('category')
delays_df['Flight Status'] = delays_df['Flight Status'].astype('category')
# create hourly bins departure time
delays_df.CRS_DEP_TIME = [round(t / 100) for t in delays_df.CRS_DEP_TIME]
# delays_df.CRS_DEP_TIME = # by Yan
delays_df.CRS_DEP_TIME.astype('category')
predictors = ['DAY_WEEK', 'CRS_DEP_TIME', 'ORIGIN', 'DEST','CARRIER']
outcome = 'Flight Status'
X = pd.get_dummies(delays_df[predictors])
y = delays_df['Flight Status'].astype('category')
classes = list(y.cat.categories)


# split into training and validation
X_train, X_valid, y_train, y_valid = train_test_split(X, y,test_size=0.40,random_state=1)
# run naive Bayes
delays_nb = MultinomialNB(alpha=0.01)
delays_nb.fit(X_train, y_train)
# predict probabilities
predProb_train = delays_nb.predict_proba(X_train)
predProb_valid = delays_nb.predict_proba(X_valid)
# predict class membership
y_train_pred = delays_nb.predict(X_train)
y_valid_pred = delays_nb.predict(X_valid)


# split the original data frame into a train and test using the same random_state
train_df, valid_df = train_test_split(delays_df, test_size=0.4, random_state=1)
pd.set_option('precision', 4)
# probability of flight status
print(train_df['Flight Status'].value_counts() / len(train_df))
print()

for predictor in predictors:
   # construct the frequency table
   df = train_df[['Flight Status', predictor]]
   freqTable = df.pivot_table(index='Flight Status',
   columns=predictor, aggfunc=len)
   # divide each value by the sum of the row to get conditional probabilities
   propTable = freqTable.apply(lambda x: x / sum(x), axis=1)
   print(propTable)
   print()
pd.reset_option('precision')


# classify a specific flight by searching in the dataset
# for a flight with the same predictor values
df = pd.concat([pd.DataFrame({'actual': y_valid, 'predicted':y_valid_pred}), pd.DataFrame(predProb_valid, index=y_valid.index)], axis=1)
mask = ((X_valid.CARRIER_DL == 1) & (X_valid.DAY_WEEK_7 == 1) &  (X_valid.CRS_DEP_TIME == 1) & (X_valid.DEST_LGA == 1) & (X_valid.ORIGIN_DCA == 1)) # by Yan: CRS_DEP_TIME_10
df[mask]


# training
classificationSummary(y_train, y_train_pred, class_names=classes)
# validation
classificationSummary(y_valid, y_valid_pred, class_names=classes)

df = pd.DataFrame({'actual':1 - y_valid.cat.codes, 'prob':predProb_valid[:, 0]})
df = df.sort_values(by=['prob'], ascending=False).reset_index(drop=True)
fig, ax = plt.subplots()
fig.set_size_inches(4, 4)
gainsChart(df.actual, ax=ax)
pp.savefig()
plt.show()


