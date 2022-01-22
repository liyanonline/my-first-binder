import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from dmba import classificationSummary
bank_df = pd.read_csv('UniversalBank.csv')
bank_df.drop(columns=['ID', 'ZIP Code'], inplace=True)
# split into training and validation
X = bank_df.drop(columns=['Personal Loan'])
y = bank_df['Personal Loan']
X_train, X_valid, y_train, y_valid = train_test_split(X, y,
test_size=0.40,
random_state=3)
# single tree
defaultTree = DecisionTreeClassifier(random_state=1)
defaultTree.fit(X_train, y_train)
classes = defaultTree.classes_
classificationSummary(y_valid, defaultTree.predict(X_valid),
class_names=classes)
# bagging
bagging = BaggingClassifier(DecisionTreeClassifier(random_state=1),
n_estimators=100,
random_state=1)
bagging.fit(X_train, y_train)
classificationSummary(y_valid, bagging.predict(X_valid),
class_names=classes)
# boosting
boost = AdaBoostClassifier(DecisionTreeClassifier(random_state=1),
n_estimators=100,
random_state=1)
boost.fit(X_train, y_train)
classificationSummary(y_valid, boost.predict(X_valid),
class_names=classes)

voter_df = pd.read_csv('Voter-Persuasion.csv')
# Preprocess data frame
predictors = ['AGE', 'NH_WHITE', 'COMM_PT', 'H_F1',
'REG_DAYS',
'PR_PELIG', 'E_PELIG', 'POLITICALC',
'MESSAGE_A']
outcome = 'MOVED_AD'
classes = list(voter_df.MOVED_AD.unique())
# Partition the data
X = voter_df[predictors]
y = voter_df[outcome]
X_train, X_valid, y_train, y_valid = train_test_split(X, y,
test_size=0.40,
random_state=1)
# Train a random forest classifier using the training set
rfModel = RandomForestClassifier(n_estimators=100,
random_state=1)
rfModel.fit(X_train, y_train)
# Calculating the uplift
uplift_df = X_valid.copy() # Need to create a copy to allow modifying data
uplift_df.MESSAGE_A = 1
predTreatment = rfModel.predict_proba(uplift_df)
uplift_df.MESSAGE_A = 0
predControl = rfModel.predict_proba(uplift_df)
upliftResult_df = pd.DataFrame({
'probMessage': predTreatment[:,1],
'probNoMessage': predControl[:,1],
'uplift': predTreatment[:,1] - predControl[:,1],
}, index=uplift_df.index)
upliftResult_df.head()



