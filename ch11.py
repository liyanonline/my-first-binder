import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from dmba import classificationSummary

example_df = pd.read_csv('TinyData.csv')
predictors = ['Fat', 'Salt']
outcome = 'Acceptance'
X = example_df[predictors]
y = example_df[outcome]
classes = sorted(y.unique())
clf = MLPClassifier(hidden_layer_sizes=(3),
activation='logistic', solver='lbfgs',
random_state=1)
clf.fit(X, y)
clf.predict(X)
# Network structure
print('Intercepts')
print(clf.intercepts_)
print('Weights')
print(clf.coefs_)
# Prediction
print(pd.concat([
example_df,
pd.DataFrame(clf.predict_proba(X), columns=classes)
], axis=1))

classificationSummary(y, clf.predict(X),
class_names=classes)

####################################################################
accidents_df = pd.read_csv('accidentsnn.csv')
input_vars = ['ALCHL_I', 'PROFIL_I_R', 'VEH_INVL']
accidents_df.SUR_COND = accidents_df.SUR_COND.astype('category')
accidents_df.MAX_SEV_IR = accidents_df.MAX_SEV_IR.astype('category')
# convert the categorical data into dummy variables
# exclude the column for SUR_COND 9 = unknown
processed = pd.get_dummies(accidents_df, columns= ['SUR_COND'])
processed = processed.drop(columns=['SUR_COND_9'])
outcome = 'MAX_SEV_IR'
predictors = [c for c in processed.columns if c != outcome]
# partition data
X = processed[predictors]
y = processed[outcome]
train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.4, random_state=1)
# train neural network with 2 hidden nodes
clf = MLPClassifier(hidden_layer_sizes=(2), activation='logistic', solver='lbfgs', random_state=1)
clf.fit(train_X, train_y.values)
# training performance (use idxmax to revert the one-hotencoding)
classificationSummary(train_y, clf.predict(train_X))
# validation performance
classificationSummary(valid_y, clf.predict(valid_X))


