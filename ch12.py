import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pylab as plt
from dmba import classificationSummary

mower_df = pd.read_csv('RidingMowers.csv')
da_reg = LinearDiscriminantAnalysis()
da_reg.fit(mower_df.drop(columns=['Ownership']),
mower_df['Ownership'])
print('Coefficients', da_reg.coef_)
print('Intercept', da_reg.intercept_)

da_reg = LinearDiscriminantAnalysis()
da_reg.fit(mower_df.drop(columns=['Ownership']), mower_df['Ownership'])
result_df = mower_df.copy()
result_df['Dec. Function'] = da_reg.decision_function(mower_df.drop(columns=
['Ownership']))
result_df['Prediction'] = da_reg.predict(mower_df.drop(columns=['Ownership']))
result_df['p(Owner)'] = da_reg.predict_proba(mower_df.drop(columns=['Ownership']))[:, 1]

accidents_df = pd.read_csv('accidents.csv')
lda_reg = LinearDiscriminantAnalysis()
lda_reg.fit(accidents_df.drop(columns=['MAX_SEV']),
accidents_df['MAX_SEV'])
print('Coefficients and intercept')
fct = pd.DataFrame([lda_reg.intercept_],
columns=lda_reg.classes_, index=['constant'])
fct = fct.append(pd.DataFrame(lda_reg.coef_.transpose(),
columns=lda_reg.classes_,
index=list(accidents_df.columns)[:-1]))
print(fct)
print()
classificationSummary(accidents_df['MAX_SEV'],
lda_reg.predict(accidents_df.drop(columns=['MAX_SEV'])),
class_names=lda_reg.classes_)


result = pd.concat([
pd.DataFrame({'Classification':
lda_reg.predict(accidents_df.drop(columns=['MAX_SEV'])),
'Actual': accidents_df['MAX_SEV']}),
pd.DataFrame(lda_reg.decision_function(accidents_df.drop(columns=
['MAX_SEV'])),
columns=['Score {}'.format(cls) for cls in
lda_reg.classes_]),
pd.DataFrame(lda_reg.predict_proba(accidents_df.drop(columns=
['MAX_SEV'])),
columns=['Propensity {}'.format(cls) for cls in
lda_reg.classes_])
], axis=1)
pd.set_option('precision',2)
pd.set_option('chop_threshold', .01)
print(result.head())







