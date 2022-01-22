import heapq
from collections import defaultdict
import pandas as pd
import matplotlib.pylab as plt
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split

# Load and preprocess data set
fp_df = pd.read_csv('Faceplate.csv')
fp_df.set_index('Transaction', inplace=True)
print(fp_df)
# create frequent itemsets
itemsets = apriori(fp_df, min_support=0.2,
use_colnames=True)
# convert into rules
rules = association_rules(itemsets, metric='confidence',
min_threshold=0.5)
rules.sort_values(by=['lift'], ascending=False).head(6)
print(rules.sort_values(by=['lift'], ascending=False))

# load dataset
all_books_df = pd.read_csv('CharlesBookClub.csv')
# create the binary incidence matrix
ignore = ['Seq#', 'ID#', 'Gender', 'M', 'R', 'F',
'FirstPurch', 'Related Purchase',
'Mcode', 'Rcode', 'Fcode', 'Yes_Florence',
'No_Florence']
count_books = all_books_df.drop(columns=ignore)
count_books[count_books > 0] = 1
# create frequent itemsets and rules
itemsets = apriori(count_books, min_support=200/4000,
use_colnames=True)
rules = association_rules(itemsets, metric='confidence',
min_threshold=0.5)
# Display 25 rules with highest lift
rules.sort_values(by=['lift'], ascending=False).head(25)




import random
random.seed(0)
nratings = 5000
randomData = pd.DataFrame({
'itemID': [random.randint(0,99) for _ in
range(nratings)],
'userID': [random.randint(0,999) for _ in
range(nratings)],
'rating': [random.randint(1,5) for _ in
range(nratings)],
})
def get_top_n(predictions, n=10):
  # First map the predictions to each user.
  byUser = defaultdict(list)
  for p in predictions:
    byUser[p.uid].append(p)
    # For each user, reduce predictions to top-n
    for uid, userPredictions in byUser.items():
      byUser[uid] = heapq.nlargest(n, userPredictions,key=lambda p: p.est)
      return byUser


# Convert the data set into the format required by the surprise package
# The columns must correspond to user id, item id, and ratings (in that order)
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(randomData[['userID', 'itemID',
'rating']], reader)
# Split into training and test set
trainset, testset = train_test_split(data, test_size=.25,
random_state=1)
## User-based filtering
# compute cosine similarity between users
sim_options = {'name': 'cosine', 'user_based': True}
algo = KNNBasic(sim_options=sim_options)
algo.fit(trainset)
# predict ratings for all pairs (u, i) that are NOT in the training set.
predictions = algo.test(testset)
# Print the recommended items for each user
top_n = get_top_n(predictions, n=4)
print('Top-3 recommended items for each user')
for uid, user_ratings in list(top_n.items())[:5]:
  print('User {}'.format(uid))
  # for prediction in user_ratings:
    # print(' Item {0.iid}l({0.est:.2f})'.format(predictions, end=''))


trainset = data.build_full_trainset()
sim_options = {'name': 'cosine', 'user_based': False}
algo = KNNBasic(sim_options=sim_options)
algo.fit(trainset)
# Predict rating for user 383 and item 7
algo.predict(383, 7)


