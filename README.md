eBay-case
==============================

Ad click prediction demo case

### Guide to run the scripts

* Create python 3.6 environment and install requirements
  * Anaconda: `conda env create -f environment.yml`
  * Otherwise: `pip install -r requirements.txt`
* `python src/preprocessing.py`
* `python src/train.py`

N.B. I started off with a notebook for exploratory data analysis (notebooks/01-explore-data.ipynb)

## Report

### General remark
This is a demo case and not a real case. In a real case I would handle some things differently (like making sure that the problem definition is clear and that the approach results in a useful solution).
Since time is very critical I need to quickly give insight into my thought process and approach.
I'm used to working in an agile manner, by delivering an MVP as soon as possible and iteratively improving. As such, what I deliver is an MVP with a plan on how to improve from there on.

### About the case and problem definition
The most important question I have about the case is how the model would be used?
Why do you want to predict the click probability?
* Is it for real-time bidding on paid channels?
* Is it for optimizing ad design and/or ad serving parameters in the chanel?
* Is it for prioritizing a set of candidate ads for each user?
The answer to this is important in order to determine which features to (not) use.

Since the case description states "Your objective is to build an ML model predicting whether a certain ad will be clicked or not" (not offering any context) I'll resort to naively trying to predict the features ad hand.

### About modeling

#### Feature use:
* Datetime: I used features such as hour and derivatives (working hour or not, night or not). Since the data is all from a single day no other day/month/year-related features are relevant
* Site features: Only the category is used since site id is too granular
* App features: Only the category is used since app id is too granular
* Device features: Only the categories are used since id is too granular
  N.B. You could derive location features (Country, Region, City) from the device ip address, but that's too time-consuming for now
* Proprietary features: Not used since I have no idea what they are.

Almost all features are categorical and need to be one-hot-encoded.

My approach to modeling is to start off with a RandomForest model:
* Overall good performance for many cases
* GBT/XGBoost probably slightly better but significantly slower
* Not intended as best model ever, but more of an easy yet solid benchmark

I did some hyperparameter tuning. I don't expect big differences in performance, but it's good to get an impression of how much improvement can be made by tuning.

#### Data split

The data was split randomly. It might be worthwhile to stratify on highly imbalanced features such as channel and possibly hour to ensure comparable distributions in all sets.
I split off 200K records for the test set. The rest is used in the grid search using cross-validation, where the data is split into a train set for training and a dev set for validation / model selection.


#### Metrics

I use one metric for optimizing and some metrics for explaining model performance to stakeholders:
* Optimizing metric: Any of balanced_accuracy, average_precision, f1-score, log-loss would be fine for this case (they all handle imbalanced cases well). If the order of scored cases is more important than the actual absolute difference between score and true value (which might be the case here) then ROC AUC or average precision is a good choice.
* Model explanation metric:
  * Accuracy: while not the best metric for unbalanced cases, it is definitely the most easy to understand for the non-initiated.
  * Area under ROC: nice overall metric between 0 and 1 which can be understood as a grade (0.7 is ok, 0.8 is good, 0.9 is great)
  * Confusion matrix: also easily understandable
  * Feature importance: to give some insight into how the model works

### Improvements backlog

* Features & pre-processing
  * Derive location features from ip address
  * Replace pandas get_dummies by a robust way of encoding categorical features, that could be used for scoring the model on later data with possibly unseen values.
* Modeling
  * It would be good to be able to use user data in the model. I would work on creating an embedding (representation) for the user (device). The embedding features can then be used for training the model.
  * Similarly: also create embeddings for site and app
* Predicting / deploying
  * I didn't include a script or predicting on new data. If the model was actually to be used, then the pre-processing script needs to be made suitable for scoring purposes.

#### How to create embeddings

Consider the word2vec concept, where words are defined as the context they appear in, we train Pr(word) = f(context words) (CBOW) or Pr(context words = f(word) (Skip-gram), only to use the intermediate weight matrix as embeddings of the word:

![CBOW & Skip-gram](https://i.stack.imgur.com/O2YeO.png)

Analogously, we can define a user as the set of sites or apps he/she visits.
And vice-versa: an webpage/app id as the set of users that visit it.

An even better approach is to use graph node embedding techniques such as DeepWalk or Node2vec:
The data set can be represented as a graph where sites, apps and devices are nodes and ad servings are edges between those nodes. By performing (semi-)random walks through the graph we assemble data points to train a neural network (similar to skip-gram) to get the embeddings.
These embeddings should give a representation of the website/app/device in question which can effectively be used to train the model.

