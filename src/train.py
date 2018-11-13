import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, confusion_matrix
import logging


logging.basicConfig(level=logging.INFO)


INPUT_FILENAME = './data/interim/data_preprocessed.csv'
RANDOM_SEED = 1234


# read preprocessed data
logging.info("Start reading data")
dat = pd.read_csv(INPUT_FILENAME)
logging.info("End reading data")

# for now: take a sample just to speed things up a bit...
dat = dat.sample(1000000, random_state=RANDOM_SEED)

feature_names = dat.columns.values[1:]

# define X and Y (features and labels)
logging.info("Start define X/y")
y = dat['is_clicked'].values
X = dat.drop(['is_clicked'], axis=1).values
logging.info("End define X/y")

# split data into:
# dev set: 100K
# test set: 100K
# training set: the rest (approx 2.8M)
logging.info("Start train/test split")
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=200000)
X_dev, X_test, y_dev, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5)
logging.info("End train/test split")

# cleanup
del dat, X, y, X_tmp, y_tmp


# rf = RandomForestClassifier(class_weight='balanced', random_state=RANDOM_SEED)
# grid_search = GridSearchCV(estimator=rf,
#                            param_grid=param_grid,
#                            cv=3,
#                            scoring='average_precision',
#                            refit=True,
#                            n_jobs=-1,
#                            verbose=2
#                            )
# grid_search.fit(X_train, y_train)
# logging.info("End training models")
#
# logging.info(grid_search.cv_results_)
# logging.info(grid_search.best_params_)
# logging.info(grid_search.best_score_)
#
# best_model = grid_search.best_estimator_



#
# Doing a cross-validated grid search took too much time to train
# so instead I used a smaller grid and evaluated on a single separate dev set instead of using cross-validation
#


def evaluate_model(clf, X, y_true):
    """
    Evaluate classifier on given data using AUROC metric
    """
    y_hat = clf.predict_proba(X)[:, 1]
    return roc_auc_score(y_true, y_hat)


logging.info("Start training models")
param_grid = ParameterGrid({
    'n_estimators': [50, 100],
    'max_depth': [8, 12]
})
model_data = []
for config in param_grid:
    # train model
    logging.info("Training model using params: {}".format(config))
    clf = RandomForestClassifier(class_weight='balanced', random_state=RANDOM_SEED, **config)
    clf.fit(X_train, y_train)

    # evaluate
    auroc = evaluate_model(clf, X_dev, y_dev)
    logging.info("\tAUROC score: {}".format(auroc))

    # store results
    model_data.append({
        'params': config,
        'model': clf,
        'result': auroc
    })


# save the best model
idx = 0
best_model = model_data[idx]['model']
joblib.dump(best_model, './models/random_forest.joblib')

# show feature importance
imp = pd.DataFrame(list(zip(feature_names, best_model.feature_importances_)), columns=['feature', 'importance'])
print(imp.sort_values(by='importance', ascending=False)[0:20])

#
# Evaluate the model on the test set
#
y_hat_proba = best_model.predict_proba(X_test)[:, 1]
y_hat_pred = best_model.predict(X_test)

logging.info("AUROC score: {}".format(roc_auc_score(y_test, y_hat_proba)))
logging.info("AUPRC score: {}".format(average_precision_score(y_test, y_hat_proba)))
logging.info("Accuracy: {}".format(accuracy_score(y_test, y_hat_pred)))
logging.info("Confusion matrix:")
logging.info(confusion_matrix(y_test, y_hat_pred))

