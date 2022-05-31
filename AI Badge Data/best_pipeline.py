import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('AI_Badge_03/AI Badge Data/prepared_data.csv')
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: 1.0
exported_pipeline = GradientBoostingClassifier(learning_rate=0.5, max_depth=3, max_features=0.5, min_samples_leaf=19, min_samples_split=5, n_estimators=100, subsample=0.35000000000000003)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
