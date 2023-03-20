import pandas as pd
from data_visualization import DataPlot
from gridsearch_postprocess import GridSearchPostProcess
import utils
import sys

feat_engineering = 0
visualization = DataPlot()  # Instantiate an object for DataPlot to manage plots in this exercise
sweep = GridSearchPostProcess()  # Instantiate an object for GridSearchPostProcess to manage the grid search results
pd.set_option('display.max_columns', None)  # Enable option to display all dataframe columns
pd.set_option('display.max_colwidth', None)  # Enable printing the whole column content
df = pd.read_csv('train.csv')  # Read CSV file

column_target = 'Transported'  # Name for the target feature
# List for the categorical features
feat_cat = ['HomePlanet', 'CryoSleep', 'Deck', 'Num', 'Side', 'Destination', 'VIP']
# List for the numerical features
feat_num = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

# Scrub data and generate plots
df = utils.data_analytics(visualization, df, column_target, feat_cat, feat_num, feat_engineering)

# Data split and scaling
x_train, x_test, y_train, y_test, index_cat, index_num = utils.onehot_split(df, column_target, feat_cat, feat_num)

params = [
    {'preprocess': ['std'], 'estimator': ['knn'], 'estimator__n_neighbors': [1, 3, 5, 10, 15, 25]},
    {'preprocess': ['std'], 'estimator': ['logreg'],
     'estimator__C': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]},
    {'preprocess': ['std'], 'estimator': ['linearsvc'],
     'estimator__C': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]},
     {'preprocess': ['', 'std', 'norm'], 'estimator': ['bernoulli']},
     {'preprocess': ['', 'std', 'norm'], 'estimator': ['gaussian']},
     {'preprocess': ['', 'std', 'norm'], 'estimator': ['multinomial']},
    {'preprocess': [''], 'estimator': ['tree'], 'estimator__max_depth': [3, 5, 7, 10, 15, 25, 50]},
    {'preprocess': [''], 'estimator': ['random forest'], 'estimator__n_estimators': [50, 100, 150, 200],
     'estimator__max_depth': [5, 10, 15, 25], 'estimator__max_features': [10, 20, 40, 80]},
    {'preprocess': [''], 'estimator': ['gradient boosting'], 'estimator__n_estimators': [50, 100, 150, 200],
     'estimator__max_depth': [2, 4, 6, 8, 10], 'estimator__learning_rate': [0.05, 0.075, 0.1, 0.3, 0.5]},
    {'preprocess': ['std'], 'estimator': ['svm'],
     'estimator__gamma': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100], 'estimator__C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]},
    {'preprocess': ['std'], 'estimator': ['mlp'], 'estimator__alpha': [0.001, 0.01, 0.1, 1, 10],
     'estimator__activation': ['relu'], 'estimator__hidden_layer_sizes': [250, 500, [250, 250], [500, 500]]}]

# Create model, make predictions and plot results
pd_grid = utils.pipeline_gridsearch(x_train, x_test, y_train, y_test, index_num, params, nfolds=5, scoring='accuracy')
sweep.param_sweep_matrix(params=pd_grid['params'], test_score=pd_grid['mean_test_score'])
