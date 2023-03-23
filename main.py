import pandas as pd
from data_visualization import DataPlot
from gridsearch_postprocess import GridSearchPostProcess
import ensembling
import data_processing
import utils

sim = 2
# sim = 0 --> No feature engineering and remove all NA values (Train 0.7864 and validation 0.7767)
# sim = 1 --> No feature engineering and apply simple imputer to manage NA values (Train 0.7843 and validation 0.7721)
# sim = 2 --> Feature engineering and smart NA management individual model grid search parametrization
ensemble = 0
# ensemble = 0 --> Individual model grid search cross validation
# ensemble = 1 --> Ensemble models and generate submission
plots = 0
# plots = 0 --> No plots
# plots = 1 --> Only the non-combined plots are created
# plots = 2 --> Only the combined plots are created (500)
# plots = 3 --> All plots are created
params = [
    {'preprocess': ['std'], 'estimator': ['knn'], 'estimator__n_neighbors': [1, 3, 5, 10, 15, 25]},
    {'preprocess': ['std'], 'estimator': ['logreg'],
     'estimator__C': [0.3, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 3.5, 4]},
    {'preprocess': ['std'], 'estimator': ['linearsvc'],
     'estimator__C': [0.25]},
    {'preprocess': [''], 'estimator': ['tree'], 'estimator__max_depth': [3, 5, 7, 10, 15, 25, 50]},
    {'preprocess': [''], 'estimator': ['random forest'], 'estimator__n_estimators': [50, 100, 150, 200],
     'estimator__max_depth': [6, 8, 10, 12, 14], 'estimator__max_features': [50, 60, 70, 80, 90, 100]},
    {'preprocess': [''], 'estimator': ['gradient boosting'], 'estimator__n_estimators': [25, 35, 50, 65, 80, 100],
     'estimator__max_depth': [3, 4, 5, 6], 'estimator__learning_rate': [0.01, 0.05, 0.075, 0.1, 0.15, 0.2]},
    {'preprocess': ['std'], 'estimator': ['svm'], 'estimator__gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05],
     'estimator__C': [1, 5, 10, 50, 100, 150, 250, 500]},
    {'preprocess': ['std'], 'estimator': ['mlp'], 'estimator__alpha': [0.5, 1, 2.5, 5],
     'estimator__activation': ['relu'], 'estimator__hidden_layer_sizes': [100, 200, 300, [100, 100]]}]

visualization = DataPlot()  # Instantiate an object for DataPlot to manage plots in this exercise
sweep = GridSearchPostProcess()  # Instantiate an object for GridSearchPostProcess to manage the grid search results
pd.set_option('display.max_columns', None)  # Enable option to display all dataframe columns
pd.set_option('display.max_rows', None)  # Enable option to display all dataframe columns
pd.set_option('display.max_colwidth', None)  # Enable printing the whole column content
df = pd.read_csv('train.csv')  # Read CSV file
df_submission = pd.read_csv('test.csv')  # Read CSV file
target = 'Transported'  # Target feature
feat_cat_ini = ['PassengerId', 'HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'VIP', 'Name']  # Categorical features
feat_num_ini = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']  # Numerical features

pass_id = df_submission['PassengerId']
# Apply feature engineering
df, feat_cat, feat_num, df_na, df_imp, feat_cat_na, feat_num_na = data_processing.feature_engineering(
    visualization, df, feat_cat_ini, feat_num_ini, plots, column_target=target)
df_sub, _, _, _, _, _, _ = data_processing.feature_engineering(visualization, df_submission, feat_cat_ini, feat_num_ini, 0)

# Data split and scaling
if sim == 0:
    x_sub, x_train, x_val, x_test, y_train, y_val, y_test, index_num = utils.onehot_split(df_na, df_sub, target,
                                                                                          feat_cat_na, feat_num_na)
elif sim == 1:
    x_sub, x_train, x_val, x_test, y_train, y_val, y_test, index_num = utils.onehot_split(df_imp, df_sub, target,
                                                                                          feat_cat_na, feat_num_na)
else:
    x_sub, x_train, x_val, x_test, y_train, y_val, y_test, index_num = utils.onehot_split(df, df_sub, target,
                                                                                          feat_cat, feat_num)
if ensemble == 1:
    ensembling.ensemble_models(pass_id, x_sub, x_train, x_val, x_test, y_train, y_val, y_test, index_num, 250)
else:
    # Create model, make predictions and plot results
    pd_grid = utils.pipeline_gridsearch(x_train, x_val, y_train, y_val, index_num, params, nfolds=5, scoring='accuracy')
    sweep.param_sweep_matrix(params=pd_grid['params'], test_score=pd_grid['mean_test_score'])
