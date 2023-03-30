import pandas as pd
import numpy as np
from data_visualization import DataPlot
from gridsearch_postprocess import GridSearchPostProcess
import ensembling
import data_processing
import utils


sim = 1
# sim = 0 --> Create input feature combination plots
# sim = 1 --> Individual model grid search cross validation and generate submission for best model
# sim = 2 --> Grid search cross validation loop using as input a different new combined feature
# sim = 3 --> Ensemble models and generate submission
ensemble = ['linearsvc', 'svc']
weights = [0.575, 0.425]
th = 0.5
sim_model = 'logreg'
sim_method = 1
# sim_method = 0 --> single parametrization simulation
# sim_method = 1 --> sweep parametrization simulation
test_size = 0.2

visualization = DataPlot()  # Instantiate an object for DataPlot to manage plots in this exercise
sweep = GridSearchPostProcess()  # Instantiate an object for GridSearchPostProcess to manage the grid search results
pd.set_option('display.max_columns', None)  # Enable option to display all dataframe columns
pd.set_option('display.max_colwidth', None)  # Enable printing the whole column content
df_original = pd.read_csv('train.csv')  # Read CSV file
df_submission = pd.read_csv('test.csv')  # Read CSV file
target = 'Transported'  # Target feature
fcat_ini = ['PassengerId', 'HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'VIP', 'Name']  # Categorical features

if sim == 0:
    df_total = pd.concat([df_original, df_submission])
    data_processing.dataset_correlations(visualization, df_total)
    data_processing.dataset_correlations(visualization, df_original, target)
else:
    pass_id = df_submission['PassengerId']
    params = utils.get_sim_params(sim_model, sim_method)
    if sim == 1 or sim == 2:
        # Apply feature engineering
        df, feat_cat, feat_num, df_full, feat_cat_sweep = data_processing.feature_engineering(
            visualization, df_original, fcat_ini, model=sim_model, column_target=target)
        df_sub, _, _, _, _ = data_processing.feature_engineering(
            visualization, df_submission, fcat_ini, model=sim_model)
        # One hot encoding and train, val and test split
        x_sub, x_train, x_test, y_train, y_test, index_num = utils.onehot_split(
            test_size, df, target, feat_cat, feat_num, df_sub)
        # Model simulation depending on sim parametrization
        if sim == 1:
            pd_grid, score = utils.pipeline_gridsearch(x_train, x_test, y_train, y_test, index_num, params,
                                                       'accuracy', 5, pass_id=pass_id, x_sub=x_sub)
            sweep.param_sweep_matrix(params=pd_grid['params'], test_score=pd_grid['mean_test_score'])
        elif sim == 2:
            best_feats, best_scores = utils.feat_sweep_gridsearch(
                test_size, df, df_full, params, target, feat_cat, feat_cat_sweep, feat_num, 3)
    elif sim == 3:
        tot_test_pred = np.zeros([round(len(df_original) * test_size), len(ensemble)])
        tot_sub_pred = np.zeros([len(df_submission), len(ensemble)])
        for i, current_model in enumerate(ensemble):
            # Apply feature engineering
            df, feat_cat, feat_num, _, _ = data_processing.feature_engineering(
                visualization, df_original, fcat_ini, model=current_model, column_target=target)
            df_sub, _, _, _, _ = data_processing.feature_engineering(
                visualization, df_submission, fcat_ini, model=current_model)
            # One hot encoding and train, val and test split
            x_sub, x_train, x_test, y_train, y_test, index_num = utils.onehot_split(
                test_size, df, target, feat_cat, feat_num, df_sub)
            test_pred, sub_pred = ensembling.make_predictions(
                current_model, x_sub, x_train, x_test, y_train, y_test, index_num)
            # Store the algorithm results in a combined matrix
            tot_test_pred[:, i] = test_pred
            tot_sub_pred[:, i] = sub_pred
        ensembling.ensemble_predictions(pass_id, weights, th, tot_test_pred, tot_sub_pred, y_test)
