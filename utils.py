import time
import itertools
import copy
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures


def feat_sweep_gridsearch(test_size, df, df_full, param_dict, target, feat_cat, feat_cat_sweep, feat_num, degree):
    print('\nFIXED CATEGORICAL FEATURES: {}'.format(feat_cat))
    print('FIXED DATAFRAME FEATURES: {}'.format(list(df.columns)))
    print('SWEEP CATEGORICAL FEATURES TO CREATE NEW COMBINED FEATURE: {}\n'.format(feat_cat_sweep))
    combs = list(itertools.combinations(feat_cat_sweep, degree))
    best_scores = [0] * 10
    best_feats = [''] * 10
    for i, comb in enumerate(combs):
        print('Combination number: {}'.format(i+1))
        fcat = feat_cat.copy()
        params = copy.deepcopy(param_dict)
        fcat.append('Feat')
        df['Feat'] = df_full[list(comb)].apply(lambda x: '_'.join(x.values.astype(str)), axis=1)
        x_sub, x_train, x_test, y_train, y_test, index_num = onehot_split(test_size, df, target, fcat, feat_num)
        pd_grid, score = pipeline_gridsearch(x_train, x_test, y_train, y_test, index_num, params,
                                             'accuracy', 5)
        df.drop(['Feat'], axis=1, inplace=True)
        best_scores.append(score)
        best_feats.append(comb)
        minindex = best_scores.index(min(best_scores))
        del best_scores[minindex]
        del best_feats[minindex]
    print('BEST NEW FEATURES:\n{}'.format(best_feats))
    print('BEST SCORES FOR NEW FEATURES:\n{}'.format(best_scores))
    return best_feats, best_scores


def pipeline_gridsearch(x_train, x_test, y_train, y_test, index_num, params, scoring, nfolds,
                        pass_id=None, x_sub=None):
    time0 = time.time()
    param_grid = decode_gridsearch_params(params, index_num)
    pipe = Pipeline([('preprocess', []), ('scaling', []), ('estimator', [])])
    grid_search = GridSearchCV(pipe, param_grid, cv=nfolds, scoring=scoring)  # Define grid search cross validation
    grid_search.fit(x_train, y_train)  # Fit grid search for training set
    # Save a CSV file with the results for the grid search
    pd_grid = pd.DataFrame(grid_search.cv_results_)[['mean_test_score', 'params']]
    print(pd_grid)
    print('\nGrid search time: {:.1f}\n'.format(time.time() - time0))  # Calculate grid search timing
    print("Best parameters: {}\n".format(grid_search.best_params_))  # Show best parameters
    print("Best cross-validation score: {:.4f}\n".format(grid_search.best_score_))  # Show best scores
    model = grid_search.best_estimator_  # Create model with best parametrization
    print('\nBEST MODEL TEST SCORE: {:.4f}\n'.format(model.score(x_test, y_test)))  # Calculate best model val acc
    if pass_id is not None:
        y_sub = model.predict(x_sub)
        df_submission = pass_id.to_frame()
        df_submission['Transported'] = y_sub
        df_submission.replace({1: True, 0: False}, inplace=True)
        df_submission.to_csv('Submission.csv', index=False)
    return pd_grid, grid_search.best_score_


def create_preprocess(pre, columns):
    if 'norm' in pre.lower():
        preprocess = ColumnTransformer(transformers=[('scaling', MinMaxScaler(), columns)], remainder='passthrough')
    elif 'std' in pre.lower():
        preprocess = ColumnTransformer(transformers=[('scaling', StandardScaler(), columns)], remainder='passthrough')
    elif 'poly' in pre.lower():
        preprocess = ColumnTransformer(transformers=[('preprocess', PolynomialFeatures(), columns[1:])],
                                       remainder='passthrough')
    else:
        preprocess = None
        print('WARNING: no preprocessor was selected\n')
    return preprocess


def create_model(algorithm):
    if 'knn' in algorithm.lower():
        model = KNeighborsClassifier()
    elif 'logistic' in algorithm.lower() or 'logreg' in algorithm.lower():
        model = LogisticRegression(random_state=0)
    elif 'linear svc' in algorithm.lower() or 'linearsvc' in algorithm.lower():
        model = LinearSVC(random_state=0, dual=False)
    elif 'gaussian' in algorithm.lower():
        model = GaussianNB()
    elif 'bernoulli' in algorithm.lower():
        model = BernoulliNB()
    elif 'multinomial' in algorithm.lower():
        model = MultinomialNB()
    elif 'tree' in algorithm.lower():
        model = DecisionTreeClassifier(random_state=0)
    elif 'forest' in algorithm.lower() or 'random' in algorithm.lower():
        model = RandomForestClassifier(random_state=0)
    elif 'gradient' in algorithm.lower() or 'boosting' in algorithm.lower():
        model = GradientBoostingClassifier(random_state=0)
    elif 'svm' in algorithm.lower():
        model = SVC(random_state=0)
    elif 'mlp' in algorithm.lower():
        model = MLPClassifier(random_state=0)
    else:
        print('\nERROR: Algorithm was NOT provided. Note the type must be a list.\n')
        model = None
    return model


def decode_gridsearch_params(params, columns):
    """Assess the input grid search params defined by the user and transform them to the official sklearn estimators"""
    for i in range(len(params)):
        model = []
        for j in range(len(params[i]['estimator'])):
            model.append(create_model(params[i]['estimator'][j]))
        params[i]['estimator'] = model
        preproc = []
        for j in range(len(params[i]['preprocess'])):
            preproc.append(create_preprocess(params[i]['preprocess'][j], columns))
        params[i]['preprocess'] = preproc
        scale = []
        for j in range(len(params[i]['scaling'])):
            scale.append(create_preprocess(params[i]['scaling'][j], columns))
        params[i]['scaling'] = scale
    return params


def onehot_split(test_size, df_ini, column_target, feat_cat, feat_num, df_sub_ini=None):
    df = df_ini.copy()
    # Set Target column to individual tensor with 1/0 and remove it from dataframe
    target = df[column_target].replace({True: 1, False: 0})
    df.drop(column_target, axis=1, inplace=True)
    # Sanity check: transform categorical features to string
    for cat in feat_cat:
        df[cat] = df[cat].astype(str)
    # Apply one hot encoding to categorical features
    onehot = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    onehot_apply = ColumnTransformer(transformers=[('onehot', onehot, feat_cat)],
                                     remainder='passthrough')
    df = onehot_apply.fit_transform(df)
    df = pd.DataFrame(df, columns=onehot_apply.get_feature_names_out())
    print(f'\nOne hot encoding dataframe size: {df.shape}\n')
    # Repeat process only if submission dataframe is provided
    if df_sub_ini is not None:
        df_sub = df_sub_ini.copy()
        for cat in feat_cat:
            df_sub[cat] = df_sub[cat].astype(str)
        df_sub = onehot_apply.transform(df_sub)
        df_sub = pd.DataFrame(df_sub, columns=onehot_apply.get_feature_names_out())
        print(f'\nOne hot encoding submission size: {df_sub.shape}\n')
        x_sub = df_sub.to_numpy()
        print(f'\nX SUBMISSION SHAPE: {x_sub.shape}\n')
    else:
        x_sub = None
    # Transform feat_num from strings names to index numbers
    index_num = []
    for i, column in enumerate(list(df.columns)):
        column = column[column.index('__')+2:]
        if column in feat_num:
            index_num.append(i)
    # Create training and testing sets with 80/20% keeping target distribution same as original
    x_train, x_test, y_train, y_test = train_test_split(df, target, test_size=test_size,
                                                        shuffle=True, stratify=target, random_state=1)
    x_test = np.array(x_test)
    x_train = np.array(x_train)
    y_test = np.array(y_test)
    y_train = np.array(y_train)
    print(f'\nX TRAIN SHAPE: {x_train.shape}\nY TRAIN SHAPE: {y_train.shape}\n')
    print(f'\nX TEST SHAPE: {x_test.shape}\nY TEST SHAPE: {y_test.shape}\n')

    return x_sub, x_train, x_test, y_train, y_test, index_num


def get_sim_params(model, mode):
    if mode == 0:
        if 'gradient' in model.lower():
            params = [{'preprocess': [''], 'scaling': [''], 'estimator': ['gradient boosting'],
                       'estimator__n_estimators': [50], 'estimator__max_depth': [5], 'estimator__learning_rate': [0.1]}]
        elif 'random' in model.lower() or 'forest' in model.lower():
            params = [{'preprocess': [''], 'scaling': [''], 'estimator': ['random forest'],
                       'estimator__n_estimators': [100], 'estimator__max_depth': [10],
                       'estimator__max_features': [90]}]
        elif 'logreg' in model.lower() or 'logistic' in model.lower() or 'regression' in model.lower():
            params = [{'preprocess': [''], 'scaling': ['std'], 'estimator': ['logreg'], 'estimator__penalty': ['l1'],
                       'estimator__C': [2.5], 'estimator__solver': ['saga']}]
            # params = [{'preprocess': [''], 'scaling': ['std'], 'estimator': ['logreg'], 'estimator__C': [2.5]}]
        elif 'mlp' in model.lower():
            params = [{'preprocess': [''], 'scaling': ['std'], 'estimator': ['mlp'], 'estimator__alpha': [0.5],
                       'estimator__activation': ['relu'], 'estimator__hidden_layer_sizes': [128]}]
        elif 'svc' in model.lower():
            params = [{'preprocess': [''], 'scaling': ['std'], 'estimator': ['svm'], 'estimator__gamma': [0.005],
                       'estimator__C': [50]}]
    else:
        if 'gradient' in model.lower():
            params = [{'preprocess': [''], 'scaling': [''], 'estimator': ['gradient boosting'],
                       'estimator__n_estimators': [25, 35, 50, 65, 80, 100], 'estimator__max_depth': [3, 4, 5, 6],
                       'estimator__learning_rate': [0.01, 0.05, 0.075, 0.1, 0.15, 0.2]}]
        elif 'random' in model.lower() or 'forest' in model.lower():
            params = [{'preprocess': [''], 'scaling': [''], 'estimator': ['random forest'],
                       'estimator__n_estimators': [50, 100, 150, 200], 'estimator__max_depth': [4, 6, 8, 10, 12, 14],
                       'estimator__max_features': [50, 60, 70, 80, 90, 100]}]
        elif 'logreg' in model.lower() or 'logistic' in model.lower() or 'regression' in model.lower():
            params = [{'preprocess': [''], 'scaling': ['std'], 'estimator': ['logreg'], 'estimator__penalty': ['l1'],
                       'estimator__C': [0.01, 0.1, 0.5, 1, 2.5, 5, 7.5, 10, 25, 100], 'estimator__solver': ['saga']},
                      {'preprocess': [''], 'scaling': ['std'], 'estimator': ['logreg'], 'estimator__penalty': ['l2'],
                       'estimator__C': [0.01, 0.1, 0.5, 1, 2.5, 5, 7.5, 10, 25, 100],
                       'estimator__solver': ['saga', 'lbfgs', 'liblinear', 'newton-cholesky']}]
        elif 'mlp' in model.lower():
            params = [{'preprocess': [''], 'scaling': ['std', 'norm'], 'estimator': ['mlp'],
                       'estimator__alpha': [0.1, 0.25, 0.5, 0.75, 1, 2.5, 5], 'estimator__activation': ['relu'],
                       'estimator__hidden_layer_sizes': [64, 128, 256, [128, 64], [128, 64, 32]]}]
        elif 'svc' in model.lower():
            params = [{'preprocess': [''], 'scaling': ['std', 'norm'], 'estimator': ['svm'],
                       'estimator__gamma': [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
                       'estimator__C': [0.1, 1, 5, 10, 25, 50, 75, 100, 500]}]
    return params
