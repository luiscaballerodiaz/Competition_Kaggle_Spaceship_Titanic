import pandas as pd
import numpy as np
import time
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
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


def pipeline_gridsearch(x_train, x_test, y_train, y_test, index_num, params, nfolds, scoring):
    time0 = time.time()
    param_grid = decode_gridsearch_params(params, index_num)
    pipe = Pipeline([('preprocess', []), ('estimator', [])])
    grid_search = GridSearchCV(pipe, param_grid, cv=nfolds, scoring=scoring)  # Define grid search cross validation
    grid_search.fit(x_train, y_train)  # Fit grid search for training set
    # Save a CSV file with the results for the grid search
    pd_grid = pd.DataFrame(grid_search.cv_results_)[['mean_test_score', 'params']]
    print(pd_grid)
    print('\nGrid search time: {:.1f}\n'.format(time.time() - time0))  # Calculate grid search timing
    print("Best parameters: {}\n".format(grid_search.best_params_))  # Show best parameters
    print("Best cross-validation score: {:.4f}\n".format(grid_search.best_score_))  # Show best scores
    model = grid_search.best_estimator_  # Create model with best parametrization
    print('\nBEST MODEL VALIDATION SCORE: {:.4f}\n'.format(model.score(x_test, y_test)))  # Calculate best model val acc
    return pd_grid


def create_preprocess(pre, columns):
    if 'norm' in pre.lower():
        preprocess = ColumnTransformer(transformers=[('scaling', MinMaxScaler(), columns)], remainder='passthrough')
    elif 'std' in pre.lower():
        preprocess = ColumnTransformer(transformers=[('scaling', StandardScaler(), columns)], remainder='passthrough')
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
    return params


def onehot_split(df, df_sub, column_target, feat_cat, feat_num):
    # Set Target column to individual tensor with 1/0 and remove it from dataframe
    target = df[column_target].replace({True: 1, False: 0})
    df.drop(column_target, axis=1, inplace=True)
    # Sanity check: transform categorical features to string
    for cat in feat_cat:
        df[cat] = df[cat].astype(str)
        df_sub[cat] = df_sub[cat].astype(str)
    # Apply one hot encoding to categorical features
    onehot = OneHotEncoder(handle_unknown='ignore')
    onehot_apply = ColumnTransformer(transformers=[('imputer_cat', onehot, feat_cat)],
                                     remainder='passthrough')
    df = onehot_apply.fit_transform(df).toarray()
    df = pd.DataFrame(df, columns=onehot_apply.get_feature_names_out())
    print(f'\nOne hot encoding dataframe size: {df.shape}\n')
    df_sub = onehot_apply.transform(df_sub).toarray()
    df_sub = pd.DataFrame(df_sub, columns=onehot_apply.get_feature_names_out())
    print(f'\nOne hot encoding submission size: {df_sub.shape}\n')
    x_sub = df_sub.to_numpy()
    # Transform feat_num from strings names to index numbers
    index_num = []
    for i, column in enumerate(list(df.columns)):
        column = column[column.index('__')+2:]
        if column in feat_num:
            index_num.append(i)
    # Create training and testing sets with 80/20% keeping target distribution same as original
    x_trainval, x_test, y_trainval, y_test = train_test_split(df, target, test_size=0.2,
                                                              shuffle=True, stratify=target, random_state=0)
    x_train, x_val, y_train, y_val = train_test_split(x_trainval, y_trainval, test_size=0.2,
                                                      shuffle=True, stratify=y_trainval, random_state=0)
    x_test = np.array(x_test)
    x_val = np.array(x_val)
    x_train = np.array(x_train)
    y_test = np.array(y_test)
    y_val = np.array(y_val)
    y_train = np.array(y_train)
    print(f'\nX TRAIN SHAPE: {x_train.shape}\nY TRAIN SHAPE: {y_train.shape}\n')
    print(f'\nX VALIDATION SHAPE: {x_val.shape}\nY VALIDATION SHAPE: {y_val.shape}\n')
    print(f'\nX TEST SHAPE: {x_test.shape}\nY TEST SHAPE: {y_test.shape}\n')
    print(f'\nX SUBMISSION SHAPE: {x_sub.shape}\n')

    return x_sub, x_train, x_val, x_test, y_train, y_val, y_test, index_num
