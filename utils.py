import pandas as pd
import numpy as np
import time
from sklearn.impute import SimpleImputer
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
    pd_grid.to_csv('Results.csv', mode='w+', index=False)
    model = grid_search.best_estimator_  # Create model with best parametrization
    print('\nBEST MODEL TEST SCORE: {:.4f}\n'.format(model.score(x_test, y_test)))  # Calculate best model testing score
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


def onehot_split(df, column_target, feat_cat, feat_num):
    # Sanity check: transform categorical features to string
    for cat in feat_cat:
        df[cat] = df[cat].astype(str)
    # Apply one hot encoding to categorical features
    df = pd.get_dummies(df, columns=feat_cat)
    print(f'\nOne hot encoding dataframe size: {df.shape}\n')
    # Transform feat_cat and feat_num from strings names to index numbers
    for i, column in enumerate(list(df.columns)):
        if column in feat_cat:
            feat_cat[feat_cat.index(column)] = i
        elif column in feat_num:
            feat_num[feat_num.index(column)] = i
    # Set Target column to individual tensor with 1/0 and remove it from dataframe
    target = df[column_target].replace({True: 1, False: 0})
    df.drop(column_target, axis=1, inplace=True)
    # Create training and testing sets with 80/20% keeping target distribution same as original
    x_train, x_test, y_train, y_test = train_test_split(df, target, test_size=0.2,
                                                        shuffle=True, stratify=target, random_state=0)
    x_test = np.array(x_test)
    x_train = np.array(x_train)
    y_test = np.array(y_test)
    y_train = np.array(y_train)
    print(f'\nX TRAIN SHAPE: {x_train.shape}\nY TRAIN SHAPE: {y_train.shape}\n')
    print(f'\nX TEST SHAPE: {x_test.shape}\nY TEST SHAPE: {y_test.shape}\n')

    return x_train, x_test, y_train, y_test, feat_cat, feat_num


def data_analytics(visualization, df, column_target, feat_cat, feat_num, feat_engineering):
    print(df)
    print(f'\nOriginal dataframe size: {df.shape}\n')
    df = df.dropna(axis=0, subset=column_target)  # Consider ONLY rows with non NA values in target column
    df['Deck'] = df['Cabin'].str[0]  # Get deck information from cabin column
    df['Num'] = df['Cabin'].str[2:-2]  # Get number information from cabin column
    df['Side'] = df['Cabin'].str[-1]  # Get side information from cabin column
    df.drop('Cabin', axis=1, inplace=True)  # Remove column Cabin (no learning power)
    df.drop('PassengerId', axis=1, inplace=True)  # Remove column PassengerId (no learning power)
    df.drop('Name', axis=1, inplace=True)  # Remove column Name (no learning power)

    visualization.pie_plot(df, column_target)  # Plot target class distribution
    # Plot target grouped by individual categorical features (one subplot per categorical feature)
    visualization.cat_features(df, ['HomePlanet', 'CryoSleep', 'Deck', 'Side', 'Destination', 'VIP'], column_target)
    # # Plot target grouped by multiple categorical features (one plot per all categorical features)
    visualization.cat_features(df, ['Deck', 'Side'], column_target, multi=1)
    # # Histogram for the numerical features
    visualization.num_features(df, feat_num, column_target)

    # Check NA values and calculate the affected rows with NA values
    print(f'\nInitial amount of NA values:\n')
    print(df.isna().sum())
    df_na = df.dropna()
    na_rows = df.shape[0] - df_na.shape[0]
    print('\nRows with NA values: {} ({:.1f} %)\n'.format(na_rows, 100 * na_rows / df.shape[0]))

    # Replace NA values in numerical features for the mean value of the column
    imputer_num = SimpleImputer(missing_values=np.nan, strategy='mean')
    # Replace NA values in categorical features for the most common value of the column
    imputer_cat = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    # Apply simple imputer transformation to remove NA values in dataframe (target column is not affected - passthrough)
    preprocessor = ColumnTransformer(transformers=[('imputer_cat', imputer_cat, feat_cat),
                                                   ('imputer_num', imputer_num, feat_num)],
                                     remainder='passthrough')
    df = preprocessor.fit_transform(df)
    # Transform output simple imputer numpy array to dataframe with the correct column names
    df = pd.DataFrame(df, columns=feat_cat + feat_num + [column_target])
    print(f'\nAmount of NA values after applying SimpleImputer:\n')
    print(df.isna().sum())  # Confirm no further NA values are present in dataframe
    print(f'\nDataframe size: {df.shape}\n')
    return df
