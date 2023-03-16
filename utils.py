import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


def data_modeling(x_train, x_test, y_train, y_test):
    model = LogisticRegression(random_state=0)
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(x_train, y_train)
    print('LOGISTIC REGRESSION')
    print("Best parameters: {}".format(grid_search.best_params_))
    print("Best cross-validation score: {:.4f}".format(grid_search.best_score_))
    print(pd.DataFrame(grid_search.cv_results_)[['mean_test_score', 'params']])
    model = grid_search.best_estimator_
    print('LOGISTIC REGRESSION MODEL TEST SCORE: {:.4f}'.format(model.score(x_test, y_test)))


def data_preprocessing(df, column_target, feat_cat, feat_num):
    for cat in feat_cat:
        df[cat] = df[cat].astype(str)
    df = pd.get_dummies(df, columns=feat_cat)
    print(f'\nOne hot encoding dataframe size: {df.shape}\n')

    target = df[column_target].replace({True: 1, False: 0})
    df.drop(column_target, axis=1, inplace=True)
    x_train, x_test, y_train, y_test = train_test_split(df, target, test_size=0.2,
                                                        shuffle=True, stratify=target, random_state=0)

    y_test = np.array(y_test)
    y_train = np.array(y_train)

    scaling = ColumnTransformer(transformers=[('scaling', StandardScaler(), feat_num)], remainder='passthrough')
    x_train = scaling.fit_transform(x_train)
    x_test = scaling.transform(x_test)
    print(f'\nX TRAIN SHAPE: {x_train.shape}\nY TRAIN SHAPE: {y_train.shape}\n')
    print(f'\nX TEST SHAPE: {x_test.shape}\nY TEST SHAPE: {y_test.shape}\n')

    return x_train, x_test, y_train, y_test


def data_analytics(visualization, df, column_target, feat_cat, feat_num):
    print(df)
    print(f'\nOriginal dataframe size: {df.shape}\n')
    df = df.dropna(axis=0, subset=column_target)  # Consider ONLY rows with non NA values in target column
    df['Deck'] = df['Cabin'].str[0]
    df['Num'] = df['Cabin'].str[2:-2]
    df['Side'] = df['Cabin'].str[-1]
    df.drop('Cabin', axis=1, inplace=True)
    df.drop('PassengerId', axis=1, inplace=True)
    df.drop('Name', axis=1, inplace=True)

    visualization.pie_plot(df, column_target)
    visualization.cat_features(df, ['HomePlanet', 'CryoSleep', 'Deck', 'Side', 'Destination', 'VIP'], column_target)
    visualization.cat_features(df, ['Deck', 'Side'], column_target, multi=1)
    visualization.num_features(df, feat_num, column_target)

    print(f'\nInitial amount of NA values:\n')
    print(df.isna().sum())
    df_na = df.dropna()
    na_rows = df.shape[0] - df_na.shape[0]
    print('\nRows with NA values: {} ({:.1f} %)\n'.format(na_rows, 100 * na_rows / df.shape[0]))

    imputer_num = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer_cat = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    preprocessor = ColumnTransformer(transformers=[('imputer_cat', imputer_cat, feat_cat),
                                                   ('imputer_num', imputer_num, feat_num)],
                                     remainder='passthrough')

    df = preprocessor.fit_transform(df)
    df = pd.DataFrame(df, columns=feat_cat + feat_num + [column_target])
    print(f'\nAmount of NA values after applying SimpleImputer:\n')
    print(df.isna().sum())
    print(f'\nDataframe size: {df.shape}\n')
    return df
