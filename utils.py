import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def modeling(x_train, x_test, y_train, y_test):
    model = LogisticRegression(random_state=0)  # Create model
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}  # Define model parameter sweep
    grid_search = GridSearchCV(model, param_grid, cv=5)  # Define grid search with 5 splits cross validation
    grid_search.fit(x_train, y_train)  # Fit grid search for training set
    print('LOGISTIC REGRESSION')
    print("Best parameters: {}".format(grid_search.best_params_))  # Show best parameters
    print("Best cross-validation score: {:.4f}".format(grid_search.best_score_))  # Show best scores
    print(pd.DataFrame(grid_search.cv_results_)[['mean_test_score', 'params']])  # Show mean cross validation scores
    model = grid_search.best_estimator_  # Create model with best parametrization
    print('BEST MODEL TEST SCORE: {:.4f}'.format(model.score(x_test, y_test)))  # Calculate testing score for best model


def split_scaling(df, column_target, feat_cat, feat_num):
    # Sanity check: transform categorical features to string
    for cat in feat_cat:
        df[cat] = df[cat].astype(str)
    # Apply one hot encoding to categorical features
    df = pd.get_dummies(df, columns=feat_cat)
    print(f'\nOne hot encoding dataframe size: {df.shape}\n')

    # Set Target column to individual tensor with 1/0 and remove it from dataframe
    target = df[column_target].replace({True: 1, False: 0})
    df.drop(column_target, axis=1, inplace=True)
    # Create training and testing sets with 80/20% keeping target distribution same as original
    x_train, x_test, y_train, y_test = train_test_split(df, target, test_size=0.2,
                                                        shuffle=True, stratify=target, random_state=0)
    y_test = np.array(y_test)
    y_train = np.array(y_train)
    # Apply scaling (null mean with unit std) in numerical features (categorical features not modified)
    scaling = ColumnTransformer(transformers=[('scaling', StandardScaler(), feat_num)], remainder='passthrough')
    # Fit scaling based on training set and transform it accordingly
    x_train = scaling.fit_transform(x_train)
    # Transform testing set accordingly to scaling
    x_test = scaling.transform(x_test)
    print(f'\nX TRAIN SHAPE: {x_train.shape}\nY TRAIN SHAPE: {y_train.shape}\n')
    print(f'\nX TEST SHAPE: {x_test.shape}\nY TEST SHAPE: {y_test.shape}\n')

    return x_train, x_test, y_train, y_test


def data_analytics(visualization, df, column_target, feat_cat, feat_num):
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
    # Plot target grouped by multiple categorical features (one plot per all categorical features)
    visualization.cat_features(df, ['Deck', 'Side'], column_target, multi=1)
    # Histogram for the numerical features
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
