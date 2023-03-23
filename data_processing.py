import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer


def feature_engineering(visualization, df, feat_cat, feat_num, plots, column_target=''):
    print(df)
    print(f'\nOriginal dataframe size: {df.shape}\n')
    df.replace('Europe', 'Europa')

    if column_target != '':
        visualization.pie_plot(df, column_target)  # Plot target class distribution
        df = df.dropna(axis=0, subset=column_target)  # Consider ONLY rows with non NA values in target column

    # Check NA values and calculate the affected rows with NA values
    print(f'\nInitial amount of NA values:\n')
    print(df.isna().sum())
    df_na = df.dropna()
    na_rows = df.shape[0] - df_na.shape[0]
    print('\nRows with NA values: {} ({:.1f} %)\n'.format(na_rows, 100 * na_rows / df.shape[0]))

    # Prepare input required for sim = 0 (no feature engineering and remove NA rows)
    feat_num_na = feat_num.copy()
    feat_cat_na = feat_cat.copy()
    df_na, feat_cat_na = check_group_correlation(df_na, feat_cat_na)

    # Prepare input required for sim = 1 (no feature engineering and apply simple feature for NA rows)
    imp_num = SimpleImputer(missing_values=np.nan, strategy='mean')  # Replace NA for num feats for mean
    imp_cat = SimpleImputer(missing_values=np.nan, strategy='most_frequent')  # Replace NA in cat feats for most common
    # Apply simple imputer transformation to remove NA values in dataframe (target column is not affected - passthrough)
    preprocessor = ColumnTransformer(transformers=[('imputer_cat', imp_cat, feat_cat_na),
                                                   ('imputer_num', imp_num, feat_num_na)],
                                     remainder='passthrough')
    df_imp = df.drop(['PassengerId', 'Name'], axis=1, inplace=False)
    df_imp = preprocessor.fit_transform(df_imp)
    if column_target != '':
        df_imp = pd.DataFrame(df_imp, columns=feat_cat_na + feat_num_na + [column_target])  # Transform np array to df

    df['Deck'] = df['Cabin'].str[0]
    df['Num'] = df['Cabin'].str[2:-2]
    df['Side'] = df['Cabin'].str[-1]
    df['Group'] = df['PassengerId'].str[:4].astype('int32')
    df['GroupSize'] = df['Group'].apply(lambda x: df['Group'].value_counts()[x])

    df = fill_missing_values(df)  # Fill in missing values

    # TRIP
    df['Trip'] = np.where(
        (df['HomePlanet'] == 'Earth') & (df['Destination'] == 'TRAPPIST-1e'), 1, np.where(
            (df['HomePlanet'] == 'Earth') & (df['Destination'] == '55 Cancri e'), 2, np.where(
                (df['HomePlanet'] == 'Earth') & (df['Destination'] == 'PSO J318.5-22'), 3, np.where(
                    (df['HomePlanet'] == 'Europa') & (df['Destination'] == 'TRAPPIST-1e'), 4, np.where(
                        (df['HomePlanet'] == 'Europa') & (df['Destination'] == '55 Cancri e'), 5, np.where(
                            (df['HomePlanet'] == 'Europa') & (df['Destination'] == 'PSO J318.5-22'), 6, np.where(
                                (df['HomePlanet'] == 'Mars') & (df['Destination'] == 'TRAPPIST-1e'), 7, np.where(
                                    (df['HomePlanet'] == 'Mars') & (df['Destination'] == '55 Cancri e'), 8, np.where(
                                        (df['HomePlanet'] == 'Mars') & (df['Destination'] == 'PSO J318.5-22'), 9,
                                        -1)))))))))

    # CABIN
    df['Num'] = df['Num'].astype('int32')
    df['NumBin'] = np.where(
        df['Num'] < 300, 1, np.where(
            df['Num'] < 600, 2, np.where(
                df['Num'] < 900, 3, np.where(
                    df['Num'] < 1200, 4, np.where(
                        df['Num'] < 1500, 5, 6)))))
    df['ShipLocation'] = df[['NumBin', 'Deck', 'Side']].apply(lambda x: '_'.join(x.values.astype(str)), axis=1)

    # PURCHASES
    df['SpendingBin'] = np.where((df['RoomService'] + df['FoodCourt'] + df['ShoppingMall'] +
                                  df['Spa'] + df['VRDeck']) > 0, True, False)
    df['Spending'] = df[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].apply(
         lambda x: sum(x.values > 0), axis=1)
    df['TotalSpending'] = df[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].apply(
        lambda x: sum(x.values), axis=1)

    # AGE
    df['AgeBin'] = np.where(df['Age'] < 18, 1, np.where(df['Age'] < 40, 2, 3))

    # COMBINED FEATURE
    df['Feat'] = df[['CryoSleep', 'ShipLocation', 'HomePlanet']].apply(lambda x: '_'.join(x.values.astype(str)), axis=1)

    # Update list for cat feats to plot and model (note it does not include feats with high categories for plotting)
    feat_cat = [feat for feat in feat_cat if feat not in ['Name', 'Cabin', 'PassengerId']]
    feat_cat_plot = feat_cat.copy()
    feat_cat.extend(['Feat', 'GroupSize', 'Deck', 'Side', 'AgeBin', 'NumBin', 'Trip', 'ShipLocation'])
    feat_cat_plot.extend(['GroupSize', 'Deck', 'Side', 'AgeBin', 'NumBin', 'Trip', 'SpendingBin'])

    if plots == 1 or plots == 3:
        visualization.num_features(df, feat_num='Age', binary_feat=column_target, bins_width=1)
        visualization.num_features(df, feat_num='Num', binary_feat=column_target, bins_width=25)
        visualization.num_features(df, feat_num='RoomService', binary_feat=column_target, bins_width=100)
        visualization.num_features(df, feat_num='FoodCourt', binary_feat=column_target, bins_width=100)
        visualization.num_features(df, feat_num='ShoppingMall', binary_feat=column_target, bins_width=100)
        visualization.num_features(df, feat_num='Spa', binary_feat=column_target, bins_width=100)
        visualization.num_features(df, feat_num='VRDeck', binary_feat=column_target, bins_width=100)
        visualization.combined_features(df, feat_cat_plot, column_target)
        visualization.cat_features(df, feat_cat_plot, column_target)
        for feat in feat_cat_plot:
            visualization.cat_features(df, feat_cat_plot, feat)
    if plots == 2 or plots == 3:
        for feat in feat_cat_plot:
            visualization.combined_features(df, feat_cat_plot, feat)

    # Remove categorical features not to consider for modeling
    df.drop(['Group', 'PassengerId', 'Cabin', 'Name', 'SpendingBin', 'Spending', 'TotalSpending', 'Num'],
            axis=1, inplace=True)

    print(f'\nAmount of NA values after processing:\n')
    print(df.isna().sum())  # Confirm no further NA values are present in dataframe
    print(f'\nDataframe size: {df.shape}\n')

    return df, feat_cat, feat_num, df_na, df_imp, feat_cat_na, feat_num_na


def check_group_correlation(df_base, feat_cat_base):
    df_base['Group'] = df_base['PassengerId'].str[:4].astype('int32')
    df_base['Deck'] = df_base['Cabin'].str[0]
    df_base['Num'] = df_base['Cabin'].str[2:-2]
    df_base['Side'] = df_base['Cabin'].str[-1]
    feat_cat_base.extend(['Deck', 'Side', 'Num'])
    print('\nNumber of groups: {}'.format(df_base['Group'].nunique()))
    for cat in feat_cat_base:
        print('Number of rows for {}: {}'.format(cat, len(df_base.groupby(['Group', cat]).size())))
    print('\n')
    df_base.drop(['PassengerId', 'Group', 'Deck', 'Num', 'Side', 'Name'], axis=1, inplace=True)
    feat_cat_base = [feat for feat in feat_cat_base if feat not in ['Name', 'Num', 'PassengerId', 'Deck', 'Side']]
    return df_base, feat_cat_base


def fill_missing_values(df):
    list_spend = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    # #### SAME GROUP #####
    # Same group has the same home planet, side and most cases same number
    home_na = df['HomePlanet'].isna()
    for i in range(1, len(home_na) - 1):
        if home_na[i]:
            if df.loc[i - 1, 'Group'] == df.loc[i, 'Group']:
                df.loc[i, 'HomePlanet'] = df.loc[i - 1, 'HomePlanet']
            elif df.loc[i + 1, 'Group'] == df.loc[i, 'Group']:
                df.loc[i, 'HomePlanet'] = df.loc[i + 1, 'HomePlanet']
    side_na = df['Side'].isna()
    for i in range(1, len(side_na) - 1):
        if side_na[i]:
            if df.loc[i - 1, 'Group'] == df.loc[i, 'Group']:
                df.loc[i, 'Side'] = df.loc[i - 1, 'Side']
                df.loc[i, 'Num'] = df.loc[i - 1, 'Num']
                df.loc[i, 'Deck'] = df.loc[i - 1, 'Deck']
            elif df.loc[i + 1, 'Group'] == df.loc[i, 'Group']:
                df.loc[i, 'Side'] = df.loc[i + 1, 'Side']
                df.loc[i, 'Num'] = df.loc[i + 1, 'Num']
                df.loc[i, 'Deck'] = df.loc[i + 1, 'Deck']

    # #### NUM AND DECK FEATURE #####
    # If HomePlanet is Europa --> NumBin = 1 (it assigns Num to 150)
    # If Deck is A, B, C, D --> NumBin = 1 (it assigns Num to 150)
    num_na = df['Num'].isna()
    for i in range(len(num_na)):
        if num_na[i]:
            df.loc[i, 'Num'] = 150
            if df.loc[i, 'HomePlanet'] == 'Mars' or df.loc[i, list_spend].sum() > 0:
                df.loc[i, 'Deck'] = 'F'
            else:
                df.loc[i, 'Deck'] = 'G'

    # #### SIDE FEATURE #####
    #  If Transported TRUE --> Side S, otherwise Side P
    # side_na = df['Side'].isna()
    # for i in range(len(side_na)):
    #     if side_na[i]:
    #         df.loc[i, 'Side'] = 'P'

    # #### PURCHASE FEATURES #####
    # If CryoSleep TRUE --> No purchases
    # Single purchase is very unlikely --> if other purchases are 0, the missing purchase is assigned to 0
    # No purchases for kids (Age <= 18)
    for spend in list_spend:
        spending_na = df[spend].isna()
        for i in range(len(spending_na)):
            if spending_na[i]:
                if df.loc[i, 'CryoSleep'] or df.loc[i, list_spend].sum() == 0 or df.loc[i, 'Age'] <= 18:
                    df.loc[i, spend] = 0
                else:
                    df.loc[i, spend] = 250

    # #### VIP FEATURE #####
    # VIP assigned to False due to the unbalanced distribution
    df['VIP'].fillna(False, inplace=True)

    # #### DESTINATION FEATURE #####
    # If Deck D and E OR Homeplanet Mars OR NumBin 6 (> 1500) --> Destination TRAPPIST-1e
    # Assume all missing points are assigned to the most common option
    dest_na = df['Destination'].isna()
    for i in range(len(dest_na)):
        if dest_na[i]:
            df.loc[i, 'Destination'] = 'TRAPPIST-1e'

    # #### AGE FEATURE #####
    # No purchases --> Age < 18 (it assigns Age to 10)
    age_na = df['Age'].isna()
    for i in range(len(age_na)):
        if age_na[i]:
            if df.loc[i, list_spend].sum() == 0:
                df.loc[i, 'Age'] = 10
            else:
                df.loc[i, 'Age'] = 30

    # #### CRYOSLEEP FEATURE #####
    # If Transported OR no purchases --> CryoSleep True
    # If purchases are done --> CryoSleep False
    cryo_na = df['CryoSleep'].isna()
    for i in range(len(cryo_na)):
        if cryo_na[i]:
            if df.loc[i, list_spend].sum() == 0:
                df.loc[i, 'CryoSleep'] = True
            else:
                df.loc[i, 'CryoSleep'] = False

    # #### HOME PLANET FEATURE #####
    # Flights with group size 8 only has home planet Earth
    # Decks A, B, C are always Europa and G Earth
    # Destination PSO J318.5-22 always when home planet Earth
    # Age < 18 when home planet Earth
    home_na = df['HomePlanet'].isna()
    for i in range(len(home_na)):
        if home_na[i]:
            if str(df.loc[i, 'Deck'])[0] == 'A' or str(df.loc[i, 'Deck'])[0] == 'B' or str(df.loc[i, 'Deck'])[0] == 'C':
                df.loc[i, 'HomePlanet'] = 'Europa'
            else:
                df.loc[i, 'HomePlanet'] = 'Earth'
    return df
