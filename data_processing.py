import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer


def dataset_correlations(visualization, df, fcat):
    feat_cat = fcat.copy()
    df = df.dropna()
    df['Deck'] = df['Cabin'].str[0]
    df['Num'] = df['Cabin'].str[2:-2]
    df['Side'] = df['Cabin'].str[-1]
    df['Group'] = df['PassengerId'].str[:4].astype('int32')
    df['GroupSize'] = df['Group'].apply(lambda x: df['Group'].value_counts()[x])
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
    df['Num'] = df['Num'].astype('int32')
    df['NumBin'] = np.where(
        df['Num'] < 300, 1, np.where(
            df['Num'] < 600, 2, np.where(
                df['Num'] < 900, 3, np.where(
                    df['Num'] < 1200, 4, np.where(
                        df['Num'] < 1500, 5, 6)))))
    df['SpendingBin'] = np.where((df['RoomService'] + df['FoodCourt'] + df['ShoppingMall'] +
                                  df['Spa'] + df['VRDeck']) > 0, True, False)
    df['AgeBin'] = np.where(df['Age'] < 18, 1, np.where(df['Age'] < 40, 2, 3))

    # Update list for cat feats to plot and model (note it does not include feats with high categories for plotting)
    feat_cat_plot = [feat for feat in feat_cat if feat not in ['Name', 'Cabin', 'PassengerId']]
    feat_cat_plot.extend(['GroupSize', 'Deck', 'Side', 'AgeBin', 'NumBin', 'Trip', 'SpendingBin'])
    for feat in feat_cat_plot:
        visualization.cat_features(df, feat_cat_plot, feat)
    for feat in feat_cat_plot:
        visualization.combined_features(df, feat_cat_plot, feat)


def feature_engineering(visualization, df_ini, fcat, fnum, sim, plots=0, column_target=''):
    feat_cat = fcat.copy()
    feat_num = fnum.copy()
    df = df_ini.copy()
    print(f'\nOriginal dataframe size: {df.shape}')
    df.replace('Europe', 'Europa')

    if column_target != '':
        visualization.pie_plot(df, column_target)  # Plot target class distribution
        df = df.dropna(axis=0, subset=column_target)  # Consider ONLY rows with non NA values in target column

    # Check NA values and calculate the affected rows with NA values
    print(f'Initial amount of NA values:')
    print(df.isna().sum())
    df_na = df.dropna()
    na_rows = df.shape[0] - df_na.shape[0]
    print('Rows with NA values: {} ({:.1f} %)'.format(na_rows, 100 * na_rows / df.shape[0]))

    # Check the feature deviations among same groups
    check_group_correlation(df_na, feat_cat)

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

    # PURCHASES
    df['FoodBin'] = np.where(df['FoodCourt'] > 0, True, False)
    df['MallBin'] = np.where(df['ShoppingMall'] > 0, True, False)
    df['SpaBin'] = np.where(df['Spa'] > 0, True, False)
    df['DeckBin'] = np.where(df['VRDeck'] > 0, True, False)
    df['RoomBin'] = np.where(df['RoomService'] > 0, True, False)
    df['SpendingDist'] = df[['RoomBin', 'MallBin', 'DeckBin', 'SpaBin', 'FoodBin']].apply(lambda x: '_'.join(x.values.astype(str)), axis=1)
    df['SpendingBin'] = np.where((df['RoomService'] + df['FoodCourt'] + df['ShoppingMall'] +
                                  df['Spa'] + df['VRDeck']) > 0, True, False)
    df['Spending'] = df[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].apply(
         lambda x: sum(x.values > 0), axis=1)
    df['TotalSpending'] = df[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].apply(
        lambda x: sum(x.values), axis=1)

    # AGE
    df['AgeBin'] = np.where(df['Age'] < 18, 1, np.where(df['Age'] < 40, 2, 3))

    # Visualization of the num and cat features
    if plots == 1:
        list_cat_plot = ['HomePlanet', 'CryoSleep', 'Destination', 'Deck', 'AgeBin', 'NumBin', 'SpendingBin', 'Trip',
                         'Side']
        list_num_plot = ['RoomService', 'ShoppingMall', 'FoodCourt', 'Spa', 'VRDeck', 'TotalSpending', 'Age']
        for feat in list_num_plot:
            if feat == 'Age':
                binw = 1
            elif feat == 'Num':
                binw = 25
            else:
                binw = 100
            visualization.num_features(df, feat_num=feat, binary_feat=column_target, bins_width=binw)
        visualization.combined_features(df, list_cat_plot, column_target)
        visualization.cat_features(df, list_cat_plot, column_target)

    # COMBINED FEATURE
    df['ShipLocation'] = df[['NumBin', 'Deck', 'Side']].apply(lambda x: '_'.join(x.values.astype(str)), axis=1)
    if sim != 2:
        df['F1'] = df[['HomePlanet', 'CryoSleep']].apply(lambda x: '_'.join(x.values.astype(str)), axis=1)
        df['F2'] = df[['Deck', 'SpendingBin']].apply(lambda x: '_'.join(x.values.astype(str)), axis=1)
        df['F3'] = df[['CryoSleep', 'Trip']].apply(lambda x: '_'.join(x.values.astype(str)), axis=1)
        df['F4'] = df[['HomePlanet', 'Deck', 'SpendingDist']].apply(lambda x: '_'.join(x.values.astype(str)), axis=1)
        df['F5'] = df[['Destination', 'MallBin']].apply(lambda x: '_'.join(x.values.astype(str)), axis=1)
        df['F6'] = df[['AgeBin', 'FoodBin']].apply(lambda x: '_'.join(x.values.astype(str)), axis=1)
        df['F7'] = df[['Trip', 'RoomBin']].apply(lambda x: '_'.join(x.values.astype(str)), axis=1)

    # Create definitive dataframe and list of num and cat features
    df_post = pd.DataFrame()
    list_cat = ['HomePlanet', 'CryoSleep', 'Destination', 'Deck', 'AgeBin', 'NumBin', 'ShipLocation', 'SpendingBin',
                'Trip']
    if sim != 2:
        list_cat.extend(['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7'])
    list_num = ['RoomService', 'ShoppingMall', 'FoodCourt', 'Spa', 'VRDeck', 'TotalSpending', 'Age']
    df_post[list_cat] = df[list_cat]
    df_post[list_num] = df[list_num]
    if column_target != '':
        df_post[column_target] = df[column_target]

    # Create full dataframe with all num and cat features to sweep parameters and look for the most optimal combinations
    all_cat = ['HomePlanet', 'CryoSleep', 'Destination', 'Deck', 'AgeBin', 'NumBin', 'ShipLocation', 'SpendingBin',
               'Trip', 'Side', 'Spending', 'SpendingDist', 'FoodBin', 'MallBin', 'SpaBin','DeckBin', 'RoomBin',
               'GroupSize', 'Group', 'PassengerId', 'Cabin', 'Name', 'VIP']

    df_full = df.copy()
    list_num_full = ['RoomService', 'ShoppingMall', 'FoodCourt', 'Spa', 'VRDeck', 'TotalSpending', 'Age']
    list_cat_full = ['HomePlanet', 'CryoSleep', 'Destination', 'Deck', 'AgeBin', 'NumBin', 'ShipLocation',
                     'SpendingBin', 'Trip', 'FoodBin', 'MallBin', 'SpaBin', 'DeckBin', 'RoomBin']
    remove_feat = [f for f in all_cat if f not in list_cat_full]
    df_full.drop(remove_feat, axis=1, inplace=True)

    print(f'Output dataframe size: {df_post.shape}\n')

    return df_post, list_cat, list_num, df_full, list_cat_full, list_num_full


def check_group_correlation(df_base, fcat):
    feat_cat = fcat.copy()
    df_base['Group'] = df_base['PassengerId'].str[:4].astype('int32')
    df_base['Deck'] = df_base['Cabin'].str[0]
    df_base['Num'] = df_base['Cabin'].str[2:-2]
    df_base['Side'] = df_base['Cabin'].str[-1]
    feat_cat.extend(['Deck', 'Side', 'Num'])
    print('\nNumber of groups: {}'.format(df_base['Group'].nunique()))
    for cat in feat_cat:
        print('Number of rows for {}: {}'.format(cat, len(df_base.groupby(['Group', cat]).size())))
    print('\n')
    df_base.drop(['PassengerId', 'Group', 'Deck', 'Num', 'Side', 'Name'], axis=1, inplace=True)


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
    # If VIP TRUE --> NumBin = 1 (it assigns Num to 150)
    # If Deck is A, B, C, D --> NumBin = 1 (it assigns Num to 150)
    # If Destination 55 Cancri e --> NumBin = 1 (it assigns Num to 150)

    # If Numbin 6 --> Deck F
    # If home planet Mars --> Deck F
    # If Numbin 3, 4, 5 and no spending --> Deck G
    # If home planet Earth and (no spending or age < 18 or cryosleep true) --> Deck G
    # If destination PSO J318.5-22 and (no spending or age < 18 or cryosleep true) --> Deck G
    num_na = df['Num'].isna()
    for i in range(len(num_na)):
        if num_na[i]:
            if df.loc[i, 'HomePlanet'] == 'Europa' or df.loc[i, 'VIP'] or df.loc[i, 'Deck'] in ['A', 'B', 'C', 'D']:
                df.loc[i, 'Num'] = 150
            else:
                df.loc[i, 'Num'] = df['Num'].value_counts().index[0]
            if df.loc[i, 'HomePlanet'] == 'Mars' or float(df.loc[i, 'Num']) >= 1500:
                df.loc[i, 'Deck'] = 'F'
            elif (df.loc[i, 'HomePlanet'] == 'Earth' or df.loc[i, 'Destination'] == 'PSO J318.5-22') and \
                    (df.loc[i, list_spend].sum() == 0 or df.loc[i, 'CryoSleep'] or df.loc[i, 'Age'] <= 18):
                df.loc[i, 'Deck'] = 'G'
            else:
                df.loc[i, 'Deck'] = df['Deck'].value_counts().index[0]

    # #### SIDE FEATURE #####
    # If homePlanet Europa --> Side S
    # If home planet Earth and Numbin 1 --> Side S
    # If Numbin 2 and Deck B/C --> Side S
    # If Numbin 3 and Deck E --> Side S
    # If Group size 7 --> Side S

    # If home planet Mars --> Side P
    # If Group size 8 --> Side P
    side_na = df['Side'].isna()
    for i in range(len(side_na)):
        if side_na[i]:
            if df.loc[i, 'GroupSize'] == 8 or float(df.loc[i, 'Num']) >= 1500 or df.loc[i, 'HomePlanet'] == 'Mars':
                df.loc[i, 'Side'] = 'P'
            elif df.loc[i, 'GroupSize'] == 7 or df.loc[i, 'HomePlanet'] == 'Europa' \
                    or ((df.loc[i, 'HomePlanet'] == 'Earth') and (float(df.loc[i, 'Num']) <= 300)) \
                    or ((df.loc[i, 'Deck'] in ['B', 'C']) and (float(df.loc[i, 'Num']) >= 300) and
                        (float(df.loc[i, 'Num']) <= 600)) \
                    or ((df.loc[i, 'Deck'] in ['E']) and (float(df.loc[i, 'Num']) >= 600) and
                        (float(df.loc[i, 'Num']) <= 900)):
                df.loc[i, 'Side'] = 'S'
            else:
                df.loc[i, 'Side'] = df['Side'].value_counts().index[0]

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
                    df.loc[i, spend] = 1000

    # #### VIP FEATURE #####
    # VIP assigned to False due to the unbalanced distribution
    df['VIP'].fillna(False, inplace=True)

    # #### DESTINATION FEATURE #####
    # If Deck D, E, F, G --> TRAPPIST-1e
    # If Numbin 3, 4, 5, 6 --> TRAPPIST-1e
    # If home planet Mars --> TRAPPIST-1e
    dest_na = df['Destination'].isna()
    for i in range(len(dest_na)):
        if dest_na[i]:
            if df.loc[i, 'HomePlanet'] == 'Mars' or float(df.loc[i, 'Num']) >= 600 or \
                    df.loc[i, 'Deck'] in ['D', 'E', 'F', 'G']:
                df.loc[i, 'Destination'] = 'TRAPPIST-1e'
            else:
                df.loc[i, 'Destination'] = df['Destination'].value_counts().index[0]

    # #### AGE FEATURE #####
    # If Deck G and group size 3, 4, 5, 6, 7, 8 --> age 1
    # If purchases --> age 2
    age_na = df['Age'].isna()
    for i in range(len(age_na)):
        if age_na[i]:
            if df.loc[i, list_spend].sum() >= 0:
                df.loc[i, 'Age'] = 30
            elif df.loc[i, 'GroupSize'] >= 3 or df.loc[i, 'Deck'] in ['G']:
                df.loc[i, 'Age'] = 10
            else:
                df.loc[i, 'Age'] = df['Age'].value_counts().index[0]

    # #### CRYOSLEEP FEATURE #####
    # If no purchases --> CryoSleep True
    # If purchases are done --> CryoSleep False
    cryo_na = df['CryoSleep'].isna()
    for i in range(len(cryo_na)):
        if cryo_na[i]:
            if df.loc[i, list_spend].sum() == 0:
                df.loc[i, 'CryoSleep'] = True
            else:
                df.loc[i, 'CryoSleep'] = False

    # #### HOME PLANET FEATURE #####
    # If VIP TRUE and NumBin 1 --> Europa
    # If VIP TRUE and destination 55 Cancri e --> Europa
    # If Numbin 1 and destination 55 Cancri e --> Europa
    # If Decks A, B, C --> Europa

    # If Numbin 3, 4, 5, 6 and destination 55 Cancri e --> Earth
    # If destination PSO J318.5-22 --> Earth
    # If Age < 18 and Numbin 2, 3, 4, 5 --> Earth
    # If group size 6, 7, 8 --> Earth
    # If Deck G --> Earth

    # If Numbin 6 and no spending --> Mars
    # If Deck F and no spending --> Mars
    # If CryoSleep True and Numbin 6 --> Mars
    # If CryoSleep True and Deck F --> Mars
    home_na = df['HomePlanet'].isna()
    for i in range(len(home_na)):
        if home_na[i]:
            if df.loc[i, 'Deck'] in ['A', 'B', 'C'] or ((df.loc[i, 'Destination'] == '55 Cancri e') and
                                                        (float(df.loc[i, 'Num']) <= 300)):
                df.loc[i, 'HomePlanet'] = 'Europa'
            elif df.loc[i, 'Deck'] in ['G'] or df.loc[i, 'GroupSize'] >= 6 or \
                    df.loc[i, 'Destination'] == 'PSO J318.5-22' or \
                    (df.loc[i, 'Age'] <= 18 and float(df.loc[i, 'Num']) >= 600):
                df.loc[i, 'HomePlanet'] = 'Earth'
            elif (float(df.loc[i, 'Num']) >= 1500 or df.loc[i, 'Deck'] in ['F']) and \
                    (df.loc[i, list_spend].sum() == 0 or df.loc[i, 'CryoSleep']):
                df.loc[i, 'HomePlanet'] = 'Mars'
            else:
                df.loc[i, 'HomePlanet'] = df['HomePlanet'].value_counts().index[0]
    return df
