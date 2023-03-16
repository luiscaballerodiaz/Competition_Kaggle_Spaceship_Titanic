import pandas as pd
from data_visualization import DataPlot
import utils


visualization = DataPlot()
pd.set_option('display.max_columns', None)
df = pd.read_csv('train.csv')
column_target = 'Transported'
feat_cat = ['HomePlanet', 'CryoSleep', 'Deck', 'Num', 'Side', 'Destination', 'VIP']
feat_num = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

df = utils.data_analytics(visualization, df, column_target, feat_cat, feat_num)
x_train, x_test, y_train, y_test = utils.data_preprocessing(df, column_target, feat_cat, feat_num)
utils.data_modeling(x_train, x_test, y_train, y_test)
