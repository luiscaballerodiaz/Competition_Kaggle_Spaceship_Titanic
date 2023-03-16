import pandas as pd
from data_visualization import DataPlot
import utils


visualization = DataPlot()  # Instantiate an object for DataPlot to manage plots in this exercise
pd.set_option('display.max_columns', None)  # Enable option to display all dataframe columns
df = pd.read_csv('train.csv')  # Read CSV file

column_target = 'Transported'  # Name for the target feature
feat_cat = ['HomePlanet', 'CryoSleep', 'Deck', 'Num', 'Side', 'Destination', 'VIP']  # List for the categorical features
feat_num = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']  # List for the numerical features

df = utils.data_analytics(visualization, df, column_target, feat_cat, feat_num)  # Scrub data and generate plots
x_train, x_test, y_train, y_test = utils.split_scaling(df, column_target, feat_cat, feat_num)  # Data split and scaling
utils.modeling(x_train, x_test, y_train, y_test)  # Create model and make predictions
