import pandas as pd
from data_visualization import DataPlot
from gridsearch_postprocess import GridSearchPostProcess
import ensembling
import data_processing
import utils


sim = 1
# sim = 0 --> Individual model grid search cross validation and generate submission for best model
# sim = 1 --> Ensemble models and generate submission
# sim = 2 --> Grid search cross validation loop using as input a different new combined feature
plots = 0
# plots = 0 --> No plots
# plots = 1 --> Plots created

params = [
    # {'preprocess': [''], 'scaling': ['std'], 'estimator': ['knn'], 'estimator__n_neighbors': [1, 3, 5, 10, 15, 25]},
    {'preprocess': [''], 'scaling': ['std'], 'estimator': ['logreg'], 'estimator__C': [0.5]}]
    # {'preprocess': [''], 'scaling': ['std'], 'estimator': ['logreg'],
    #  'estimator__C': [0.001, 0.01, 0.1, 0.5, 1, 2.5, 5, 7.5, 10, 25, 100, 250, 500, 1000]},
    # {'preprocess': [''], 'scaling': ['std'], 'estimator': ['linearsvc'],
    #  'estimator__C': [0.001, 0.01, 0.1, 0.5, 1, 2.5, 5, 7.5, 10, 25, 100, 250, 500, 1000]},
    # {'preprocess': [''], 'scaling': [''], 'estimator': ['tree'], 'estimator__max_depth': [3, 5, 7, 10, 15, 25, 50]},
    # {'preprocess': [''], 'scaling': [''], 'estimator': ['random forest'],
    #  'estimator__n_estimators': [50, 100, 150, 200], 'estimator__max_depth': [6, 8, 10, 12, 14],
    #  'estimator__max_features': [50, 60, 70, 80, 90, 100]},
    # {'preprocess': [''], 'scaling': [''], 'estimator': ['gradient boosting'],
    #  'estimator__n_estimators': [25, 35, 50, 65, 80, 100], 'estimator__max_depth': [3, 4, 5, 6],
    #  'estimator__learning_rate': [0.01, 0.05, 0.075, 0.1, 0.15, 0.2]},
    # {'preprocess': [''], 'scaling': ['std'], 'estimator': ['svm'],
    #  'estimator__gamma': [0.0005, 0.001, 0.005, 0.01, 0.05],
    #  'estimator__C': [0.1, 1, 10, 50, 100, 500]},
    # {'preprocess': [''], 'scaling': ['std'], 'estimator': ['mlp'], 'estimator__alpha': [0.1, 0.5, 1, 2.5, 5],
    #  'estimator__activation': ['relu'], 'estimator__hidden_layer_sizes': [128, 256, [128, 64], [128, 64, 32]]}]


visualization = DataPlot()  # Instantiate an object for DataPlot to manage plots in this exercise
sweep = GridSearchPostProcess()  # Instantiate an object for GridSearchPostProcess to manage the grid search results
pd.set_option('display.max_columns', None)  # Enable option to display all dataframe columns
pd.set_option('display.max_colwidth', None)  # Enable printing the whole column content
df = pd.read_csv('train.csv')  # Read CSV file
df_submission = pd.read_csv('test.csv')  # Read CSV file
target = 'Transported'  # Target feature
fcat_ini = ['PassengerId', 'HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'VIP', 'Name']  # Categorical features
fnum_ini = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']  # Numerical features

if plots == 1:
    df_total = pd.concat([df, df_submission])
    data_processing.dataset_correlations(visualization, df_total, fcat_ini)

pass_id = df_submission['PassengerId']
# Apply feature engineering
df, feat_cat, feat_num, df_full, feat_cat_full, feat_num_full = data_processing.feature_engineering(
    visualization, df, fcat_ini, fnum_ini, sim, plots=plots, column_target=target)
df_sub, _, _, _, _, _ = data_processing.feature_engineering(visualization, df_submission, fcat_ini, fnum_ini, sim)
# One hot encoding and train, val and test split
x_sub, x_train, x_val, x_test, y_train, y_val, y_test, index_num = utils.onehot_split(
    df, target, feat_cat, feat_num, df_sub)
# Model simulation depending on sim parametrization
if sim == 0:
    pd_grid, score = utils.pipeline_gridsearch(x_train, x_val, x_test, y_train, y_val, y_test, index_num, params,
                                               'accuracy', 5, pass_id=pass_id, x_sub=x_sub)
    sweep.param_sweep_matrix(params=pd_grid['params'], test_score=pd_grid['mean_test_score'])
elif sim == 1:
    ensembling.ensemble_models(pass_id, x_sub, x_train, x_val, x_test, y_train, y_val, y_test, index_num, 250)
elif sim == 2:
    best_feats, best_scores = utils.feat_sweep_gridsearch(df_full, params, target, feat_cat_full, feat_num_full, 2)
elif sim == 3:
    from keras import models
    from keras import callbacks
    from keras import layers
    from keras import models
    from keras import optimizers
    from keras import regularizers
    import numpy as np

    model = models.Sequential()
    model.add(layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0), input_shape=(x_train.shape[1],)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(learning_rate=0.0001), metrics=['acc'])
    callbacks_list = callbacks.ModelCheckpoint('model.h5', monitor='val_acc', save_best_only=True,
                                               verbose=1, mode='max')
    history = model.fit(x_train, y_train, batch_size=20, callbacks=callbacks_list, epochs=500, validation_data=(x_val, y_val))

    model = models.load_model('model.h5')
    print(model.evaluate(x_test, y_test))
    y_sub = np.round(model.predict(x_sub, batch_size=20))
    df_submission = pass_id.to_frame()
    df_submission['Transported'] = y_sub
    df_submission.replace({1: True, 0: False}, inplace=True)
    df_submission.to_csv('Submission.csv', index=False)
