import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


def make_predictions(input_model, x_sub, x_train, x_test, y_train, y_test, columns):

    if 'gradient' in input_model.lower():
        model = GradientBoostingClassifier(random_state=0, learning_rate=0.1, n_estimators=50, max_depth=5)
        pipe = Pipeline([('estimator', model)])

    elif 'random' in input_model.lower() or 'forest' in input_model.lower():
        model = RandomForestClassifier(random_state=0, max_features=90, n_estimators=100, max_depth=10)
        pipe = Pipeline([('estimator', model)])

    elif 'logreg' in input_model.lower() or 'logistic' in input_model.lower() or 'regression' in input_model.lower():
        model = LogisticRegression(random_state=0, C=2.5, penalty='l1', solver='saga')
        scale = ColumnTransformer(transformers=[('scaling', StandardScaler(), columns)], remainder='passthrough')
        pipe = Pipeline([('scaling', scale), ('estimator', model)])

    elif 'mlp' in input_model.lower():
        model = MLPClassifier(random_state=0, alpha=0.5, hidden_layer_sizes=128, activation='relu')
        scale = ColumnTransformer(transformers=[('scaling', StandardScaler(), columns)], remainder='passthrough')
        pipe = Pipeline([('scaling', scale), ('estimator', model)])

    elif 'svc' in input_model.lower():
        model = SVC(random_state=0, probability=True, C=50, gamma=0.005)
        scale = ColumnTransformer(transformers=[('scaling', StandardScaler(), columns)], remainder='passthrough')
        pipe = Pipeline([('scaling', scale), ('estimator', model)])

    else:
        print('ERROR: NO MODEL DETECTED TO ENSEMBLE')

    grid_search = GridSearchCV(pipe, {}, cv=5, scoring='accuracy')
    grid_search.fit(x_train, y_train)
    cv_acc = grid_search.best_score_
    print("{} CROSS VALIDATION SCORE: {:.4f}\n".format(input_model.upper(), cv_acc))
    print('{} TEST SCORE: {:.4f}\n'.format(input_model.upper(), grid_search.score(x_test, y_test)))
    test_preds = grid_search.predict_proba(x_test)[:, 1]
    sub_preds = grid_search.predict_proba(x_sub)[:, 1]

    return test_preds, sub_preds, cv_acc


def ensemble_predictions(pass_id, y_test_preds, y_sub_preds, acc, y_test):

    y_test = np.array(y_test)
    # Optimal weight and threshold calculation according to cross validation performance
    opt_weights1 = acc - min(acc) * 0.999
    opt_weights1 /= sum(opt_weights1)
    opt_th1, opt_acc1 = optimal_threshold(y_test, y_test_preds, opt_weights1)
    print('\nOPTIMAL WEIGHTS (distribution based on cross validation performance): {}'.format(opt_weights1))
    print('OPTIMAL THRESHOLD: {:.4f}'.format(opt_th1))
    print('ENSEMBLE TEST SCORE WITH OPTIMAL WEIGHTS AND DECISION THRESHOLD: {:.4f}'.format(opt_acc1))

    # Optimal weight and threshold calculation uniform distribution among models
    opt_weights2 = [1 / len(acc)] * 3
    opt_th2, opt_acc2 = optimal_threshold(y_test, y_test_preds, opt_weights2)
    print('\nOPTIMAL WEIGHTS (uniform distribution): {}'.format(opt_weights2))
    print('OPTIMAL THRESHOLD: {:.4f}'.format(opt_th2))
    print('ENSEMBLE TEST SCORE WITH OPTIMAL WEIGHTS AND DECISION THRESHOLD: {:.4f}'.format(opt_acc2))

    # Optimal weight and threshold calculation with customized weights
    opt_weights3 = [0.3, 0.5, 0.2]
    opt_th3 = 0.5
    opt_acc3 = accuracy_score(y_test, y_test_preds, opt_weights3, opt_th3)
    print('\nOPTIMAL WEIGHTS (customized): {}'.format(opt_weights3))
    print('OPTIMAL THRESHOLD: {:.4f}'.format(opt_th3))
    print('ENSEMBLE TEST SCORE WITH OPTIMAL WEIGHTS AND DECISION THRESHOLD: {:.4f}'.format(opt_acc3))

    # Create submission
    y_sub1 = np.where(np.dot(y_sub_preds, opt_weights1) >= opt_th1, True, False)
    y_sub2 = np.where(np.dot(y_sub_preds, opt_weights2) >= opt_th2, True, False)
    y_sub3 = np.where(np.dot(y_sub_preds, opt_weights3) >= opt_th3, True, False)
    df_submission = pass_id.to_frame()
    df_submission['Transported'] = y_sub1
    df_submission.to_csv('Submission1.csv', index=False)
    df_submission['Transported'] = y_sub2
    df_submission.to_csv('Submission2.csv', index=False)
    df_submission['Transported'] = y_sub3
    df_submission.to_csv('Submission3.csv', index=False)


def optimal_threshold(y_test, y_test_preds, opt_weights):
    # Optimal threshold calculation
    max_acc = 0
    max_th = 0
    threshold = np.linspace(0.25, 0.75, 100)
    for th in threshold:
        acc = accuracy_score(y_test, y_test_preds, opt_weights, th)
        if acc >= max_acc:
            max_acc = acc
            max_th = th
    return max_th, max_acc


def accuracy_score(y_true, y_preds, weights, threshold):
    ok = 0
    for i in range(len(y_true)):
        y_pred = np.dot(y_preds[i], weights)
        if y_pred >= threshold:
            if y_true[i] == 1:
                ok += 1
        else:
            if y_true[i] == 0:
                ok += 1
    return ok / len(y_true)
