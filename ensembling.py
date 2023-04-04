import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV


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

    elif 'linearsvc' in input_model.lower():
        svc = LinearSVC(random_state=0, dual=False, C=0.1, penalty='l1')
        model = CalibratedClassifierCV(svc)
        scale = ColumnTransformer(transformers=[('scaling', StandardScaler(), columns)], remainder='passthrough')
        pipe = Pipeline([('scaling', scale), ('estimator', model)])

    elif 'svm' in input_model.lower():
        model = SVC(random_state=0, probability=True, C=50, gamma=0.005)
        scale = ColumnTransformer(transformers=[('scaling', StandardScaler(), columns)], remainder='passthrough')
        pipe = Pipeline([('scaling', scale), ('estimator', model)])

    else:
        print('ERROR: NO MODEL DETECTED TO ENSEMBLE')

    grid_search = GridSearchCV(pipe, {}, cv=5, scoring='accuracy')
    grid_search.fit(x_train, y_train)
    cv_acc = grid_search.best_score_
    print("{} CROSS VALIDATION SCORE: {:.4f}\n".format(input_model.upper(), cv_acc))
    print('{} TRAIN SCORE: {:.4f}\n'.format(input_model.upper(), grid_search.score(x_train, y_train)))
    print('{} TEST SCORE: {:.4f}\n'.format(input_model.upper(), grid_search.score(x_test, y_test)))
    test_preds = grid_search.predict_proba(x_test)[:, 1]
    sub_preds = grid_search.predict_proba(x_sub)[:, 1]

    return test_preds, sub_preds


def ensemble_predictions(pass_id, weights, th, y_preds, y_sub_preds, y_true):

    y_true = np.array(y_true)
    # Optimal weight and threshold calculation with customized weights
    acc = accuracy_score(y_true, y_preds, weights, th)
    print('\nOPTIMAL WEIGHTS (customized): {}'.format(weights))
    print('OPTIMAL THRESHOLD: {:.4f}'.format(th))
    print('ENSEMBLE TEST SCORE WITH OPTIMAL WEIGHTS AND DECISION THRESHOLD: {:.4f}'.format(acc))

    # Create submission
    y_sub = np.where(np.dot(y_sub_preds, weights) >= th, True, False)
    df_submission = pass_id.to_frame()
    df_submission['Transported'] = y_sub
    df_submission.to_csv('Submission.csv', index=False)


def optimal_threshold(y_true, y_preds, opt_weights):
    # Optimal threshold calculation
    max_acc = 0
    max_th = 0
    threshold = np.linspace(0.25, 0.75, 100)
    for th in threshold:
        acc = accuracy_score(y_true, y_preds, opt_weights, th)
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
