import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize


def ensemble_models(pass_id, x_sub, x_train, x_val, x_test, y_train, y_val, y_test, columns, loops):
    preprocess = ColumnTransformer(transformers=[('scaling', StandardScaler(), columns)], remainder='passthrough')
    x_train_scaled = preprocess.fit_transform(x_train)
    x_val_scaled = preprocess.transform(x_val)
    x_test_scaled = preprocess.transform(x_test)
    x_sub_scaled = preprocess.transform(x_sub)

    model = KNeighborsClassifier(n_neighbors=15)
    model.fit(x_train_scaled, y_train)
    print('KNN TRAIN SCORE: {:.4f}'.format(model.score(x_train_scaled, y_train)))
    print('KNN VALIDATION SCORE: {:.4f}'.format(model.score(x_val_scaled, y_val)))
    print('KNN TEST SCORE: {:.4f}\n'.format(model.score(x_test_scaled, y_test)))
    knn_train_preds = model.predict_proba(x_train_scaled)[:, 1].reshape(-1, 1)
    knn_val_preds = model.predict_proba(x_val_scaled)[:, 1].reshape(-1, 1)
    knn_test_preds = model.predict_proba(x_test_scaled)[:, 1].reshape(-1, 1)
    knn_sub_preds = model.predict_proba(x_sub_scaled)[:, 1].reshape(-1, 1)

    model = RandomForestClassifier(random_state=0, max_features=80, n_estimators=200, max_depth=10)
    model.fit(x_train, y_train)
    print('RANDOM FOREST TRAIN SCORE: {:.4f}'.format(model.score(x_train, y_train)))
    print('RANDOM FOREST VALIDATION SCORE: {:.4f}'.format(model.score(x_val, y_val)))
    print('RANDOM FOREST TEST SCORE: {:.4f}\n'.format(model.score(x_test, y_test)))
    forest_train_preds = model.predict_proba(x_train)[:, 1].reshape(-1, 1)
    forest_val_preds = model.predict_proba(x_val)[:, 1].reshape(-1, 1)
    forest_test_preds = model.predict_proba(x_test)[:, 1].reshape(-1, 1)
    forest_sub_preds = model.predict_proba(x_sub)[:, 1].reshape(-1, 1)

    model = GradientBoostingClassifier(random_state=0, learning_rate=0.1, n_estimators=50, max_depth=5)
    model.fit(x_train, y_train)
    print('GRADIENT BOOSTING TRAIN SCORE: {:.4f}'.format(model.score(x_train, y_train)))
    print('GRADIENT BOOSTING VALIDATION SCORE: {:.4f}'.format(model.score(x_val, y_val)))
    print('GRADIENT BOOSTING TEST SCORE: {:.4f}\n'.format(model.score(x_test, y_test)))
    gradient_train_preds = model.predict_proba(x_train)[:, 1].reshape(-1, 1)
    gradient_val_preds = model.predict_proba(x_val)[:, 1].reshape(-1, 1)
    gradient_test_preds = model.predict_proba(x_test)[:, 1].reshape(-1, 1)
    gradient_sub_preds = model.predict_proba(x_sub)[:, 1].reshape(-1, 1)

    model = LinearSVC(random_state=0, dual=False, C=0.1)
    model = CalibratedClassifierCV(model)
    model.fit(x_train_scaled, y_train)
    print('LINEARSVC MODEL TRAIN SCORE: {:.4f}'.format(model.score(x_train_scaled, y_train)))
    print('LINEARSVC MODEL VALIDATION SCORE: {:.4f}'.format(model.score(x_val_scaled, y_val)))
    print('LINEARSVC MODEL TEST SCORE: {:.4f}\n'.format(model.score(x_test_scaled, y_test)))
    linearsvc_train_preds = model.predict_proba(x_train_scaled)[:, 1].reshape(-1, 1)
    linearsvc_val_preds = model.predict_proba(x_val_scaled)[:, 1].reshape(-1, 1)
    linearsvc_test_preds = model.predict_proba(x_test_scaled)[:, 1].reshape(-1, 1)
    linearsvc_sub_preds = model.predict_proba(x_sub_scaled)[:, 1].reshape(-1, 1)

    model = LogisticRegression(random_state=0, C=0.5)
    model.fit(x_train_scaled, y_train)
    print('LOGISTIC REGRESSION MODEL TRAIN SCORE: {:.4f}'.format(model.score(x_train_scaled, y_train)))
    print('LOGISTIC REGRESSION MODEL VALIDATION SCORE: {:.4f}'.format(model.score(x_val_scaled, y_val)))
    print('LOGISTIC REGRESSION TEST SCORE: {:.4f}\n'.format(model.score(x_test_scaled, y_test)))
    logreg_train_preds = model.predict_proba(x_train_scaled)[:, 1].reshape(-1, 1)
    logreg_val_preds = model.predict_proba(x_val_scaled)[:, 1].reshape(-1, 1)
    logreg_test_preds = model.predict_proba(x_test_scaled)[:, 1].reshape(-1, 1)
    logreg_sub_preds = model.predict_proba(x_sub_scaled)[:, 1].reshape(-1, 1)

    model = MLPClassifier(random_state=0, alpha=0.5, hidden_layer_sizes=128, activation='relu')
    model.fit(x_train_scaled, y_train)
    print('MLP MODEL TRAIN SCORE: {:.4f}'.format(model.score(x_train_scaled, y_train)))
    print('MLP MODEL VALIDATION SCORE: {:.4f}'.format(model.score(x_val_scaled, y_val)))
    print('MLP MODEL TEST SCORE: {:.4f}\n'.format(model.score(x_test_scaled, y_test)))
    mlp_train_preds = model.predict_proba(x_train_scaled)[:, 1].reshape(-1, 1)
    mlp_val_preds = model.predict_proba(x_val_scaled)[:, 1].reshape(-1, 1)
    mlp_test_preds = model.predict_proba(x_test_scaled)[:, 1].reshape(-1, 1)
    mlp_sub_preds = model.predict_proba(x_sub_scaled)[:, 1].reshape(-1, 1)

    model = SVC(random_state=0, C=50, gamma=0.005)
    model = CalibratedClassifierCV(model)
    model.fit(x_train_scaled, y_train)
    print('SVC MODEL TRAIN SCORE: {:.4f}'.format(model.score(x_train_scaled, y_train)))
    print('SVC MODEL VALIDATION SCORE: {:.4f}'.format(model.score(x_val_scaled, y_val)))
    print('SVC MODEL TEST SCORE: {:.4f}\n'.format(model.score(x_test_scaled, y_test)))
    svc_train_preds = model.predict_proba(x_train_scaled)[:, 1].reshape(-1, 1)
    svc_val_preds = model.predict_proba(x_val_scaled)[:, 1].reshape(-1, 1)
    svc_test_preds = model.predict_proba(x_test_scaled)[:, 1].reshape(-1, 1)
    svc_sub_preds = model.predict_proba(x_sub_scaled)[:, 1].reshape(-1, 1)

    y_train = np.array(y_train)
    y_test = np.array(y_test)
    y_val = np.array(y_val)

    y_train_preds = np.c_[gradient_train_preds, logreg_train_preds, linearsvc_train_preds, svc_train_preds, mlp_train_preds]
    y_val_preds = np.c_[gradient_val_preds, logreg_val_preds, linearsvc_val_preds, svc_val_preds, mlp_val_preds]
    y_test_preds = np.c_[gradient_test_preds, logreg_test_preds, linearsvc_test_preds, svc_test_preds, mlp_test_preds]
    y_sub_preds = np.c_[gradient_sub_preds, logreg_sub_preds, linearsvc_sub_preds, svc_sub_preds, mlp_sub_preds]
    opt_weights, acc_val = calculate_optimal_weights(y_val, y_val_preds, loops)
    acc_test = accuracy_score(opt_weights, y_test, y_test_preds)
    acc_train = accuracy_score(opt_weights, y_train, y_train_preds)

    print('\nENSEMBLE TRAIN SCORE: {:.4f}'.format(acc_train))
    print('ENSEMBLE VALIDATION SCORE: {:.4f}'.format(acc_val))
    print('ENSEMBLE TEST SCORE: {:.4f}'.format(acc_test))
    print('OPTIMAL WEIGHTS: {}'.format(opt_weights))

    y_sub = np.round(np.dot(y_sub_preds, opt_weights))
    df_submission = pass_id.to_frame()
    df_submission['Transported'] = y_sub
    df_submission.replace({1: True, 0: False}, inplace=True)
    df_submission.to_csv('Submission.csv', index=False)


def accuracy_score(weights, y_true, y_preds):
    ok = 0
    for i in range(len(y_true)):
        y_pred = np.round(np.dot(y_preds[i], weights))
        if y_pred == y_true[i]:
            ok += 1
    return ok / len(y_true)


def minimize_acc(weights, y_true, y_preds):
    """ Calculate the score of a weighted model predictions"""
    return -accuracy_score(weights, y_true, y_preds)


def calculate_optimal_weights(y_true, y_preds, sims):
    acc_opt = 100
    acc_weights_opt = 0
    for i in range(sims):
        weights_ini = np.random.rand(y_preds.shape[1])
        weights_ini /= sum(weights_ini)
        acc_minimizer = minimize(fun=minimize_acc,
                                 x0=weights_ini,
                                 method='trust-constr',
                                 args=(y_true, y_preds),
                                 bounds=[(0, 1)] * y_preds.shape[1],
                                 options={'disp': True, 'maxiter': 10000},
                                 constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1})
        if acc_minimizer.fun < acc_opt:
            acc_opt = acc_minimizer.fun
            acc_weights_opt = acc_minimizer.x
    return acc_weights_opt, -acc_opt
