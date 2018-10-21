from sklearn.model_selection import KFold

def stack_pred(estimator, X, y, Xt, k, method):
    """
        estimator - sklearn classifier or regressor
        method - 'predict' or 'predict_proba'
        X, y, Xt - numpy.array
        k - number of folds
        
        return sX, sXt - numpy.array
    """
    if method == 'predict':
        sX = np.zeros(X.shape[0])
        sXt = np.zeros(Xt.shape[0])
    elif method == 'predict_proba':
        sX = np.zeros((X.shape[0], np.unique(y).shape[0]))
        sXt = np.zeros((Xt.shape[0], np.unique(y).shape[0]))
    
    if estimator._estimator_type == 'classifier':
        # classifier
        kf = sklearn.model_selection.KFold(n_splits=k, shuffle=True, random_state=0)
        
        for train_index, test_index in kf.split(X, y):
            
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            estimator.fit(X_train, y_train)
            
            if method == 'predict':
                sX[test_index] = estimator.predict(X_test)
                sXt += estimator.predict(Xt)
            elif method == 'predict_proba':
                pred = estimator.predict_proba(X_test)                
                sX[test_index] = pred
                sXt += estimator.predict_proba(Xt)
        
    if estimator._estimator_type == 'regressor':
        # regressor
        kf = sklearn.model_selection.KFold(n_splits=k, shuffle=True, random_state=0)
        
        for train_index, test_index in kf.split(X, y):
            
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            estimator.fit(X_train, y_train)
            
            if method == 'predict':
                sX[test_index] = estimator.predict(X_test)
                sXt += estimator.predict(Xt)
            elif method == 'predict_proba':
                pred = estimator.predict_proba(X_test)
                sX[test_index] = pred
                sXt += estimator.predict_proba(Xt)
            
    sXt = sXt / k
    return sX, sXt
