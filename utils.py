import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import plot_confusion_matrix, accuracy_score, recall_score, precision_score, f1_score



def no_bytes(df):
    
    for col in df:
        if df[col].dtype == 'object':
            df[col] = df[col].map(lambda x: x.decode("utf-8"))
            
    return df 


def import_and_decode(file_name):
    
    df = pd.read_sas(file_name)
    
    for col in df:
        if df[col].dtype == 'object':
            df[col] = df[col].map(lambda x: x.decode("utf-8"))
            
    return df 


def k_fold_validator(X, y, classifier, cv=5):

    """Uses k-fold cross-validation to calculate the mean recall, precision, and f1 scores 
    for train and test sets for a model.  Also plots a confusion matrix for each test set.
    
    Parameters
    ----------
    
    X : DataFrame, Predictors
    
    y : series, Labels assigned
    
    classifier : An instance of a classifier.
    
    cv : int, How many folds to use when cross-validating.  Default = 5.
    
    Returns
    -------
    
    No objects returned.
    
    Prints mean recall, precision, and f1 scores for train and test sets.
    
    Plots a confusion matrix for each test set."""
    
    kf = KFold(n_splits=cv)
    clf = classifier

    train_recall_scores = []
    train_precision_scores = []
    train_f1_scores = []
    test_recall_scores = []
    test_precision_scores = []
    test_f1_scores = []
    
    print('Classifier:', clf)
    print('Cross-validation folds:', cv)
    
    for train_index, test_index in kf.split(X):

        X_tr, X_test = X.iloc[train_index].astype(str), X.iloc[test_index].astype(str)
        y_tr, y_test = y.iloc[train_index].astype(str), y.iloc[test_index].astype(str)
        
        clf.fit(X_tr, y_tr)

        y_pred_tr = clf.predict(X_tr)
        y_pred_test = clf.predict(X_test)

        train_recall_scores.append(recall_score(y_tr, y_pred_tr, pos_label='1.0'))
        train_precision_scores.append(precision_score(y_tr, y_pred_tr, pos_label='1.0'))
        train_f1_scores.append(f1_score(y_tr, y_pred_tr, pos_label='1.0'))       
        test_recall_scores.append(recall_score(y_test, y_pred_test, pos_label='1.0'))
        test_precision_scores.append(precision_score(y_test, y_pred_test, pos_label='1.0'))
        test_f1_scores.append(f1_score(y_test, y_pred_test, pos_label='1.0'))       
        
        plot_confusion_matrix(clf, X_test, y_test)
        plt.title('Test set')
        
    print('\n')
    
    print('Train mean recall: {} +/- {}'.format(round(pd.Series(train_recall_scores).mean(), 2), 
                                               round(pd.Series(train_recall_scores).std(), 2)))
    
    print('Train mean precision: {} +/- {}'.format(round(pd.Series(train_precision_scores).mean(), 2),
                                                  round(pd.Series(train_precision_scores).std(), 2)))
    
    print('Train mean F1: {} +/- {}'.format(round(pd.Series(train_f1_scores).mean(), 2),
                                           round(pd.Series(train_f1_scores).std(), 2)))
    print('\n')
    
    print('Test mean recall: {} +/- {}'.format(round(pd.Series(test_recall_scores).mean(), 2),
                                               round(pd.Series(test_recall_scores).std(), 2)))
    
    print('Test mean precision: {} +/- {}'.format(round(pd.Series(test_precision_scores).mean(), 2),
                                                  round(pd.Series(test_precision_scores).std(), 2)))
    
    print('Test mean F1: {} +/- {}'.format(round(pd.Series(test_f1_scores).mean(), 2),
                                           round(pd.Series(test_f1_scores).std(), 2)))