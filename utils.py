import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import plot_confusion_matrix, accuracy_score, recall_score, precision_score, f1_score


def import_and_decode(file_name):
    
    """Imports sas files and converts columns of type 'bytes' to 'utf-8'.
    
    Parameters
    ----------
    file_name : String.  File path and name with .xpt extension (sas file).
    
    Returns
    -------
    DataFrame"""
    
    df = pd.read_sas(file_name)
    
    for col in df:
        if df[col].dtype == 'object':
            df[col] = df[col].map(lambda x: x.decode("utf-8"))
            
    return df


def replace_with_median(col, value_to_replace):
    
    """Replaces a dummy number with the median of the other numbers in the column.
    
    Parameters
    ----------
    
    col : Pandas DataFrame column with numeric values
    
    value_to_replace : Dummy value that needs to be replaced
    
    Returns
    -------
    
    DataFrame column with dummy values now replaced with median."""
    
    real_values = col.loc[(~col.isna()) & (col != value_to_replace)]

    true_median = real_values.median()

    return col.replace(value_to_replace, true_median)


# import data and create 'age' df for use in get_years function
demo_j = import_and_decode('data/demo_j.xpt') # demographics
age = demo_j[['SEQN', 'RIDAGEYR']] # create age df


def get_years(df, age_diagnosed_col, new_col_name):
    
    # create new dataframe that includes participant age column
    new_df = df.merge(age, how='left', on='SEQN')
    
    # create new column showing how many years participant had the condition
    # if age at diagnosis = 80, impute 1 year, since both age at diagnosis and 
    # participant age are top-coded at 80 in NHANES
    new_df[new_col_name] = np.where(new_df[age_diagnosed_col] == 80, 1, 
                                    new_df['RIDAGEYR'] - new_df[age_diagnosed_col])

    # some values of 'age at diagnosis' may be substituted with median if unknown
    # if participant age minus median is negative, substitute participant age for years with condition
    new_df.loc[new_df[new_col_name] < 0, new_col_name] = new_df['RIDAGEYR']
    
    # clean up new df by dropping columns no longer needed
    new_df.drop(columns = [age_diagnosed_col, 'RIDAGEYR'], inplace=True)
    return new_df


def make_binary(df, cols):
    binary_df = df.copy()
    binary_df[cols] = binary_df[cols].applymap(lambda x: 1 if x > 0 else 0)
    for col in cols:
        binary_df.rename(columns = {col: col[4:] + '_binary'}, inplace=True)
    return binary_df


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