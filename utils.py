# import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score
from sklearn.metrics import confusion_matrix, plot_confusion_matrix

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.naive_bayes import ComplementNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

# define functions

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


def order_features(weights, X_train):
    
    """Helper function to put model coefficients in order according to the absolute value of their weights.
    
    Parameters
    ----------
    weights : ndarray of shape (1, n_features), for example the 'coef_' attribute of sklearn models.
    
    X: DataFrame of predictors used to train the model.
    
    Returns
    -------
    DataFrame of coefficients ordered from greatest to least weight"""
    
    coef_dict = {}

    for n, c in enumerate(X_train.columns):
        coef_dict[c]=round(weights[0][n],4)

    sorted_coef_dict = {k: v for k, v in sorted(coef_dict.items(), key=lambda item: item[1], reverse=True)}
    df = pd.DataFrame.from_dict(sorted_coef_dict, orient='index', columns=['weight'])
    df['abs_weight']=np.abs(df['weight'])
    weights_df = df.sort_values(by = 'abs_weight', ascending=False)
    
    return weights_df


def order_features_tree(weights, X_train):
    
    """Helper function to put model coefficients in order according to the absolute value of their weights.
    
    Parameters
    ----------
    weights : nndarray of shape (n_features,), for example the 'feature_importances_' attribute of tree-based sklearn models.
    
    X: DataFrame of predictors used to train the model.
    
    Returns
    -------
    DataFrame of coefficients ordered from greatest to least weight"""
    
    coef_dict = {}

    for n, c in enumerate(X_train.columns):
        coef_dict[c]=round(weights[n],4)

    sorted_coef_dict = {k: v for k, v in sorted(coef_dict.items(), key=lambda item: item[1], reverse=True)}
    df = pd.DataFrame.from_dict(sorted_coef_dict, orient='index', columns=['weight'])
    df['abs_weight']=np.abs(df['weight'])
    weights_df = df.sort_values(by = 'abs_weight', ascending=False)
    
    return weights_df


def k_fold_validator(X, y, classifier, cv=5):

    """Uses k-fold cross-validation to calculate the mean recall, precision, and f1 scores 
    for train and test sets for a model.  Also prints the weights of the model coefficients 
    and plots a confusion matrix for each test set.
    
    Parameters
    ----------
    
    X : DataFrame, Predictors
    
    y : series, Labels assigned
    
    classifier : An instance of a classifier.
    
    cv : int, How many folds to use when cross-validating.  Default = 5.
    
    Returns
    -------
    
    No objects returned.
    
    Prints mean recall and precision scores for train and test sets.
    Prints a list of the model coefficients and their weights.
    Plots a confusion matrix for each test set."""
    
    scaler = MinMaxScaler()

    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
    
    kf = KFold(n_splits=cv, random_state=807, shuffle=True)
    clf = classifier

    train_recall_scores = []
    train_precision_scores = []
    train_roc_auc_scores = []
    test_recall_scores = []
    test_precision_scores = []
    test_roc_auc_scores = []
    
    print('Classifier:', clf)
    print('Cross-validation folds:', cv)
    
    
    for train_index, test_index in kf.split(X_scaled):

        X_tr, X_test = X_scaled.iloc[train_index], X_scaled.iloc[test_index]
        y_tr, y_test = y.iloc[train_index], y.iloc[test_index]
        
        clf.fit(X_tr, y_tr)

        y_pred_tr = clf.predict(X_tr)
        y_pred_test = clf.predict(X_test)

        train_recall_scores.append(recall_score(y_tr, y_pred_tr, pos_label=1.0))
        train_precision_scores.append(precision_score(y_tr, y_pred_tr, pos_label=1.0))
        train_roc_auc_scores.append(roc_auc_score(y_tr, y_pred_tr))
        
        test_recall_scores.append(recall_score(y_test, y_pred_test, pos_label=1.0))
        test_precision_scores.append(precision_score(y_test, y_pred_test, pos_label=1.0))
        test_roc_auc_scores.append(roc_auc_score(y_test, y_pred_test))
        

        plot_confusion_matrix(clf, X_test, y_test)
        plt.title('Error Matrix - Test Set', fontsize=18, pad=15)
        plt.xticks(ticks=(0,1), labels=['Not \nHospitalized', 'Hospitalized'], fontsize=12)
        plt.yticks(ticks=(0,1), labels=['Not \nHospitalized', 'Hospitalized'], fontsize=12)
        plt.xlabel('Predicted Label', labelpad=15)
        plt.ylabel('True Label', labelpad=15)
        
    print('\n')
    
    print('Train mean recall: {} +/- {}'.format(round(pd.Series(train_recall_scores).mean(), 2), 
                                               round(pd.Series(train_recall_scores).std(), 2)))
    
    print('Train mean precision: {} +/- {}'.format(round(pd.Series(train_precision_scores).mean(), 2),
                                                  round(pd.Series(train_precision_scores).std(), 2)))
    
    print('Train mean ROC-AUC: {} +/- {}'.format(round(pd.Series(train_roc_auc_scores).mean(), 2),
                                                  round(pd.Series(train_roc_auc_scores).std(), 2)))   
    print('\n')
    
    print('Test mean recall: {} +/- {}'.format(round(pd.Series(test_recall_scores).mean(), 2),
                                               round(pd.Series(test_recall_scores).std(), 2)))
    
    print('Test mean precision: {} +/- {}'.format(round(pd.Series(test_precision_scores).mean(), 2),
                                                  round(pd.Series(test_precision_scores).std(), 2)))
    
    print('Test mean ROC-AUC: {} +/- {}'.format(round(pd.Series(test_roc_auc_scores).mean(), 2),
                                                  round(pd.Series(test_roc_auc_scores).std(), 2)))  
    
    print('\n')
    
    if type(clf) == DecisionTreeClassifier:
        features = order_features_tree(clf.feature_importances_, X_scaled)
    elif type(clf) == RandomForestClassifier:
        features = order_features_tree(clf.feature_importances_, X_scaled)
    elif type(clf) == AdaBoostClassifier:
        features = order_features_tree(clf.feature_importances_, X_scaled)
    elif type(clf) == GradientBoostingClassifier:
        features = order_features_tree(clf.feature_importances_, X_scaled)   
    elif type(clf) == KNeighborsClassifier:
        pass
    elif type(clf) == XGBClassifier:
        pass
    else:
        features = order_features(clf.coef_, X_scaled)
    
    if (type(clf) != KNeighborsClassifier) and (type(clf) != XGBClassifier) and (type(clf) != AdaBoostClassifier):
         if (type(clf) != GradientBoostingClassifier):
            print('Feature weights:', '\n', features, '\n')
            print('Confusion matrices for each fold test set:', '\n')
    


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
    
    """Replaces a specified column in a DataFrame, showing the age a participant was diagnosed with a condition,
    with a new column showing the number of years the participant has had the condition.
    
    Parameters
    ----------
    
    df : Pandas DataFrame
    
    age_diagnosed_col : Column in the DataFrame that needs to be replaced, 
    showing participant age at diagnosis
    
    new_col_name : string, name chosen for the new column
    
    Returns
    -------
    
    DataFrame with age diagnosed column replaced by new years with condition column."""
    
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
    
    """Converts continuous columns to binary columns.
    
    Parameters
    ----------
    
    df : Pandas DataFrame
    
    cols : List of columns that need to be converted in the DataFrame
    
    Returns
    -------
    
    DataFrame with continuous columns replaced by binary columns"""
    
    # copy the original df
    binary_df = df.copy()
    
    # map 1 if the value is greater than 0, otherwise 0 to each column in the list
    binary_df[cols] = binary_df[cols].applymap(lambda x: 1 if x > 0 else 0)
    
    # rename the columns to include 'binary' instead of 'yrs'
    for col in cols:
        binary_df.rename(columns = {col: col[4:] + '_binary'}, inplace=True)
        
    # return new DataFrame
    return binary_df


