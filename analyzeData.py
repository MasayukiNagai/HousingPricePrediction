import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from main import preprocess, dropOutliers, transformData, standardize

train_file = './data/train.csv'
test_file = './data/test.csv'
trainDF = pd.read_csv(train_file)
testDF = pd.read_csv(test_file)

def plotHistOfSalePrice(df):
    salePrices = df.loc[:, 'SalePrice']
    print(salePrices.describe())
    # salePrices.hist(bins=50)
    sns.distplot(salePrices, fit=stats.norm)
    plt.show()


def createPairPlots(df):
    sns.set()
    sns.pairplot(df, height = 2.5)
    plt.show()


def checkMissingValues(df):
    # BEGIN: from from https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python
    # EXPLANATION: Caclulate the number and percentage of missing values and diplay
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    print(missing_data.loc[missing_data['Total'] !=0])
    # END: from from https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python


def checkNormality(df, col):
    # BEGIN: from from https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python
    # EXPLANATION: Displays histogram and normal probability plot
    sns.distplot(df[col], fit=stats.norm)
    fig = plt.figure()
    res = stats.probplot(df[col], plot=plt)
    plt.show()  # ADDED
    # END: from from https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python


def checkCorr(df, k=11):
    # BEGIN: from from https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python
    # EXPLANATION: Visualization of correlation between SalePrice and k highly correlated elements
    corrmat = df.corr().abs()  # ADDED (.abs())
    color = plt.get_cmap('RdPu')  # ADDED
        
    f, ax = plt.subplots(figsize=(12, 9))
    sns.set(font_scale=0.7)
    sns.heatmap(corrmat, cmap="RdPu", vmax=.8, square=True)
    plt.show()
    # END: from https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python


def checkHighCorr(df, k=16):
    # BEGIN: from from https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python
    # EXPLANATION: Visualization of correlation between SalePrice and k highly correlated elements
    f, ax = plt.subplots(figsize=(12, 9))
    corrmat = df.corr().abs()  # ADDED (.abs())
    color = plt.get_cmap('RdPu')  # ADDED
    cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
    cm = np.corrcoef(df[cols].values.T)
    sns.set(font_scale=1.0)
    hm = sns.heatmap(cm, cmap=color, cbar=True, annot=True, square=True, 
                     fmt='.2f', annot_kws={'size': 10}, vmin=0.2, vmax=0.8,
                     yticklabels=cols.values, xticklabels=cols.values)
    plt.show()
    # END: from https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python


def analysis1():
    '''
    Display correlations between SalePrice and other variables
    before and after preprocess
    '''
    trainDF = pd.read_csv(train_file)
    checkCorr(trainDF)
    # checkHighCorr(trainDF)
    preprocess(trainDF, trainDF)
    checkHighCorr(trainDF)


def analysis2():
    '''
    SalePrice is not normally distributed and has positive skewness
    Log transformation solves the issue
    '''
    trainDF = pd.read_csv(train_file)
    plotHistOfSalePrice(trainDF)
    checkNormality(trainDF, 'SalePrice')
    trainDF['SalePrice'] = np.log1p(trainDF['SalePrice'])
    plotHistOfSalePrice(trainDF)
    checkNormality(trainDF, 'SalePrice')


def analysis3():
    '''
    Display pair plots of variables that have n highest corr with SalePrice
    '''
    trainDF = pd.read_csv(train_file)
    preprocess(trainDF, trainDF)
    cols = ["SalePrice", "OverallQual", "GrLivArea", "ExterQual", "KitchenQual", "GarageCars", "BsmtQual", "1stFlrSF", "FullBath", "GarageFinish", "YearsOld"]
    createPairPlots(trainDF.loc[:, cols])


def analysis4():
    '''
    Check missing values in trainDF and testDF
    '''
    trainDF = pd.read_csv(train_file)
    testDF = pd.read_csv(test_file)
    all_data = pd.concat(objs=[trainDF, testDF], axis=0)
    print('========== BEFORE Preprocess ==========')
    print('----------- trainDF ----------')
    checkMissingValues(trainDF)
    print('----------- testDF ----------')
    checkMissingValues(testDF)
    print('----------- All data ----------')
    checkMissingValues(all_data)
    print('========== AFTER Preprocess ==========')
    preprocess(testDF, trainDF)
    preprocess(trainDF, trainDF)
    dropOutliers(trainDF)
    print('----------- trainDF ----------')
    checkMissingValues(trainDF)
    print('----------- testDF ----------')
    checkMissingValues(testDF)


def analysis5():
    trainDF = pd.read_csv(train_file)
    preprocess(trainDF, trainDF)
    print(stats.skew(trainDF['GrLivArea']))
    print(stats.skew(trainDF['AllSF']))
    # trainDF['GrLivArea'] = np.log1p(trainDF['GrLivArea'])
    # checkNormality(trainDF, 'GrLivArea')
    # trainDF['AllSF'] = np.log1p(trainDF['AllSF'])
    # checkNormality(trainDF, 'AllSF')


def analysis6():
    trainDF = pd.read_csv(train_file)
    predictors = ['OverallQual', 'AllSF', 'GarageCars', 'BsmtUnfSF', 'YearRemodAdd', 
                  'MSZoning_RM', 'MSZoning_C (all)', 'OverallCond', 'YearsOld', 
                  'Fireplaces', 'SaleCondition_Partial', 'Functional', 'SaleCondition_Normal', 
                  'HeatingQC', 'NearArtery', 'LotArea', 'MSSubClass_160', 'KitchenAbvGr']
    preprocess(trainDF, trainDF)
    
    GrLivArea_outliers = trainDF.sort_values(by = 'GrLivArea', ascending = False)[:2].index
    y_outliers = trainDF.iloc[GrLivArea_outliers, :].loc[:, 'SalePrice']

    y_train = trainDF['SalePrice']
    X_train = trainDF.drop(['SalePrice'], axis=1)

    X_outliers = trainDF.iloc[GrLivArea_outliers, :]
    dropOutliers(trainDF)

    X_outliers = standardize(X_outliers, X_train)
    X_train = standardize(X_train, X_train)
    y_train = np.log1p(y_train)

    lr = LinearRegression()
    lr.fit(X_train[predictors], y_train)
    log_preds = lr.predict(X_outliers[predictors])
    predictions = np.expm1(log_preds)
    print(f'Actual Price vs. Prediction')
    print(f'1) {y_outliers.iloc[0]} vs. {predictions[0]}')
    print(f'2) {y_outliers.iloc[1]} vs. {predictions[1]}')

    # replace AllSF with 0 (0 is mean since the df is standardized)
    X_outliers['AllSF'] = X_outliers['AllSF'].map(lambda v: 0)
    log_preds = lr.predict(X_outliers[predictors])
    predictions = np.expm1(log_preds)
    print(f'Actual Price vs. Prediction')
    print(f'1) {y_outliers.iloc[0]} vs. {predictions[0]}')
    print(f'2) {y_outliers.iloc[1]} vs. {predictions[1]}')


if __name__ == '__main__':
    # analysis1()
    # analysis2()
    # analysis3()
    analysis4()
    # analysis5()
    # analysis6()