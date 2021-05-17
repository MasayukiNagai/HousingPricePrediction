import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.linear_model import LinearRegression, TweedieRegressor, RidgeCV, LassoCV
from sklearn.ensemble import GradientBoostingRegressor
# import xgboost as xgb
# import lightgbm as lgb
from utils import getNumericAttrs


train_file = './data/train.csv'
test_file = './data/test.csv'
trainDF = pd.read_csv(train_file)
testDF = pd.read_csv(test_file)
seed = 17


def transformData(trainDF):
    preprocess(trainDF, trainDF)
    dropOutliers(trainDF)  # Only for training

    y_train = trainDF['SalePrice']
    X_train = trainDF.drop(['SalePrice'], axis=1)

    X_train = standardize(X_train, X_train)
    y_train = np.log1p(y_train)
    return X_train, y_train


def preprocess(targetDF, sourceDF):
    handleMissingValues(targetDF, sourceDF)
    recodeValues(targetDF)
    createDummies(targetDF)
    featureEngineering(targetDF)
    dropUnusedCols(targetDF)


def handleMissingValues(targetDF, sourceDF):
    # replace with 0
    zero_cols = ['Utilities', 'KitchenQual', 'Functional', 'FireplaceQu', 'BsmtQual', 
                 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 
                 'BsmtHalfBath', 'BsmtFullBath', 'BsmtFinSF1', 'BsmtFinSF2',
                 'BsmtUnfSF', 'TotalBsmtSF', 'GarageCars', 'GarageArea', 'GarageYrBlt', 
                 'GarageFinish', 'GarageQual', 'GarageCond', 'Fence']
    for col in zero_cols:
        targetDF[col] = targetDF[col].fillna(0)

    # replace with median
    median_cols = ['MasVnrArea']  # LotFrontage
    for col in median_cols:
        targetDF[col] = targetDF[col].fillna(sourceDF[col].median())


def recodeValues(targetDF):
    targetDF['LotShape'].replace({"Reg": 3, "IR1": 2, "IR2": 1, "IR3": 0}, inplace=True)
    targetDF['LandContour'].replace({"Lvl": 3, "Bnk": 2, "HLS": 1, "Low": 0}, inplace=True)
    targetDF['Utilities'].replace({"AllPub": 3, "NoSewr": 2, "NoSeWa": 1, "ELO": 0}, inplace=True)
    targetDF['LandSlope'].replace({"Gtl": 2, "Mod": 1, "Sev": 0}, inplace=True)
    targetDF['HouseStyle'].replace({"1Story": 1, "1.5Fin": 1.25, "1.5Unf": 1.5, "2Story": 2,
                                    "2.5Fin": 2.5, "2.5Unf": 2.25, "SFoyer": 2, "SLvl": 2}, inplace=True)
    targetDF['ExterQual'].replace({"Ex": 4, "Gd": 3, "TA": 2, "Fa": 1, "Po": 0}, inplace=True)
    targetDF['ExterCond'].replace({"Ex": 4, "Gd": 3, "TA": 2, "Fa": 1, "Po": 0}, inplace=True)
    targetDF['BsmtQual'].replace({"Ex": 4, "Gd": 3, "TA": 2, "Fa": 1, "Po": 0}, inplace=True)
    targetDF['BsmtCond'].replace({"Ex": 4, "Gd": 3, "TA": 2, "Fa": 1, "Po": 0}, inplace=True)
    targetDF['BsmtExposure'].replace({"Gd": 3, "Av": 2, "Mn": 1, "No": 0}, inplace=True)
    targetDF['BsmtFinType1'].replace({"GLQ": 5, "ALQ": 4, "BLQ": 3, "Rec": 2, "LwQ": 1, "Unf": 0}, inplace=True)
    targetDF['BsmtFinType2'].replace({"GLQ": 5, "ALQ": 4, "BLQ": 3, "Rec": 2, "LwQ": 1, "Unf": 0}, inplace=True)
    targetDF['HeatingQC'].replace({"Ex": 4, "Gd": 3, "TA": 2, "Fa": 1, "Po": 0}, inplace=True)
    targetDF['CentralAir'].replace({"Y": 1, "N": 0}, inplace=True)
    targetDF['Electrical'].replace({"SBrkr": 3, "FuseA": 2, "FuseF": 1, "FuseP": 0, "Mix": 0}, inplace=True)
    targetDF['KitchenQual'].replace({"Ex": 4, "Gd": 3, "TA": 2, "Fa": 1, "Po": 0}, inplace=True)
    targetDF['Functional'].replace({"Sal": 7, "Sev": 6, "Maj2": 5, "Maj1": 4, 
                                    "Mod": 3, "Min2": 2, "Min1": 1, "Typ": 0}, inplace=True)
    targetDF['FireplaceQu'].replace({"Ex": 4, "Gd": 3, "TA": 2, "Fa": 1, "Po": 0}, inplace=True)
    targetDF['GarageFinish'].replace({"Fin": 2, "RFn": 1, "Unf": 0}, inplace=True)
    targetDF['GarageQual'].replace({"Ex": 4, "Gd": 3, "TA": 2, "Fa": 1, "Po": 0}, inplace=True)
    targetDF['GarageCond'].replace({"Ex": 4, "Gd": 3, "TA": 2, "Fa": 1, "Po": 0}, inplace=True)
    targetDF['PavedDrive'].replace({"Y": 2, "P": 1, "N": 0}, inplace=True)
    targetDF['Fence'].replace({"GdPrv": 3, "MnPrv": 2, "GdWo": 1, "MnWw": 0}, inplace=True)


def createDummies(targetDF):
    cols = ['MSSubClass', 'MSZoning', 'Street', 'Neighborhood', 'SaleCondition', 'Exterior1st']
    newDF = pd.get_dummies(targetDF, columns=cols)
    newColumns = list(set(newDF.columns) - set(targetDF.columns))
    targetDF.loc[:, newColumns] = newDF.loc[:, newColumns]


def featureEngineering(targetDF):
    ## Indicator variable for adjacent to or within 200 of either arterial street, feeder street, or railroad
    artery = ['Artery', 'Feedr', 'RRNn', 'RRAn', 'RRNe', 'RRAe']
    targetDF['NearArtery'] = targetDF['Condition1'].map(lambda v: 1 if v in artery else 0)
    
    ## Recode YearBuilt to YearsOld and DecadesOld 
    newestYearBuilt = 2010
    targetDF['YearsOld'] = targetDF['YearBuilt'].map(lambda v: newestYearBuilt - v)
    targetDF.loc[:, 'DecadesOld'] = targetDF.loc[:, 'YearsOld'].map(lambda v: v // 10)
    
    ## Create new features as a combination of existing features
    # BEGIN: from https://www.kaggle.com/juliencs/a-study-on-regression-applied-to-the-ames-dataset
    # EXPLANATION: New featureas by combining existing features
    targetDF["AllSF"] = targetDF["GrLivArea"] + targetDF["TotalBsmtSF"]
    # END: from from https://www.kaggle.com/juliencs/a-study-on-regression-applied-to-the-ames-dataset
    
    ## Polynomials
    # Idea: from https://www.kaggle.com/juliencs/a-study-on-regression-applied-to-the-ames-dataset
    targetDF['OverallQual-s2'] = targetDF['OverallQual'] ** 2
    targetDF['OverallQual-s3'] = targetDF['OverallQual'] ** 3
    targetDF['OverallQual-sq'] = targetDF['OverallQual'] ** 0.5
    targetDF['AllSF-s2'] = targetDF['AllSF'] ** 2
    targetDF['AllSF-s3'] = targetDF['AllSF'] ** 3
    targetDF['AllSF-sq'] = targetDF['AllSF'] ** 0.5
    targetDF['AllSF-log'] = np.log1p(targetDF['AllSF'])
    targetDF['GrLivArea-s2'] = targetDF['GrLivArea'] ** 2
    targetDF['GrLivArea-s3'] = targetDF['GrLivArea'] ** 3
    targetDF['GrLivArea-sq'] = targetDF['GrLivArea'] ** 0.5
    targetDF['GrLivArea-log'] = np.log1p(targetDF['GrLivArea'])
    targetDF['TotalBsmtSF-s2'] = targetDF['TotalBsmtSF'] ** 2
    targetDF['TotalBsmtSF-s3'] = targetDF['TotalBsmtSF'] ** 3
    targetDF['TotalBsmtSF-sq'] = targetDF['TotalBsmtSF'] ** 0.5
    targetDF['TotalBsmtSF-log'] = np.log1p(targetDF['TotalBsmtSF'])
    targetDF['GarageCars-s2'] = targetDF['GarageCars'] ** 2
    targetDF['GarageCars-s3'] = targetDF['GarageCars'] ** 3
    targetDF['GarageCars-sq'] = targetDF['GarageCars'] ** 0.5
    targetDF['GarageCars-log'] = np.log1p(targetDF['GarageCars'])
    targetDF['BsmtUnfSF-s2'] = targetDF['BsmtUnfSF'] ** 2
    targetDF['BsmtUnfSF-s3'] = targetDF['BsmtUnfSF'] ** 3
    targetDF['BsmtUnfSF-sq'] = targetDF['BsmtUnfSF'] ** 0.5   
    targetDF['BsmtUnfSF-log'] = np.log1p(targetDF['BsmtUnfSF'])
    targetDF['YearRemodAdd-s2'] = targetDF['YearRemodAdd'] ** 2
    targetDF['YearRemodAdd-s3'] = targetDF['YearRemodAdd'] ** 3
    targetDF['YearRemodAdd-sq'] = targetDF['YearRemodAdd'] ** 0.5
    targetDF['YearRemodAdd-log'] = np.log1p(targetDF['YearRemodAdd'])
    targetDF['YearsOld-s2'] = targetDF['YearsOld'] ** 2
    targetDF['YearsOld-s3'] = targetDF['YearsOld'] ** 3
    targetDF['YearsOld-sq'] = targetDF['YearsOld'] ** 0.5
    targetDF['YearsOld-log'] = np.log1p(targetDF['YearsOld'])
    targetDF['OverallCond-s2'] = targetDF['OverallCond'] ** 2
    targetDF['OverallCond-s3'] = targetDF['OverallCond'] ** 3
    targetDF['OverallCond-sq'] = targetDF['OverallCond'] ** 0.5
    targetDF['OverallCond-log'] = np.log1p(targetDF['OverallCond'])

    '''
    ## Log transformation
    # BEGIN: from https://www.kaggle.com/juliencs/a-study-on-regression-applied-to-the-ames-dataset
    # EXPLANATION: Log transformation of the skewed numberical features to lessen impact of outliers
    numerical_features = targetDF.select_dtypes(exclude = ["object"]).columns
    numerical_features = numerical_features.drop("SalePrice")
    target_num = targetDF[numerical_features]
    skewness = target_num.apply(lambda x: stats.skew(x))  # ATTENTION: this should use sourceDF after preprocessed
    skewness = skewness[abs(skewness) > 1.0]
    print(str(skewness.shape[0]) + " skewed numerical features to log transform")
    skewed_features = skewness.index
    targetDF[skewed_features] = np.log1p(targetDF[skewed_features])
    # END from: https://www.kaggle.com/juliencs/a-study-on-regression-applied-to-the-ames-dataset
    '''

def dropUnusedCols(targetDF):
    replaced_cols = ['MSSubClass', 'MSZoning', 'Street', 'Condition1', 'Condition2', 
                     'SaleCondition', 'Exterior1st', 'YearBuilt']
    missing_cols = ['Alley', 'PoolQC', 'MiscFeature', 'FireplaceQu', 'LotFrontage']
    unused_cols = ['LotConfig', 'Neighborhood', 'BldgType', 'RoofStyle', 'RoofMatl', 
                   'Exterior2nd', 'MasVnrType', 'Foundation', 'Heating', 'GarageType', 
                   'SaleType']
    drop_cols = missing_cols + replaced_cols + unused_cols
    targetDF.drop(columns=drop_cols, inplace=True)


def dropOutliers(targetDF):
    # Delete an obs with Electrical missing
    targetDF.drop(targetDF[targetDF['Electrical'].isna()].index, inplace=True)
    # Delete two outliers in terms of GrLivArea 
    GrLivArea_outliers = targetDF.sort_values(by = 'GrLivArea', ascending = False)[:2].index
    targetDF.drop(GrLivArea_outliers, inplace=True)


def handleOutliers(testDF):
    # Replace too large AllSF with mean value (0 when standardized)
    testDF['AllSF'] = testDF['AllSF'].map(lambda v: v  if v < 4 else 0)


def standardize(targetDF, sourceDF, cols=None):
    if cols:
        mean = sourceDF.loc[:, cols].mean()
        std = sourceDF.loc[:, cols].std()
        standardizedDF = (targetDF.loc[:, cols] - mean)/std
    else:
        mean = sourceDF.mean()
        std = sourceDF.std()
        standardizedDF = (targetDF - mean)/std
    return standardizedDF


# BEGIN: from https://www.kaggle.com/juliencs/a-study-on-regression-applied-to-the-ames-dataset
# EXPLANATION: Define error measure for official scoring: RMSE
def rmse_cv(model, X, y, k=10):
    kf = KFold(k, shuffle=True, random_state=seed).get_n_splits(X.values)
    rmse = np.sqrt(-cross_val_score(model, X, y, cv=kf, scoring="neg_mean_squared_error", n_jobs=-1))
    return rmse
# END: from https://www.kaggle.com/juliencs/a-study-on-regression-applied-to-the-ames-dataset


def testModel(model, predictors):
    trainDF = pd.read_csv(train_file)
    X, y = transformData(trainDF)
    X = X.loc[:, predictors]
    # print(X.head())
    cvScores = rmse_cv(model, X, y)
    print("RMSE: {:.5f} ({:.4f})".format(cvScores.mean(), cvScores.std()))


def forwardSelection(model, numPredictors=20):
    print('----- Forward Selction -----')
    trainDF = pd.read_csv(train_file)
    X, y = transformData(trainDF)
    numericAttrs = getNumericAttrs(X)
    candidateAttrs = list(numericAttrs)
    candidateAttrs.remove('Id')
    print(f'{len(candidateAttrs)} candidate attributes')
    predictors = []  # ['OverallQual', 'AllSF', 'GarageCars']
    for pred in predictors:
        candidateAttrs.remove(pred)
    bestScore = 1
    bestAttr = ''
    scores = [0.3]  # arbitrary initial value
    while len(predictors) < numPredictors:
        for attr in candidateAttrs:
            tmp_predictors = predictors.copy()
            tmp_predictors.append(attr)       
            score = rmse_cv(model, X[tmp_predictors], y).mean()
            if score < bestScore:
                bestScore = score
                bestAttr = attr
        if bestAttr in predictors:
            break
        predictors.append(bestAttr)
        candidateAttrs.remove(bestAttr)
        improved_rate = (scores[-1] - bestScore)/scores[-1] * 100
        print(f'{len(predictors)}) {predictors}: {bestScore:.5f} ({improved_rate:.3f}%)')
        scores.append(round(bestScore, 6))
    print(f'The change in scores: {scores}')
    showCoefs(model, X[predictors], y)
    print('----- Selection complete -----')
    return predictors


def findBestModelforLR(numPredictors=16):
    lr = LinearRegression()
    predictors = forwardSelection(lr, numPredictors)
    trainDF = pd.read_csv(train_file)
    X_train, y_train = transformData(trainDF) 
    testBySplitTrainData(lr, predictors)
    showCoefs(lr, X_train[predictors], y_train)


def findBestModelforRidge(numPredictors=16):
    ridge = RidgeCV(alphas = [0.1, 0.3, 0.6, 1, 3, 6, 10, 30, 60, 100, 300, 600])
    predictors = forwardSelection(ridge, numPredictors)
    findBestAlphaforRidge(predictors)


def findBestAlphaforRidge(predictors):
    # BEGIN: from https://www.kaggle.com/juliencs/a-study-on-regression-applied-to-the-ames-dataset
    # EXPLANATION: This tries to find the best alpha value for the Ridge model
    trainDF = pd.read_csv(train_file)
    X_train, y_train = transformData(trainDF)
    ridge = RidgeCV(alphas = [0.1, 0.3, 0.6, 1, 3, 6, 10, 30, 60, 100, 300, 600])
    ridge.fit(X_train, y_train)
    alpha = ridge.alpha_
    print("Best alpha :", alpha)
    ridge = RidgeCV(alphas=[alpha])  # ADDED
    scores = rmse_cv(ridge, X_train[predictors], y_train)  # ADDED
    print(f'RMSE with alpha = {alpha}: {scores.mean():.5f} ({scores.std():.4f})')  # ADDED

    print("Try again for more precision with alphas centered around " + str(alpha))
    ridge = RidgeCV(alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, alpha * .85, 
                              alpha * .9, alpha * .95, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15,
                              alpha * 1.25, alpha * 1.3, alpha * 1.35, alpha * 1.4], 
                    cv = 10)
    ridge.fit(X_train, y_train)
    alpha = ridge.alpha_
    print("Best alpha :", alpha)
    ridge = RidgeCV(alphas=[alpha])  # ADDED
    scores = rmse_cv(ridge, X_train[predictors], y_train)  # ADDED
    print(f'RMSE alpha = {alpha} : {np.mean(scores):.5f} ({np.std(scores):.4f})')  # ADDED
    # END: from https://www.kaggle.com/juliencs/a-study-on-regression-applied-to-the-ames-dataset

    testBySplitTrainData(ridge, predictors)
    showCoefs(ridge, X_train[predictors], y_train)


def findBestModelforLasso(numPredictors=16):
    lasso = LassoCV(alphas = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 
                              0.3, 0.6, 1], 
                    max_iter = 10000, cv = 10)
    predictors = forwardSelection(lasso, numPredictors)
    findBestAlphaforLasso(predictors)


def findBestAlphaforLasso(predictors):
    # BEGIN: from https://www.kaggle.com/juliencs/a-study-on-regression-applied-to-the-ames-dataset
    # EXPLANATION: This tried to find the best alpha value for the Lasso Model
    trainDF = pd.read_csv(train_file)
    X_train, y_train = transformData(trainDF)
    lasso = LassoCV(alphas = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 
                              0.3, 0.6, 1], 
                    max_iter = 10000, cv = 10)
    lasso.fit(X_train, y_train)
    alpha = lasso.alpha_
    print("Best alpha :", alpha)
    lasso = LassoCV(alphas=[alpha])  # ADDED
    scores = rmse_cv(lasso, X_train[predictors], y_train)  # ADDED
    print(f'RMSE with alpha = {alpha}: {np.mean(scores):.5f} ({np.std(scores):.4f})')  # ADDED

    print("Try again for more precision with alphas centered around " + str(alpha))
    lasso = LassoCV(alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, 
                            alpha * .85, alpha * .9, alpha * .95, alpha, alpha * 1.05, 
                            alpha * 1.1, alpha * 1.15, alpha * 1.25, alpha * 1.3, alpha * 1.35, 
                            alpha * 1.4], 
                    max_iter = 10000, cv = 10)
    lasso.fit(X_train, y_train)
    alpha = lasso.alpha_
    print("Best alpha :", alpha)
    lasso = LassoCV(alphas=[alpha])  # ADDED
    scores = rmse_cv(lasso, X_train[predictors], y_train)  # ADDED
    print(f'RMSE with alpha = {alpha}: {np.mean(scores):.5f} ({np.std(scores):.4f})')  # ADDED
    # END: from https://www.kaggle.com/juliencs/a-study-on-regression-applied-to-the-ames-dataset
    
    testBySplitTrainData(lasso, predictors)
    showCoefs(lasso, X_train[predictors], y_train)


def findBestModelforGBR(numPredictors=16):
    max_depth = 5
    # maybe loss, learning_rate, min_samples_X as well
    params = {'n_estimators': 1000, 'learning_rate': 0.01, 
              'max_depth':max_depth, 'max_features':'sqrt',
              'min_samples_leaf':15, 'loss':'huber', 'random_state': seed}
    GBR = GradientBoostingRegressor(**params)
    # predictors = forwardSelection(GBR, numPredictors)
    predictors = ['OverallQual', 'AllSF', 'GarageCars', 'BsmtUnfSF', 'YearRemodAdd-s3', 
                  'MSZoning_RM', 'MSZoning_C (all)', 'OverallCond-sq', 'YearsOld-sq', 
                  'Fireplaces', 'Neighborhood_Crawfor', 'MSSubClass_160', 'SaleCondition_Abnorml', 
                  'LotArea', 'KitchenQual', 'Functional']
    testBySplitTrainData(GBR, predictors)
    trainDF = pd.read_csv(train_file)
    X_train, y_train = transformData(trainDF)
    analyzeGBR(GBR, params, X_train[predictors], y_train)


def analyzeGBR(GBR, params, X_test, y_test):
    # BEGIN: from https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_regression.html#sphx-glr-auto-examples-ensemble-plot-gradient-boosting-regression-py
    # EXPLANATION: Evaluate GBR by plotting the change in the score and feature importance
    GBR.fit(X_test, y_test)
    mse = mean_squared_error(y_test, GBR.predict(X_test))**0.5
    print(f"The mean squared error (MSE) on test set: {mse:.4f}")
    test_score = np.zeros((params['n_estimators'],), dtype=np.float64)
    for i, y_pred in enumerate(GBR.staged_predict(X_test)):
        test_score[i] = GBR.loss_(y_test, y_pred)

    fig = plt.figure(figsize=(6, 6))
    plt.subplot(1, 1, 1)
    plt.title('Deviance')
    plt.plot(np.arange(params['n_estimators']) + 1, GBR.train_score_, 'b-',
             label='Training Set Deviance')
    plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
             label='Test Set Deviance')
    plt.legend(loc='upper right')
    plt.xlabel('Boosting Iterations')
    plt.ylabel('Deviance')
    fig.tight_layout()
    plt.show()

    feature_importance = GBR.feature_importances_
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    fig = plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, np.array(X_test.columns)[sorted_idx])
    plt.title('Feature Importance (MDI)')

    result = permutation_importance(GBR, X_test, y_test, n_repeats=10,
                                    random_state=42, n_jobs=2)
    sorted_idx = result.importances_mean.argsort()
    plt.subplot(1, 2, 2)
    plt.boxplot(result.importances[sorted_idx].T,
                vert=False, labels=np.array(X_test.columns)[sorted_idx])
    plt.title("Permutation Importance (test set)")
    fig.tight_layout()
    plt.show()
    # END: from https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_regression.html#sphx-glr-auto-examples-ensemble-plot-gradient-boosting-regression-py


def showCoefs(model, X, y):
    # BEGIN: from https://www.kaggle.com/juliencs/a-study-on-regression-applied-to-the-ames-dataset
    # Plot important coefficients
    model.fit(X, y)
    coefs = pd.Series(model.coef_, index = X.columns)
    if len(coefs) > 20:
        imp_coefs = pd.concat([coefs.sort_values().head(10),
                            coefs.sort_values().tail(10)])
        imp_coefs.plot(kind = "barh")
    else:
        coefs.sort_values().plot(kind='barh')
    plt.title("Coefficients in the Model")
    plt.tight_layout()
    plt.show()
    # END: from https://www.kaggle.com/juliencs/a-study-on-regression-applied-to-the-ames-dataset


def isValidPredictions(predictions: pd.Series):
    invalid_preds = predictions.map(lambda v: v < 0 or np.isinf(v))
    assert not invalid_preds.any()
    print('No zero, negative, or inf numbers found in the predictions')


def testBySplitTrainData(model, predictors):
    print('----- Test By Splitting Train Data -----')
    originalDF = pd.read_csv("data/train.csv")
    preprocess(originalDF, originalDF)
    dropOutliers(originalDF)
    
    X = originalDF[predictors]
    y = np.log1p(originalDF['SalePrice'])

    X = standardize(X, X)
    
    seeds = [0, 4, 11, 17, 29]
    scores = []
    for seed in seeds:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed)
        model.fit(X_train, y_train)
        y_preds = model.predict(X_test)
        rmse = mean_squared_error(y_test, y_preds)**0.5
        scores.append(rmse)
    print("RMSE on Test: {:.5f} ({:.4f})".format(np.mean(scores), np.std(scores)))


def makeSubmission(model, predictors, outfile):
    # Read files
    trainDF = pd.read_csv(train_file)
    testDF = pd.read_csv(test_file)
    
    # Delete outliers from trainDF so that the model does not get disturbed
    dropOutliers(trainDF)
    y_train = trainDF['SalePrice'].values
    X_train = trainDF.drop(["SalePrice", "Id"], axis=1)
    testIds = testDF['Id']
    X_test = testDF.drop(['Id'], axis=1)

    # Concatenate train and test data to preprocess together
    n_train = trainDF.shape[0]
    all_data = pd.concat(objs=[X_train, X_test], axis=0)

    preprocess(all_data, all_data)

    # MSSubClass_150 only has one record 
    # thus needs to be removed for standardization
    all_data.drop(['MSSubClass_150'], axis=1, inplace=True)

    # Split the data back into train and test again
    X_train = all_data.iloc[:n_train, :].copy()
    X_test = all_data.iloc[n_train:, :].copy()
    
    # Standardization
    X_test = standardize(X_test, X_train)
    X_train = standardize(X_train, X_train)

    # Replace some outlier values in the test set
    handleOutliers(X_train)

    # Log transform SalePrice and fit a model
    y_train = np.log1p(y_train)
    score = rmse_cv(model, X_train[predictors], y_train).mean()
    print(f'RMSE: {score}')
    model.fit(X_train[predictors], y_train)
    
    # Make a prediction and inverse the results 
    log_preds = model.predict(X_test[predictors])
    predictions = np.expm1(log_preds)

    isValidPredictions(pd.Series(predictions))

    submissionDF = pd.DataFrame({
        "Id": testIds,
        "SalePrice": predictions
    })

    submissionDF.to_csv(outfile, index=False)


models = {'LinearRegression': LinearRegression(),
          'TweedieRegressor': TweedieRegressor(power=1, alpha=0.5, link='log'),
          'Ridge': RidgeCV(alphas=[240]),
          'Lasso': LassoCV(alphas=[0.00255]),
          'GBR': GradientBoostingRegressor(n_estimators=1000, learning_rate=0.01, 
                                           max_depth=5,  
                                           min_samples_leaf=15, loss='huber', random_state=seed)}


def main():
    # predictors = ['OverallQual', 'AllSF', 'YearRemodAdd', 'BsmtUnfSF', 'GarageCars', 
    #               'SaleCondition_Partial', 'MSZoning_RM', 'OverallCond', 'YearsOld', 
    #               'Fireplaces', 'SaleCondition_Abnorml', 'LandSlope', 'KitchenQual', 
    #               'LotShape', 'MSZoning_C (all)', 'NearArtery']

    # tr = TweedieRegressor(power=1, alpha=0.5, link='log')
    # ridge = RidgeCV(alphas = [0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6, 10, 30, 60])
    
    # testModel(lr, predictors)
    
    ### LinearRegression
    # lr = LinearRegression()
    ## 16
    # predictors = ['AllSF-sq', 'OverallQual', 'YearsOld', 'OverallCond', 'BsmtUnfSF', 
    #               'GarageCars', 'MSZoning_RM', 'YearsOld-log', 'MSZoning_C (all)', 'Fireplaces', 
    #               'OverallQual-s3', 'NearArtery', 'Functional', 'TotalBsmtSF-sq', 'KitchenAbvGr', 
    #               'SaleCondition_Abnorml']
    ## 30
    # predictors = ['AllSF-sq', 'OverallQual', 'YearsOld', 'OverallCond', 'BsmtUnfSF', 
    #               'GarageCars', 'MSZoning_RM', 'YearsOld-log', 'MSZoning_C (all)', 'Fireplaces', 
    #               'OverallQual-s3', 'NearArtery', 'Functional', 'TotalBsmtSF-sq', 'KitchenAbvGr', 
    #               'SaleCondition_Abnorml', 'LotArea', 'Neighborhood_Crawfor', 'MSSubClass_160', 'CentralAir', 
    #               'BsmtExposure', 'KitchenQual', 'ScreenPorch', 'Neighborhood_BrkSide', 'Exterior1st_BrkFace', 
    #               'Neighborhood_Blmngtn', 'Neighborhood_StoneBr', 'Neighborhood_MeadowV', 'Neighborhood_NoRidge', 'WoodDeckSF']
    # outfile = './data/lr30.csv'

    ### Ridge
    # ridge = RidgeCV(alphas = [270])
    ## 40
    # predictors = ['AllSF-sq', 'OverallQual', 'YearsOld', 'OverallCond', 'BsmtUnfSF', 
    #               'GarageCars', 'MSZoning_RM', 'YearsOld-log', 'MSZoning_C (all)', 'Fireplaces', 
    #               'OverallQual-s3', 'NearArtery', 'Functional', 'TotalBsmtSF-sq', 'KitchenAbvGr', 
    #               'SaleCondition_Abnorml', 'LotArea', 'Neighborhood_Crawfor', 'MSSubClass_160', 'CentralAir', 
    #               'BsmtExposure', 'KitchenQual', 'ScreenPorch', 'Neighborhood_BrkSide', 'Neighborhood_Blmngtn', 
    #               'Exterior1st_BrkFace', 'Neighborhood_MeadowV', 'Neighborhood_StoneBr', 'Neighborhood_NoRidge', 'WoodDeckSF', 
    #               'SaleCondition_Partial', 'HeatingQC', 'GarageQual', 'Neighborhood_ClearCr', 'HalfBath', 
    #               'Neighborhood_Somerst', 'Neighborhood_NridgHt', 'PoolArea', 'MSSubClass_30', 'BsmtFinSF2']
    # outfile = './data/ridge40.csv'

    ### Lasso
    lasso = LassoCV(alphas=[0.00255])
    # 50
    predictors = ['AllSF-sq', 'OverallQual', 'YearsOld', 'OverallCond', 'BsmtUnfSF', 
                  'GarageCars', 'MSZoning_RM', 'YearsOld-log', 'MSZoning_C (all)', 'Fireplaces', 
                  'OverallQual-s3', 'LotArea', 'Neighborhood_Crawfor', 'Functional', 'NearArtery', 
                  'TotalBsmtSF-sq', 'KitchenAbvGr', 'SaleCondition_Abnorml', 'MSSubClass_160', 'CentralAir', 
                  'BsmtExposure', 'KitchenQual', 'ScreenPorch', 'Neighborhood_BrkSide', 'Exterior1st_BrkFace', 
                  'Neighborhood_Blmngtn', 'Neighborhood_MeadowV', 'Neighborhood_StoneBr', 'HeatingQC', 'SaleCondition_Partial', 
                  'WoodDeckSF', 'Neighborhood_NoRidge', 'Neighborhood_Edwards', 'GarageQual', 'Neighborhood_ClearCr', 
                  'BsmtQual', 'Neighborhood_Somerst', 'Neighborhood_NridgHt', 'PoolArea', 'HalfBath', 
                  'MSSubClass_30', 'BsmtFinSF2', 'YearRemodAdd-log', 'OverallCond-s3', 'Neighborhood_Veenker', 
                  'MSSubClass_120', 'BsmtFinType1', 'GarageCars-log', '3SsnPorch', 'Exterior1st_MetalSd']
    outfile = './data/lasso.csv'
    
    makeSubmission(lasso, predictors, outfile)


if __name__ == '__main__':
    main()
    # findBestModelforLR(30)
    # findBestModelforRidge(40)
    # findBestModelforLasso(50)
    # findBestModelforGBR()