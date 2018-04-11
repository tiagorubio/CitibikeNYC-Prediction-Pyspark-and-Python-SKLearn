# Importing the libraries
import pandas as pd
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import linear_model
import matplotlib.pyplot as plt
#PrintFunction
def printing(name,y_test,predicted_test,targetComlumn):
    print("{} {}:".format(name,targetComlumn))
    #print(f'Out-of-bag R-2 score estimate: {rf.oob_score_}')
    print('Test data R-2 score:            :{:.2f}'.format(r2_score(y_test, predicted_test)))
    print('Test data Spearman correlation: :{:.2f}'.format(spearmanr(y_test, predicted_test)[0]))
    print('Test data Pearson correlation:  :{:.2f}'.format(pearsonr(y_test, predicted_test)[0]))
    print('Test data mean_absolute_error:  :{:.2f}'.format(mean_absolute_error(y_test,predicted_test)))
    y_test = y_test.values
    result = []
    for i in range(0,len(y_test)):
        c=1
        if abs(y_test[i]-predicted_test[i])> mean_absolute_error(y_test,predicted_test): c=0
        result.append(c)
    print('Predictions with in-the mean abs error: {:.2f}%'.format((sum(result)/len(result))*100))

def processDataSet(features):
    #Transforming the categorical columns flag columns
    cols_to_transform = ['Hour','Day','Month']
    features = pd.get_dummies(data= features, columns = cols_to_transform )
    try: #excluding columns
        del features['YearMonth']
        del features['Year']
        del features['DATE']
        del features['DateHour']
        del features['HOURLYPRSENTWEATHERTYPE']
        del features['Result']
        del features['InBike']
        del features['OutBike']
        del features['Hour']
        del features['Day']
        del features['Month']
        del features['StationID']
    except:
        pass
    return features

def regressorFunc(X_train,y_train,estimator = 25,option=1,random = 123):
    if option ==1:
        regressor = RandomForestRegressor(min_samples_split=2, n_estimators=estimator, min_samples_leaf=2,random_state = random)
        regressor.fit(X_train, y_train)
        return regressor
    if option ==2:
        regressor = DecisionTreeRegressor(random_state= random)
        regressor.fit(X_train, y_train)
        return regressor
    if option ==3:
        regressor = AdaBoostRegressor(DecisionTreeRegressor(), n_estimators=estimator,random_state= random)
        regressor.fit(X_train, y_train)
        return regressor
    if option ==4:
        regressor = GradientBoostingRegressor(random_state= random)
        regressor.fit(X_train, y_train)
        return regressor
    if option ==5:
        regressor = linear_model.LinearRegression()
        regressor.fit(X_train, y_train)
        return regressor
def regressionComparisson(targets,features,estimator = 25, targetComlumnLabel = "",numberOfPredictions = 1):
    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(features, targets, train_size=0.8,test_size=0.2, random_state=42)
    if numberOfPredictions == 1 or numberOfPredictions >= 6 :
        # Fitting Random Forest Regression to the dataset
        rf = regressorFunc(X_train,y_train,estimator = 25,option=1)
        predicted_test = rf.predict(X_test)
        # Printing Score
        printing('\nRandom Forest',y_test,predicted_test,targetComlumnLabel)
    if numberOfPredictions == 2 or numberOfPredictions >= 6 :
        # Fitting Decision Tree Regression to the dataset
        dt = regressorFunc(X_train,y_train,estimator = 25,option=2)
        predicted_test2= dt.predict(X_test)
        #Score
        printing('\nDecision Tree',y_test,predicted_test2,targetComlumnLabel)
    if numberOfPredictions == 3 or numberOfPredictions >= 6 :
        #Fitting Decision Tree Regression with AdaBoost to the dataset
        bdt = regressorFunc(X_train,y_train,estimator = 25,option=3)
        predicted_test3= bdt.predict(X_test)
        #Score
        printing('\nDecision Tree Regression with AdaBoost',y_test,predicted_test3,targetComlumnLabel)
    if numberOfPredictions == 4 or numberOfPredictions >= 6 :
        #Fitting Gradient Boosting Regressor to the dataset
        gb = regressorFunc(X_train,y_train,estimator = 25,option=4)
        predicted_test4= gb.predict(X_test)
        #Score
        printing('\nGradient Boosting Regressor',y_test,predicted_test4,targetComlumnLabel)
    if numberOfPredictions == 5 or numberOfPredictions >= 6 :
        #Fitting Linear Regression to the dataset
        lr = regressorFunc(X_train,y_train,estimator = 25,option=5)
        predicted_test5= lr.predict(X_test)
        #Score
        printing('\nLinear Regression',y_test,predicted_test5,targetComlumnLabel)

def main():
    # Importing the dataset
    Station = 359 # 517
    dataset = pd.read_csv('TrainingData_2016-2017.csv')
    dataset = dataset[dataset['StationID']== Station]
#    dataset = dataset[dataset['Month']<= 2]
    features = processDataSet(dataset)
    InBike = dataset['InBike']
    OutBike = dataset['OutBike']  
    #Compare regressions
    regressionComparisson(InBike,features, estimator = 1000,targetComlumnLabel = '- InBike',  numberOfPredictions = 6)
    regressionComparisson(OutBike,features,estimator = 1000,targetComlumnLabel = '- OutBike',numberOfPredictions = 6)
    

#ForeCasting 2018 based on 2016-2017 data
    #reloading the database
#    dataset = pd.read_csv('TrainingData_2016-2017.csv')
#    dataset = dataset[dataset['StationID']== Station]
#    #doing the regressing again only for jan and feb (forecast period)
#    dataset = dataset[dataset['Month']<= 2]
#    features = processDataSet(dataset)
#    InBike = dataset['InBike']
#    OutBike = dataset['OutBike']  
    foreCasterInBike = regressorFunc(features,InBike,estimator = 1000,option=1)
    foreCasterOutBike = regressorFunc(features,OutBike,estimator = 1000,option=1)    
    datasetForeCast = pd.read_csv('TestData_2018Q1.csv')
    datasetForeCast = datasetForeCast[datasetForeCast['StationID']== Station]
    InBikeForeCast = datasetForeCast['InBike']
    OutBikeForeCast = datasetForeCast['OutBike']
    featuresForeCast = processDataSet(datasetForeCast)
    try:
        featuresForeCast["Month_3"] = 0
        featuresForeCast["Month_4"] = 0
        featuresForeCast["Month_5"] = 0
        featuresForeCast["Month_6"] = 0
        featuresForeCast["Month_7"] = 0
        featuresForeCast["Month_8"] = 0
        featuresForeCast["Month_9"] = 0
        featuresForeCast["Month_10"] = 0
        featuresForeCast["Month_11"] = 0
        featuresForeCast["Month_12"] = 0
    except:
        pass
    y_forecastedInBike = foreCasterInBike.predict(featuresForeCast)
    y_forecastedOutBike = foreCasterOutBike.predict(featuresForeCast)
    
    printing('Forecast 2018 Based on 2016-2017 data',InBikeForeCast,y_forecastedInBike,'- InBike')
    printing('Forecast 2018 Based on 2016-2017 data',OutBikeForeCast,y_forecastedOutBike,'- OutBike')

#Saving the resultes
    featuresForeCast['InBike'] = datasetForeCast['InBike']
    featuresForeCast["forecastedValueInBike"] = y_forecastedInBike
    featuresForeCast['OutBike'] = datasetForeCast['OutBike']
    featuresForeCast["forecastedValueOutBike"] = y_forecastedOutBike
    featuresForeCast['Hour'] = datasetForeCast['Hour']    
    featuresForeCast['Day'] = datasetForeCast['Day']
    featuresForeCast['Month'] = datasetForeCast['Month'] 
    featuresForeCast['DateHour'] = datasetForeCast['DateHour']
    try:
        featuresForeCast.to_csv("ForeCast2018.csv")
    except:
        pass
#pltoting
    plt.scatter(InBikeForeCast, y_forecastedInBike, color = 'red')
    plt.plot(y_forecastedInBike, y_forecastedInBike, color = 'blue')
    plt.title('RandomForest Regression - InBike Jan-Fev 2018')
    plt.xlabel('True Value')
    plt.ylabel('Prediction')
    plt.show()  
    
    plt.scatter(OutBikeForeCast, y_forecastedOutBike, color = 'red')
    plt.plot(y_forecastedOutBike, y_forecastedOutBike, color = 'blue')
    plt.title('RandomForest Regression - OutBike Jan-Fev 2018')
    plt.xlabel('True Value')
    plt.ylabel('Prediction')
    plt.show()  
    
if __name__ == "__main__":
    # calling main function
    main()
