# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 16:08:42 2019

@author: dhata
"""

##----- Importing required packages -----##
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV

##----- Reading the data -----##
data = pd.read_csv("D:\\Kaggle Datasets\\FIFA 19\\Fifa_data.csv")

##----- Dropping unnecessary columns -----##
data.drop(columns = ['ID','Photo','Club Logo','Work Rate','Body Type',
                     'Real Face','Loaned From','LS','ST','RS','LW','LF','CF','RF','RW','LAM',
                     'CAM','RAM','LM','LCM','CM','RCM','RM','LWB','LDM','CDM','RDM','RWB','LB','LCB',
                     'CB','RCB','RB'], inplace = True)

data.drop(data.columns[0],axis = 1, inplace = True)
data.drop(columns = "Flag",axis = 1, inplace = True)
##----- Getting the info of columns -----##
data.info()

##----- Dropping NA value rows -----##
data.dropna(axis = 0, how = "any", inplace = True)
data.info()

##----- Data Columns in data -----##
# =============================================================================
# ['Name', 'Age', 'Nationality', 'Overall', 'Potential', 'Club', 'Value',
#        'Wage', 'Special', 'Preferred Foot', 'International Reputation',
#        'Weak Foot', 'Skill Moves', 'Position', 'Jersey Number', 'Joined',
#        'Contract Valid Until', 'Height', 'Weight', 'Crossing', 'Finishing',
#        'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling', 'Curve',
#        'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration',
#        'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower',
#        'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression',
#        'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure',
#        'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling',
#        'GKKicking', 'GKPositioning', 'GKReflexes', 'Release Clause']
# 
# =============================================================================
##----- Cleaning value, wage and weight columns so that they are numeric -----##

def valClean(a):
    x = re.findall(r'(\d+\.\d+|\d+)([A-Z])', a) 
    if x[0][1] == "M":
        return(pd.to_numeric(x[0][0]))
    else:
        return(pd.to_numeric(x[0][0]) / 1000)
    
data.Value = data.Value.apply(lambda x : valClean(x)).astype("float")
data.Value = pd.to_numeric(data.Value)

def wageClean(a):
    x = re.findall(r'(\d+\.\d+|\d+)([A-Z])', a) 
    return(pd.to_numeric(x[0][0]))
    
data.Wage = data.Wage.apply(lambda x : wageClean(x))
data.Wage = pd.to_numeric(data.Wage)

def wtClean(x):
    return(re.findall(r"\d+\.\d+|\d+",x))[0]
    
data.Weight = data.Weight.apply(lambda x : wtClean(x))
data.Weight = pd.to_numeric(data.Weight)

##----- Vizualisations -----##
figure(num = None, figsize = (8,6), dpi = 100)
with plt.style.context(("ggplot")):
    plt.hist(data.loc[:,"Overall"], bins = 12, ec = "black", alpha = 0.8, color = "maroon")
    plt.xlabel("Overall Rating")
    plt.ylabel("Number of Players")
    plt.xticks([50,55,60,65,70,75,80,85,90,95,100])
    plt.show()

##----- Looking at the various positions and grouping them as Forwards, Mids, Defenders and GK -----##
data.Position.unique()
defenders = data.loc[data.Position.isin(['RCB', 'CB', 'LCB', 'LB', 'RB', 'GK', 'RWB', 'LWB'])]
figure(num = None, figsize = (8,6), dpi = 100)
sns.boxplot(x = defenders.Position, y = defenders.Overall)

mids = data.loc[data.Position.isin(['RCM', 'LCM', 'LDM', 'CAM', 'CDM', 'RM', 'LM', 'RDM', 'CM', 'RAM', 'LAM'])]
figure(num = None, figsize = (8,6), dpi = 100)
sns.boxplot(x = mids.Position, y = mids.Overall)

forwards = data.loc[~data.Position.isin(['RCM', 'LCM', 'LDM', 'CAM', 'CDM', 'RM', 'LM', 'RDM', 'CM', 'RAM', 'LAM',
                                    'RCB', 'CB', 'LCB', 'LB', 'RB', 'GK', 'RWB', 'LWB'])]

figure(num = None, figsize = (8,6), dpi = 100)
sns.boxplot(x = forwards.Position, y = forwards.Overall)

##----- Heatmap of players positions and abilities -----##
df_heatmap = data.loc[:,['Position','Crossing','Finishing','HeadingAccuracy','ShortPassing','Volleys','Dribbling',
                               'Curve','FKAccuracy','LongPassing','BallControl','Acceleration','SprintSpeed','Agility',
                                'Reactions','Balance','ShotPower','Jumping','Stamina','Strength','LongShots','Aggression',
                               'Interceptions','Positioning','Vision','Penalties','Composure','Marking','StandingTackle',
                               'SlidingTackle']].groupby("Position").mean()

df_heatmap = df_heatmap.loc[df_heatmap.index.isin(['RWB','RB','RCB','CB','LCB','LB','LWB'])]
figure(num=None, figsize=(15, 6), dpi=80, facecolor='w', edgecolor='k')
sns.heatmap(df_heatmap, annot= df_heatmap,cmap="RdYlGn",linewidths=0.3)

##----- Plotting Age vs Overall -----##
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
with plt.style.context(('ggplot')):
    plt.plot(sorted(data.Age.unique()),data.groupby("Age")["Overall"].mean(), color = "maroon")
    plt.grid(b=None, which='major', axis='both', color='w')
    plt.xlabel("Age of Player")
    plt.ylabel("Overall Rating of Player")
    plt.show()
    
##----- Finding Clubs with the most Talented Players -----##
talented = data.loc[:,["Club", "Overall"]]
talented = talented.loc[talented.Overall >= 85]
talented = talented.groupby("Club").count()
talented = talented.sort_values("Overall")
talentedSamp = talented.tail(10)
talentedSamp = talentedSamp.sort_values("Overall", ascending = False)
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
with plt.style.context(('ggplot')):
    plt.barh(talentedSamp.index, talentedSamp.Overall, color = "maroon")
    plt.xlabel("Number of Talented Players")
    plt.ylabel("Club")
    plt.show()
    
##----- Separating X and Y -----##
X = data.loc[:,["Age", "Overall", "Potential", "Wage"]]
Y = data.loc[:,"Value"]

##----- Splitting Data into Train and Test -----##
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

##----- Fitting Random Forest -----##
rf = RandomForestRegressor(n_estimators = 10, random_state = 42)
rf.fit(X_train, Y_train)
predictions = rf.predict(X_test)

##----- Calculating MSE -----##
mse = ((predictions - Y_test)**2).sum() / len(Y_test)
print("The MSE is:", mse)

##----- Calculating Accuracy -----##
error = abs(predictions - Y_test)
mape = 100 * (error / Y_test)
acc = 100 - mape.mean()
print("Model Accuracy is: ", round(acc,2))

##----- Cross Validation to find optimal Tuning Parameter -----##
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_features = ["auto", "sqrt"]
max_depth = [int(x) for x in np.linspace(start = 10, stop = 110, num = 11)]
max_depth.append(None)
min_sample_split = [2,5,10]
min_sample_leaf = [1,2,4]
random_grid = {"n_estimators": n_estimators,
               "max_features": max_features,
               "max_depth": max_depth,
               "min_samples_split": min_sample_split,
               "min_samples_leaf": min_sample_leaf}


## Make sure you change the n_iter from 100 to a lower number: at 100 takes 20 mins to run
rf_search = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 5,
                               verbose = 2, random_state = 42, n_jobs = -1)

rf_search.fit(X_train, Y_train)
rf_search.best_params_

##----- Fitting new Model -----##
predictions1 = rf.predict(X_test)

##----- Calculating MSE -----##
mse = ((predictions1 - Y_test)**2).sum() / len(Y_test)
print("The MSE is:", mse)

##----- Calculating Accuracy -----##
error1 = abs(predictions1 - Y_test)
mape1 = 100 * (error1 / Y_test)
acc1 = 100 - mape1.mean()
print("Model Accuracy is: ", round(acc1,2))
## Tuning didn't help